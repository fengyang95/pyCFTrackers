"""
Python re-implementation of "High-speed Tracking with Multi-kernel Correlation Filters"
@inproceedings{tang2018high,
  title={High-speed tracking with multi-kernel correlation filters},
  author={Tang, Ming and Yu, Bin and Zhang, Fan and Wang, Jinqiao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4874--4883},
  year={2018}
}
"""
import numpy as np
import cv2
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature
from .config import mkcf_up_config


class MKCFup(BaseCF):
    def __init__(self,config=mkcf_up_config.MKCFupOTB50Config()):
        super(MKCFup).__init__()
        self.gap=config.gap
        self.lr_cn_color = config.lr_cn_color
        self.lr_cn_gray = config.lr_cn_gray
        self.lr_hog_color = config.lr_hog_color
        self.lr_hog_gray = config.lr_hog_gray
        self.num_compressed_dim_cn=config.num_compressed_dim_cn
        self.num_compressed_dim_hog=config.num_compressed_dim_hog

        self.padding=config.padding
        self.output_sigma_factor=config.output_sigma_factor
        self.scale_sigma_factor=config.scale_sigma_factor
        self.lambda_ = config.lambda_
        self.interp_factor=config.interp_factor

        self.cn_sigma_color=config.cn_sigma_color
        self.hog_sigma_color=config.hog_sigma_color

        self.cn_sigma_gray=config.cn_sigma_gray
        self.hog_sigma_gray=config.hog_sigma_gray
        self.refinement_iterations=config.refinement_iterations
        self.translation_model_max_area=config.translation_model_max_area
        self.interpolate_response=config.interpolate_response
        self.num_of_scales=config.num_of_scales
        self.num_of_interp_scales=config.num_of_interp_scales
        self.scale_model_factor=config.scale_model_factor
        self.scale_step=config.scale_step
        self.scale_model_max_area=config.scale_model_max_area
        self.s_num_compressed_dim=config.s_num_compressed_dim
        self.cell_size=4
        self.sc = 1.


    def init(self,first_frame,bbox):

        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.target_sz=(w,h)
        self._center = (int(x0 + w / 2),int( y0 + h / 2))
        if w*h>self.translation_model_max_area:
            self.sc=np.sqrt(w*h/self.translation_model_max_area)
        else:
            self.sc=1.
        self.base_target_sz=(w/self.sc,h/self.sc)
        self.win_sz = (int(np.floor(self.base_target_sz[0] * (1 + self.padding))), int(np.floor(self.base_target_sz[1] * (1 + self.padding))))


        output_sigma=np.sqrt(self.base_target_sz[0]*self.base_target_sz[1])*self.output_sigma_factor/self.cell_size
        use_sz=(int(np.floor(self.win_sz[0]/self.cell_size)),int(np.floor(self.win_sz[1]/self.cell_size)))

        self.yf = fft2(0.5*gaussian2d_rolled_labels(use_sz,sigma=output_sigma))
        self.interp_sz=(use_sz[0]*self.cell_size,use_sz[1]*self.cell_size)
        self._window=cos_window(use_sz)

        if self.num_of_scales>0:
            self.scale_sigma = self.num_of_interp_scales * self.scale_sigma_factor
            scale_exp=np.arange(-int(np.floor((self.num_of_scales-1)/2)),
                                int(np.ceil((self.num_of_scales-1)/2))+1)*self.num_of_interp_scales/self.num_of_scales
            scale_exp_shift=np.roll(scale_exp,-int(np.floor((self.num_of_scales-1)/2)))
            interp_scale_exp=np.arange(-int(np.floor((self.num_of_interp_scales-1)/2)),
                                int(np.ceil((self.num_of_interp_scales-1)/2))+1)
            interp_scale_exp_shift=np.roll(interp_scale_exp,-int(np.floor((self.num_of_interp_scales-1)/2)))
            self.scale_size_factors = self.scale_step ** scale_exp
            self.interp_scale_factors=self.scale_step**interp_scale_exp_shift

            ys = np.exp(-0.5 * (scale_exp_shift ** 2) / (self.scale_sigma ** 2))
            self.ysf = np.fft.fft(ys)
            self.scale_window = np.hanning(len(self.ysf)).T

            if self.scale_model_factor**2*(self.base_target_sz[0]*self.base_target_sz[1])>self.scale_model_max_area:
                self.scale_model_factor=np.sqrt(self.scale_model_max_area/(self.base_target_sz[0]*self.base_target_sz[1]))

            self.scale_model_sz = (
            int(np.floor(self.base_target_sz[0] * self.scale_model_factor)), int(np.floor(self.base_target_sz[1] * self.scale_model_factor)))

            self.min_scale_factor = self.scale_step ** (
                int(np.ceil(np.log(max(5 / self.win_sz[0], 5 / self.win_sz[1])) /
                            np.log(self.scale_step))))
            self.max_scale_factor = self.scale_step ** (
                int(np.floor((np.log(min(first_frame.shape[1] / self.base_target_sz[0], first_frame.shape[0] / self.base_target_sz[1])) /
                              np.log(self.scale_step)))))

            self.s_num_compressed_dim=len(self.scale_size_factors)
            xs = self.get_scale_subwindow(first_frame, self._center)

            bigY=xs
            bigY_den=xs
            self.s_num=xs
            scale_basis,_=np.linalg.qr(bigY)
            scale_basis_den,_=np.linalg.qr(bigY_den)
            self.scale_basis=scale_basis.T
            self.scale_basis_den=scale_basis_den.T
            sf_proj=np.fft.fft(self.scale_window*self.scale_basis.dot(self.s_num),axis=1)
            self.sf_num=self.ysf*np.conj(sf_proj)
            xs = self.scale_basis_den.dot(xs)
            xsf=np.fft.fft(xs,axis=1)
            new_sf_den=np.sum(xsf*np.conj(xsf),axis=0)
            self.sf_den=new_sf_den

        self.cn_sigma = self.cn_sigma_color
        self.hog_sigma = self.hog_sigma_color
        self.lr_hog = self.lr_hog_color
        self.lr_cn = self.lr_cn_color
        self.modnum = self.gap
        self.is_gray = False



        patch=cv2.getRectSubPix(first_frame,self.win_sz,self._center).astype(np.uint8)
        self.z_hog,self.z_cn=self.get_features(patch,cell_size=self.cell_size)


        data_matrix_cn=self.z_cn.reshape((-1,self.z_cn.shape[2]))
        pca_basis_cn,_,_=np.linalg.svd(data_matrix_cn.T.dot(data_matrix_cn))
        self.projection_matrix_cn=pca_basis_cn[:,:self.num_compressed_dim_cn]

        data_matrix_hog=self.z_hog.reshape((-1,self.z_hog.shape[2]))
        pca_basis_hog,_,_=np.linalg.svd(data_matrix_hog.T.dot(data_matrix_hog))
        self.projection_matrix_hog=pca_basis_hog[:,:self.num_compressed_dim_hog]

        self.z_cn2,self.z_hog2=self.feature_projection(self.z_cn,self.z_hog,self.projection_matrix_cn,self.projection_matrix_hog,
                                             self._window)
        self.frame_index=1
        self.d=self.train_model()


    def update(self,current_frame,vis=False):
        self.frame_index+=1
        old_pos=(np.inf,np.inf)
        iter=1
        while iter<=self.refinement_iterations and np.any(np.array(old_pos)!=np.array(self._center)):
            patch = cv2.getRectSubPix(current_frame,(int(self.base_target_sz[0]*self.sc*(1+self.padding)),
                                                     int(self.base_target_sz[1]*self.sc*(1+self.padding))), self._center)
            patch=cv2.resize(patch,self.win_sz).astype(np.uint8)
            xo_hog,xo_cn= self.get_features(patch,self.cell_size)
            xo_cn2, xo_hog2 = self.feature_projection(xo_cn, xo_hog, self.projection_matrix_cn, self.projection_matrix_hog,
                                                    self._window)
            detect_k_cn=self.dense_gauss_kernel(self.z_cn2,xo_cn2,self.cn_sigma)
            detect_k_hog=self.dense_gauss_kernel(self.z_hog2,xo_hog2,self.hog_sigma)
            kf=fft2(self.d[0]*detect_k_cn+self.d[1]*detect_k_hog)
            responsef=self.alphaf*np.conj(kf)
            if self.interpolate_response>0:
                if self.interpolate_response==2:
                    self.interp_sz=(int(self.yf.shape[1]*self.cell_size*self.sc),
                               int(self.yf.shape[0]*self.cell_size*self.sc))
                else:
                    responsef=self.resize_dft2(responsef,self.interp_sz)
            response=np.real(ifft2(responsef))
            if vis is True:
                self.score = response
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
                self.win_sz=self.win_sz

            row,col=np.unravel_index(np.argmax(response, axis=None),response.shape)
            disp_row=np.mod(row+np.floor((self.interp_sz[1]-1)/2),self.interp_sz[1])-np.floor((self.interp_sz[1]-1)/2)
            disp_col=np.mod(col+np.floor((self.interp_sz[0]-1)/2),self.interp_sz[0])-np.floor((self.interp_sz[0]-1)/2)
            if self.interpolate_response==0:
                translation_vec=list(np.array([disp_row,disp_col])*self.cell_size*self.sc)
            elif self.interpolate_response==1:
                translation_vec=list(np.array([disp_row,disp_col])*self.sc)
            elif self.interpolate_response==2:
                translation_vec=[disp_row,disp_col]
            trans=np.sqrt(self.win_sz[0]*self.win_sz[1])*self.sc/3
            old_pos=self._center
            self._center=(old_pos[0]+translation_vec[1],old_pos[1]+translation_vec[0])
            iter+=1

        if self.num_of_scales>0:
            xs = self.get_scale_subwindow(current_frame, self._center)
            xs=self.scale_window*self.scale_basis.dot(xs)
            xsf = np.fft.fft(xs, axis=1)
            scale_responsef = np.sum(self.sf_num * xsf, axis=0) / (self.sf_den + self.lambda_)
            interp_scale_response=np.real(np.fft.ifft(self.resize_dft(scale_responsef,self.num_of_interp_scales)))
            recovered_scale = np.argmax(interp_scale_response)
            self.sc = self.sc * self.interp_scale_factors[recovered_scale]
            self.sc = np.clip(self.sc, a_min=self.min_scale_factor,
                              a_max=self.max_scale_factor)

            new_xs = self.get_scale_subwindow(current_frame, self._center)
            self.s_num=(1-self.interp_factor)*self.s_num+self.interp_factor*new_xs
            bigY = self.s_num
            bigY_den = new_xs
            self.scale_basis, _ = np.linalg.qr(bigY)
            self.scale_basis_den, _ = np.linalg.qr(bigY_den)
            self.scale_basis = self.scale_basis.T
            self.scale_basis_den=self.scale_basis_den.T
            sf_proj = np.fft.fft(self.scale_window*self.scale_basis.dot(self.s_num),axis=1)
            self.sf_num = self.ysf * np.conj(sf_proj)
            new_xs =  self.scale_window*self.scale_basis_den.dot(new_xs)
            xsf = np.fft.fft(new_xs,axis=1)
            new_sf_den = np.sum(xsf * np.conj(xsf), axis=0)
            self.sf_den = (1-self.interp_factor)*self.sf_den+self.interp_factor*new_sf_den

        patch = cv2.getRectSubPix(current_frame, (int(self.base_target_sz[0] * self.sc * (1 + self.padding)),
                                                  int(self.base_target_sz[1] * self.sc * (1 + self.padding))),
                                  self._center)
        patch = cv2.resize(patch, self.win_sz).astype(np.uint8)
        xo_hog,xo_cn=self.get_features(patch,self.cell_size)
        self.z_hog=(1-self.lr_hog)*self.z_hog+self.lr_hog*xo_hog
        self.z_cn=(1-self.lr_cn)*self.z_cn+self.lr_cn*xo_cn

        data_matrix_cn = self.z_cn.reshape((-1, self.z_cn.shape[2]))
        pca_basis_cn, _, _ = np.linalg.svd(data_matrix_cn.T.dot(data_matrix_cn))
        self.projection_matrix_cn = pca_basis_cn[:, :self.num_compressed_dim_cn]

        data_matrix_hog = self.z_hog.reshape((-1, self.z_hog.shape[2]))
        pca_basis_hog, _, _ = np.linalg.svd(data_matrix_hog.T.dot(data_matrix_hog))
        self.projection_matrix_hog = pca_basis_hog[:, :self.num_compressed_dim_hog]

        self.z_cn2, self.z_hog2 = self.feature_projection(self.z_cn, self.z_hog, self.projection_matrix_cn, self.projection_matrix_hog,
                                                  self._window)
        if self.frame_index%self.modnum==0:
            self.train_model()
        target_sz=((self.base_target_sz[0]*self.sc),(self.base_target_sz[1]*self.sc))
        return [(self._center[0] - target_sz[0] / 2), (self._center[1] - target_sz[1] / 2), target_sz[0],target_sz[1]]

    def dense_gauss_kernel(self,x1,x2,sigma):
        c=ifft2(np.sum(fft2(x1)*np.conj(fft2(x2)),axis=2))
        d=x1.flatten().conj().T.dot(x1.flatten())+x2.flatten().conj().T.dot(x2.flatten())-2*c
        k=np.exp(-1/sigma**2*d/np.size(d))
        return k

    def train_model(self):
        d=[0.5,0.5]
        dim=self.z_cn2.shape[2]
        kf_cn=fft2(self.dense_gauss_kernel(self.z_cn2,self.z_cn2,self.cn_sigma))
        kf_hog=fft2(self.dense_gauss_kernel(self.z_hog2,self.z_hog2,self.hog_sigma))
        count=0
        stop=False
        lambda1=0.01
        threshold=0.03
        predD=d
        while stop is not True:
            new_num1=self.yf*d[0]*kf_cn
            new_num2=self.yf*d[1]*kf_hog
            new_den1=d[0]*kf_cn*(d[0]*np.conj(kf_cn)+lambda1)
            new_den2=d[1]*kf_hog*(d[1]*np.conj(kf_hog)+lambda1)
            if self.frame_index==1:
                alphaf_num11=new_num1
                alphaf_num22=new_num2
                alphaf_den11=new_den1
                alphaf_den22=new_den2
            else:
                alphaf_num11=(1-self.lr_cn)*self.alphaf_num1+self.lr_cn*new_num1
                alphaf_num22=(1-self.lr_hog)*self.alphaf_num2+self.lr_hog*new_num2
                alphaf_den11=(1-self.lr_cn)*self.alphaf_den1+self.lr_cn*new_den1
                alphaf_den22=(1-self.lr_hog)*self.alphaf_den2+self.lr_hog*new_den2
            self.alphaf_num = alphaf_num11 +alphaf_num22
            self.alphaf_den = alphaf_den11 + alphaf_den22
            self.alphaf=self.alphaf_num/self.alphaf_den
            alpha=ifft2(self.alphaf)
            d=self.trainD(kf_cn,kf_hog,self.alphaf,alpha,lambda1,dim)
            count+=1
            if count>1:
                delta_alpha=np.abs(alpha-prev_alpha)
                deltaD=np.abs(np.array(d)-np.array(predD))
                if(np.sum(delta_alpha)<=threshold*np.sum(np.abs(prev_alpha))) and np.sum(np.array(deltaD))<=threshold*np.sum(np.abs(np.array(predD))):
                    stop=True
            prev_alpha=alpha
            predD=d
            if count>=100:
                d=[0.5,0.5]
                break
        self.alphaf_num1=alphaf_num11
        self.alphaf_num2=alphaf_num22
        self.alphaf_den1=alphaf_den11
        self.alphaf_den2=alphaf_den22
        return d

    def trainD(self,kf_cn,kf_hog,alphaf,alpha,lambda1,dim):
        d=[0,0]
        tmp1=ifft2(np.conj(kf_cn)*alphaf)
        tmp2=ifft2(np.conj(kf_hog)*alphaf)
        y=ifft2(self.yf)
        tmp=2*y-lambda1*alpha
        new_num1=tmp.flatten().conj().T.dot(tmp1.flatten())
        new_num2=tmp.flatten().conj().T.dot(tmp2.flatten())
        new_den1=2*(tmp1.flatten().conj().T.dot(tmp1.flatten()))
        new_den2=2*(tmp2.flatten().conj().T.dot(tmp2.flatten()))
        if self.frame_index==1:
            d_num11=new_num1
            d_num22=new_num2
            d_den11=new_den1
            d_den22=new_den2
        else:
            d_num11=(1-self.lr_cn)*self.d_num1+self.lr_cn*new_num1
            d_num22=(1-self.lr_hog)*self.d_num2+self.lr_hog*new_num2
            d_den11=(1-self.lr_cn)*self.d_den1+self.lr_cn*new_den1
            d_den22=(1-self.lr_hog)*self.d_den2+self.lr_hog*new_den2

        d[0]=np.real(d_num11/d_den11)
        d[1]=np.real(d_num22/d_den22)

        self.d_num1=d_num11
        self.d_num2=d_num22
        self.d_den1=d_den11
        self.d_den2=d_den22
        return d

    def resize_dft2(self, input_dft, desired_sz):
        h,w=input_dft.shape
        if desired_sz[0]!=w or desired_sz[1]!=h:
            minsz=(int(min(w, desired_sz[0])), int(min(h, desired_sz[1])))
            scaling=(desired_sz[0]*desired_sz[1]/(h*w))
            resized_dfft=np.zeros((desired_sz[1],desired_sz[0]),dtype=np.complex64)
            mids=(int(np.ceil(minsz[0]/2)),int(np.ceil(minsz[1]/2)))
            mide=(int(np.floor((minsz[0]-1)/2))-1,int(np.floor((minsz[1]-1)/2))-1)
            resized_dfft[:mids[1],:mids[0]]=scaling*input_dft[:mids[1],:mids[0]]
            resized_dfft[:mids[1],-1-mide[0]:]=scaling*input_dft[:mids[1],-1-mide[0]:]
            resized_dfft[-1-mide[1]:,:mids[0]]=scaling*input_dft[-1-mide[1]:,:mids[0]]
            resized_dfft[-1-mide[1]:,-1-mide[0]:]=scaling*input_dft[-1-mide[1]:,-1-mide[0]:]
            return resized_dfft
        else:
            return input_dft

    def resize_dft(self,input_dft,desired_len):
        length=len(input_dft)
        minsz=min(length,desired_len)
        scaling=desired_len/length
        resized_dft=np.zeros((desired_len,),dtype=np.complex64)
        mids=int(np.ceil(minsz/2))
        mide=int(np.floor((minsz-1)/2))-1
        resized_dft[:mids]=scaling*input_dft[:mids]
        resized_dft[-mide-1:]=scaling*input_dft[-mide-1:]
        return resized_dft


    def feature_projection(self,x_cn,x_hog,projection_matrix_cn, projection_matrix_hog,
                            window):
        h,w=window.shape
        num_pca_out_cn=projection_matrix_cn.shape[1]
        x_proj_cn=np.reshape(x_cn.dot(projection_matrix_cn),(h,w,num_pca_out_cn))
        num_pca_out_hog=projection_matrix_hog.shape[1]
        x_proj_hog=np.reshape(x_hog.dot(projection_matrix_hog),(h,w,num_pca_out_hog))
        x_proj_cn=x_proj_cn*window[:,:,None]
        x_proj_hog=x_proj_hog*window[:,:,None]
        return x_proj_cn,x_proj_hog

    def get_scale_subwindow(self, im, center):
        n_scales=len(self.scale_size_factors)
        out=None
        for s in range(n_scales):
            patch_sz=(int(self.base_target_sz[0] * self.sc * self.scale_size_factors[s]),
                      int(self.base_target_sz[1] * self.sc * self.scale_size_factors[s]))
            im_patch=cv2.getRectSubPix(im,patch_sz,center)
            if self.scale_model_sz[0] > patch_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            im_patch_resized=cv2.resize(im_patch,self.scale_model_sz,interpolation=interpolation).astype(np.uint8)
            tmp=extract_hog_feature(im_patch_resized, cell_size=4)
            if out is None:
                out=tmp.flatten()
            else:
                out=np.c_[out,tmp.flatten()]
        return out

    def get_features(self,patch,cell_size):
        """
        def cell_grayscale(patch,cell_size):
            gray_img=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            i_image=cv2.integral(gray_img.astype(np.float32))
            ys=np.arange(cell_size,gray_img.shape[0]+1,cell_size)
            xs=np.arange(cell_size,gray_img.shape[1]+1,cell_size)
            cell_sum=i_image[ys,:][:,xs]-i_image[ys,:][:,xs-cell_size]-\
                     i_image[ys-cell_size,:][:,xs]+i_image[ys-cell_size,:][:,xs-cell_size]
            cell_gray=cell_sum/(cell_size**2*255)-0.5
            return cell_gray
        """
        hog_feature=extract_hog_feature(patch,cell_size=cell_size)
        cn_feature=extract_cn_feature(patch,cell_size=cell_size)
        return hog_feature,cn_feature





