import numpy as np
import cv2
from .base import BaseCF
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .feature import extract_hog_feature,extract_cn_feature
from .config.asrcfhc_config import ASRCFHCConfig
from .cf_utils import mex_resize,resp_newton,resize_dft2
from .scale_estimator import LPScaleEstimator


class ASRCFHC(BaseCF):
    def __init__(self, config=ASRCFHCConfig()):
        super(ASRCFHC).__init__()
        self.cell_size=config.cell_size
        self.cell_selection_thresh=config.cell_selection_thresh
        self.search_area_shape = config.search_area_shape
        self.search_area_scale=config.search_area_scale
        self.filter_max_area = config.filter_max_area
        self.interp_factor=config.interp_factor
        self.output_sigma_factor = config.output_sigma_factor
        self.interpolate_response =config.interpolate_response
        self.newton_iterations =config.newton_iterations
        self.number_of_scales =config.number_of_scales
        self.scale_step = config.scale_step
        self.admm_iterations = config.admm_iterations
        self.admm_lambda1 = config.admm_lambda1
        self.admm_lambda2=config.admm_lambda2

        self.reg_window_min=config.reg_window_min
        self.reg_window_edge=config.reg_window_edge
        self.reg_window_power=config.reg_window_power
        self.rereg_sparsity_threshold=config.reg_sparsity_threshold
        self.scale_config = config.scale_config


    def init(self,first_frame,bbox):
        if np.all(first_frame[:,:,0]==first_frame[:,:,1]):
            self.is_color=False
        else:
            self.is_color=True
        bbox = np.array(bbox).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.feature_ratio=self.cell_size
        self.search_area=(self.w/self.feature_ratio*self.search_area_scale)*\
                         (self.h/self.feature_ratio*self.search_area_scale)
        if self.search_area<self.cell_selection_thresh*self.filter_max_area:
            self.cell_size=int(min(self.feature_ratio,max(1,int(np.ceil(np.sqrt(
                self.w*self.search_area_scale/(self.cell_selection_thresh*self.filter_max_area)*\
                self.h*self.search_area_scale/(self.cell_selection_thresh*self.filter_max_area)
            ))))))
            self.feature_ratio=self.cell_size
            self.search_area = (self.w / self.feature_ratio * self.search_area_scale) * \
                               (self.h / self.feature_ratio * self.search_area_scale)

        if self.search_area>self.filter_max_area:
            self.current_scale_factor=np.sqrt(self.search_area/self.filter_max_area)
        else:
            self.current_scale_factor=1.

        self.base_target_sz=(self.w/self.current_scale_factor,self.h/self.current_scale_factor)
        self.target_sz=self.base_target_sz
        if self.search_area_shape=='proportional':
            self.crop_size=(int(self.base_target_sz[0]*self.search_area_scale),int(self.base_target_sz[1]*self.search_area_scale))
        elif self.search_area_shape=='square':
            w=int(np.sqrt(self.base_target_sz[0]*self.base_target_sz[1])*self.search_area_scale)
            self.crop_size=(w,w)
        elif self.search_area_shape=='fix_padding':
            tmp=int(np.sqrt(self.base_target_sz[0]*self.search_area_scale+(self.base_target_sz[1]-self.base_target_sz[0])/4))+\
                (self.base_target_sz[0]+self.base_target_sz[1])/2
            self.crop_size=(self.base_target_sz[0]+tmp,self.base_target_sz[1]+tmp)
        else:
            raise ValueError
        self.crop_size=(int(round(self.crop_size[0]/self.feature_ratio)*self.feature_ratio),int(round(self.crop_size[1]/self.feature_ratio)*self.feature_ratio))
        self.feature_map_sz=(self.crop_size[0]//self.feature_ratio,self.crop_size[1]//self.feature_ratio)
        output_sigma=np.sqrt(np.floor(self.base_target_sz[0]/self.feature_ratio)*np.floor(self.base_target_sz[1]/self.feature_ratio))*self.output_sigma_factor
        y=gaussian2d_rolled_labels(self.feature_map_sz, output_sigma)
        self.yf=fft2(y)

        use_sz = self.feature_map_sz

        self.reg_window=self.reg_filter(use_sz,(self.base_target_sz[0]//self.cell_size,self.base_target_sz[1]//self.cell_size),
                                        self.reg_window_power,self.reg_window_min,
                                        self.reg_window_edge,self.rereg_sparsity_threshold)


        if self.interpolate_response==1:
            self.interp_sz=(self.feature_map_sz[0]*self.feature_ratio,self.feature_map_sz[1]*self.feature_ratio)
        else:
            self.interp_sz=(self.feature_map_sz[0],self.feature_map_sz[1])
        self._window=cos_window(self.feature_map_sz)
        if self.number_of_scales>0:
            scale_exp=np.arange(-int(np.floor((self.number_of_scales-1)/2)),int(np.ceil((self.number_of_scales-1)/2))+1)
            self.scale_factors=self.scale_step**scale_exp
            self.min_scale_factor=self.scale_step**(np.ceil(np.log(max(5/self.crop_size[0],5/self.crop_size[1]))/np.log(self.scale_step)))
            self.max_scale_factor=self.scale_step**(np.floor(np.log(min(first_frame.shape[0]/self.base_target_sz[1],
                                                                        first_frame.shape[1]/self.base_target_sz[0]))/np.log(self.scale_step)))
        if self.interpolate_response>=3:
            self.ky=np.roll(np.arange(-int(np.floor((self.feature_map_sz[1]-1)/2)),int(np.ceil((self.feature_map_sz[1]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[1]-1)/2)))
            self.kx=np.roll(np.arange(-int(np.floor((self.feature_map_sz[0]-1)/2)),int(np.ceil((self.feature_map_sz[0]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[0]-1)/2))).T

        self.small_filter_sz=(int(np.floor(self.base_target_sz[0]/self.feature_ratio)),int(np.floor(self.base_target_sz[1]/self.feature_ratio)))

        self.scale_estimator = LPScaleEstimator(self.target_sz, config=self.scale_config)

        self.scale_estimator.init(first_frame, self._center, self.base_target_sz, self.current_scale_factor)

        pixels=self.get_sub_window(first_frame,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(np.round(self.crop_size[0]*self.current_scale_factor)),
                                              int(np.round(self.crop_size[1]*self.current_scale_factor))))
        feature=self.extract_hc_feture(pixels, cell_size=self.feature_ratio)
        self.model_xf=fft2(self._window[:,:,None]*feature)

        self.h_f, self.reg_window=self.ADMM(self.model_xf, self.reg_window, self.admm_lambda1, self.admm_lambda2)


    def update(self,current_frame,vis=False):
        x=None
        for scale_ind in range(self.number_of_scales):
            current_scale=self.current_scale_factor*self.scale_factors[scale_ind]
            sub_window=self.get_sub_window(current_frame,self._center,model_sz=self.crop_size,
                                        scaled_sz=(int(round(self.crop_size[0]*current_scale)),
                                    int(round(self.crop_size[1]*current_scale))))
            feature= self.extract_hc_feture(sub_window, self.cell_size)[:, :, :, np.newaxis]
            if x is None:
                x=feature
            else:
                x=np.concatenate((x,feature),axis=3)
        xtf=fft2(x*self._window[:,:,None,None])
        responsef=np.sum(np.conj(self.h_f)[:, :, :, None] * xtf, axis=2)

        if self.interpolate_response==2:
            self.interp_sz=(int(self.yf.shape[1]*self.feature_ratio*self.current_scale_factor),
                            int(self.yf.shape[0]*self.feature_ratio*self.current_scale_factor))
        responsef_padded=resize_dft2(responsef,self.interp_sz)
        response=np.real(ifft2(responsef_padded))
        if self.interpolate_response==3:
            raise ValueError
        elif self.interpolate_response==4:
            disp_row,disp_col,sind=resp_newton(response,responsef_padded,self.newton_iterations,
                                                    self.ky,self.kx,self.feature_map_sz)
            if vis is True:
                self.score=response[:,:,sind]
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
        else:
            row,col,sind=np.unravel_index(np.argmax(response,axis=None),response.shape)

            if vis is True:
                self.score=response[:,:,sind]
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
            disp_row=(row+int(np.floor(self.interp_sz[1]-1)/2))%self.interp_sz[1]-int(np.floor((self.interp_sz[1]-1)/2))
            disp_col = (col + int(np.floor(self.interp_sz[0] - 1) / 2)) % self.interp_sz[0] - int(
                np.floor((self.interp_sz[0] - 1) / 2))

        if self.interpolate_response==0  or self.interpolate_response==3 or self.interpolate_response==4:
            factor=self.feature_ratio*self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==1:
            factor=self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==2:
            factor=self.scale_factors[sind]
        else:
            raise ValueError
        dx,dy=int(np.round(disp_col*factor)),int(np.round(disp_row*factor))
        self.current_scale_factor=self.current_scale_factor*self.scale_factors[sind]
        self.current_scale_factor=max(self.current_scale_factor,self.min_scale_factor)
        self.current_scale_factor=min(self.current_scale_factor,self.max_scale_factor)

        self.current_scale_factor = self.scale_estimator.update(current_frame, self._center, self.base_target_sz,
                                              self.current_scale_factor)

        self._center=(self._center[0]+dx,self._center[1]+dy)

        pixels=self.get_sub_window(current_frame,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(round(self.crop_size[0]*self.current_scale_factor)),
                                              int(round(self.crop_size[1]*self.current_scale_factor))))
        feature=self.extract_hc_feture(pixels, cell_size=self.cell_size)
        #feature=cv2.resize(pixels,self.feature_map_sz)/255-0.5
        xf=fft2(feature*self._window[:,:,None])
        self.model_xf=(1-self.interp_factor)*self.model_xf+self.interp_factor*xf
        self.h_f, self.reg_window = self.ADMM(self.model_xf, self.reg_window, self.admm_lambda1, self.admm_lambda2)

        target_sz=(self.target_sz[0]*self.current_scale_factor,self.target_sz[1]*self.current_scale_factor)
        return [self._center[0]-target_sz[0]/2,self._center[1]-target_sz[1]/2,target_sz[0],target_sz[1]]

    def get_subwindow_no_window(self,img,pos,sz):
        h,w=sz[1],sz[0]
        xs = (np.floor(pos[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(pos[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        out = img[ys, :][:, xs]
        xs,ys=np.meshgrid(xs,ys)
        return xs,ys,out

    def vis_regwin(self,reg_win):
        h,w=reg_win.shape[:2]
        ys,xs=np.meshgrid(np.arange(h),np.arange(w))
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.contourf(xs,ys,reg_win[ys,xs],50,cmap='binary')
        plt.show()


    def ADMM(self,xf,reg_window,lambda1,lambda2):
        g_f = np.zeros_like(xf)
        h_f=np.zeros_like(xf)
        l_f = np.zeros_like(g_f)
        mu = 1
        beta = 10
        mumax = 10000
        i = 1
        T = self.feature_map_sz[0] * self.feature_map_sz[1]
        #S_xx = np.sum(np.conj(xf) * xf,axis=2)
        xs, ys, _ = self.get_subwindow_no_window(h_f,(int(self.feature_map_sz[0] / 2), int(self.feature_map_sz[1] / 2)),
                                                 self.small_filter_sz)
        p = np.zeros((self.feature_map_sz[1], self.feature_map_sz[0], h_f.shape[2]))
        p[ys, xs, :] = 1.

        while i <= self.admm_iterations:
            # solve g
            S_lx = np.sum(np.conj(xf) * l_f, axis=2)
            S_hx = np.sum(np.conj(xf) * h_f, axis=2)
            # S_lx=np.conj(xf)*l_f
            # S_hx=np.conj(xf)*h*p
            coef = 1 / (T * mu)
            temp0 = 1 - (np.conj(xf) * xf / (mu * T + xf * np.conj(xf)))
            temp1 = self.yf[:, :, None] * xf + mu * S_hx[:, :, None] - mu * S_lx[:, :, None]
            g_f = coef * temp0 * temp1

            # solve h
            h = (mu * T * p * (np.real(ifft2(l_f + g_f)) + 1e-10)) / (
                        lambda1 * (reg_window ** 2)[:, :, None] + mu * T * p)
            h_f = fft2(h)

            # solve w
            reg_window=lambda2*reg_window/(lambda1*np.sum(h*h,axis=2)+lambda2)

            #self.vis_regwin(reg_window)
            #cv2.waitKey(0)
            # update l_f
            l_f = l_f + (g_f-h_f)
            mu = min(beta * mu, mumax)
            i += 1

        return h_f,reg_window


    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = mex_resize(im_patch, model_sz)
        return im_patch.astype(np.uint8)

    def extract_hc_feture(self,patch,cell_size):
        hog_feature=extract_hog_feature(patch,cell_size)
        cn_feature=extract_cn_feature(patch,cell_size)
        hc_feature=np.concatenate((hog_feature,cn_feature),axis=2)
        return hc_feature


    def reg_filter(self,sz,target_sz,reg_window_power,reg_window_min,reg_window_edge,reg_sparsity_threshold):
        # normalization factor
        reg_scale = (0.5*target_sz[0],0.5*target_sz[1])

        # construct grid
        wrg = np.arange(-(sz[0] - 1) / 2, (sz[1] - 1) / 2 + 1, dtype=np.float32)
        wcg = np.arange(-(sz[0] - 1) / 2, (sz[1] - 1) / 2 + 1, dtype=np.float32)
        wrs, wcs = np.meshgrid(wrg, wcg)

        # construct the regularization window
        reg_window = (reg_window_edge - reg_window_min) * (
                np.abs(wrs / reg_scale[0]) ** reg_window_power + \
                np.abs(wcs / reg_scale[1]) ** reg_window_power) + reg_window_min

        return reg_window









