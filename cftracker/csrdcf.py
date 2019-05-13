"""
Python re-implementation of "Discriminative Correlation Filter with Channel and Spatial Reliability"
@inproceedings{Lukezic2017Discriminative,
  title={Discriminative Correlation Filter with Channel and Spatial Reliability},
  author={Lukezic, Alan and Vojir, Tomas and Zajc, Luka Cehovin and Matas, Jiri and Kristan, Matej},
  booktitle={IEEE Conference on Computer Vision & Pattern Recognition},
  year={2017},
}
"""
import numpy as np
import cv2
from .base import BaseCF
from lib.utils import cos_window
from lib.fft_tools import fft2,ifft2
from lib.utils import gaussian2d_rolled_labels
from .feature import extract_hog_feature,extract_cn_feature
from .config import csrdcf_config
from cftracker.scale_estimator import LPScaleEstimator,DSSTScaleEstimator


def kernel_profile_epanechnikov(x):
    res = np.zeros_like(x)
    res[np.where(x <= 1)] = 2 / 3.14 * (1 - x[x <= 1])
    return res

class CSRDCF(BaseCF):
    def __init__(self,config):
        super(CSRDCF).__init__()

        self.padding=config.padding
        self.interp_factor = config.interp_factor
        self.y_sigma =config.y_sigma
        self.channels_weight_lr=self.interp_factor
        self.use_channel_weights = config.use_channel_weights

        self.hist_lr=config.hist_lr
        self.nbins=config.nbins
        self.segcolor_space=config.seg_colorspace
        self.use_segmentation=config.use_segmentation

        self.scale_type = config.scale_type
        self.scale_config = config.scale_config


        self.hist_fg_p_bins=None
        self.hist_bg_p_bins=None
        self.p_b=0


    def init(self,first_frame,bbox):

        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self.init_mask=np.ones((h,w),dtype=np.uint8)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        if np.all(first_frame[:,:,0]==first_frame[:,:,1]):
            self.use_segmentation=False
        # change 400 to 300
        # for larger cell_size
        self.cell_size=int(min(4,max(1,w*h/300)))
        self.base_target_sz=(w,h)
        self.target_sz=self.base_target_sz

        template_size=(int(w+self.padding*np.sqrt(w*h)),int(h+self.padding*np.sqrt(w*h)))
        template_size=(template_size[0]+template_size[1])//2
        self.template_size=(template_size,template_size)

        self.rescale_ratio=np.sqrt((200**2)/(self.template_size[0]*self.template_size[1]))
        self.rescale_ratio=np.clip(self.rescale_ratio,a_min=None,a_max=1)

        self.rescale_template_size=(int(self.rescale_ratio*self.template_size[0]),
                                    int(self.rescale_ratio*self.template_size[1]))
        self.yf=fft2(gaussian2d_rolled_labels((int(self.rescale_template_size[0]/self.cell_size),
                                               int(self.rescale_template_size[1]/self.cell_size)),
                                              self.y_sigma))

        self._window=cos_window((self.yf.shape[1],self.yf.shape[0]))
        self.crop_size=self.rescale_template_size

        self.current_scale_factor = 1.

        if self.scale_type=='normal':
            self.scale_estimator = DSSTScaleEstimator(self.target_sz, config=self.scale_config)
            self.scale_estimator.init(first_frame, self._center, self.base_target_sz, self.current_scale_factor)
            self._num_scales = self.scale_estimator.num_scales
            self._scale_step = self.scale_estimator.scale_step

            self._min_scale_factor = self._scale_step ** np.ceil(
                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(
                first_frame.shape[:2] / np.array([self.base_target_sz[1], self.base_target_sz[0]]))) / np.log(
                self._scale_step))
        elif self.scale_type=='LP':
            self.scale_estimator=LPScaleEstimator(self.target_sz,config=self.scale_config)
            self.scale_estimator.init(first_frame,self._center,self.base_target_sz,self.current_scale_factor)



        # create dummy  mask (approximation for segmentation)
        # size of the object in feature space
        obj_sz=(int(self.rescale_ratio*(self.base_target_sz[0]/self.cell_size)),
                int(self.rescale_ratio*(self.base_target_sz[1]/self.cell_size)))
        x0=int((self.yf.shape[1]-obj_sz[0])/2)
        y0=int((self.yf.shape[0]-obj_sz[1])/2)
        x1=x0+obj_sz[0]
        y1=y0+obj_sz[1]
        self.target_dummy_mask=np.zeros_like(self.yf,dtype=np.uint8)
        self.target_dummy_mask[y0:y1,x0:x1]=1
        self.target_dummy_area=np.sum(self.target_dummy_mask)
        if self.use_segmentation:
            if self.segcolor_space=='bgr':
                seg_img=first_frame
            elif self.segcolor_space=='hsv':
                seg_img=cv2.cvtColor(first_frame,cv2.COLOR_BGR2HSV)
                seg_img[:, :, 0] = (seg_img[:, :, 0].astype(np.float32)/180*255)
                seg_img = seg_img.astype(np.uint8)
            else:
                raise ValueError
            hist_fg=Histogram(3,self.nbins)
            hist_bg=Histogram(3,self.nbins)
            self.extract_histograms(seg_img,bbox,hist_fg,hist_bg)

            mask=self.segment_region(seg_img,self._center,self.template_size,self.base_target_sz,self.current_scale_factor,
                                     hist_fg,hist_bg)
            self.hist_bg_p_bins=hist_bg.p_bins
            self.hist_fg_p_bins=hist_fg.p_bins

            init_mask_padded=np.zeros_like(mask)
            pm_x0=int(np.floor(mask.shape[1]/2-bbox[2]/2))
            pm_y0=int(np.floor(mask.shape[0]/2-bbox[3]/2))
            init_mask_padded[pm_y0:pm_y0+bbox[3],pm_x0:pm_x0+bbox[2]]=1
            mask=mask*init_mask_padded
            mask=cv2.resize(mask,(self.yf.shape[1],self.yf.shape[0]))
            if self.mask_normal(mask,self.target_dummy_area) is True:
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3),anchor=(1,1))
                mask=cv2.dilate(mask,kernel)
            else:
                mask=self.target_dummy_mask
        else:
            mask=self.target_dummy_mask
        # extract features
        f=self.get_csr_features(first_frame,self._center,self.current_scale_factor,
                                self.template_size,self.rescale_template_size,self.cell_size)
        f=f*self._window[:,:,None]
        # create filters using segmentation mask
        self.H=self.create_csr_filter(f,self.yf,mask)
        response=np.real(ifft2(fft2(f)*np.conj(self.H)))
        chann_w=np.max(response.reshape(response.shape[0]*response.shape[1],-1),axis=0)
        self.chann_w=chann_w/np.sum(chann_w)


    def update(self,current_frame,vis=False):
        f=self.get_csr_features(current_frame,self._center,self.current_scale_factor,
                                self.template_size,self.rescale_template_size,self.cell_size)
        f=f*self._window[:,:,None]
        if self.use_channel_weights is True:
            response_chann=np.real(ifft2(fft2(f)*np.conj(self.H)))
            response=np.sum(response_chann*self.chann_w[None,None,:],axis=2)
        else:
            response=np.real(ifft2(np.sum(fft2(f)*np.conj(self.H),axis=2)))
        if vis is True:
            self.score=response
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        curr=np.unravel_index(np.argmax(response,axis=None),response.shape)
        if self.use_channel_weights is True:
            channel_discr=np.ones((response_chann.shape[2]))
            for i in range(response_chann.shape[2]):
                norm_response=self.normalize_img(response_chann[:,:,i])

                from skimage.feature.peak import peak_local_max
                peak_locs=peak_local_max(norm_response,min_distance=5)
                if len(peak_locs)<2:
                    continue
                vals=reversed(sorted(norm_response[peak_locs[:,0],peak_locs[:,1]]))
                second_max_val=None
                max_val=None
                for index,val in enumerate(vals):
                    if index==0:
                        max_val=val
                    elif index==1:
                        second_max_val=val
                    else:
                        break
                channel_discr[i]=max(0.5,1-(second_max_val/(max_val+1e-10)))

        v_neighbors=response[[(curr[0]-1)%response.shape[0],(curr[0])%response.shape[0],
                              (curr[0]+1)%response.shape[0]],curr[1]]
        h_neighbors=response[curr[0],
                             [(curr[1]-1) % response.shape[1], (curr[1]) % response.shape[1],
                             (curr[1]+1) % response.shape[1]]
                            ]
        row=curr[0]+self.subpixel_peak(v_neighbors)
        col=curr[1]+self.subpixel_peak(h_neighbors)
        if row+1>response.shape[0]/2:
            row=row-response.shape[0]
        if col+1>response.shape[1]/2:
            col=col-response.shape[1]
        # displacement
        dx=self.current_scale_factor*self.cell_size*(1/self.rescale_ratio)*col
        dy=self.current_scale_factor*self.cell_size*(1/self.rescale_ratio)*row
        self._center=(self._center[0]+dx,self._center[1]+dy)

        self.current_scale_factor = self.scale_estimator.update(current_frame, self._center, self.base_target_sz,
                                                                self.current_scale_factor)
        if self.scale_type == 'normal':
            self.current_scale_factor = np.clip(self.current_scale_factor, a_min=self._min_scale_factor,
                                                a_max=self._max_scale_factor)

        self.target_sz = (self.current_scale_factor * self.base_target_sz[0],
                          self.current_scale_factor * self.base_target_sz[1])
        region=[np.round(self._center[0] - self.target_sz[0] / 2),np.round( self._center[1] - self.target_sz[1] / 2),
                        self.target_sz[0], self.target_sz[1]]
        if self.use_segmentation:
            if self.segcolor_space=='bgr':
                seg_img=current_frame
            elif self.segcolor_space=='hsv':
                seg_img=cv2.cvtColor(current_frame,cv2.COLOR_BGR2HSV)
                seg_img[:, :, 0] = (seg_img[:, :, 0].astype(np.float32)/180*255)
                seg_img = seg_img.astype(np.uint8)
            else:
                raise ValueError

            hist_fg=Histogram(3,self.nbins)
            hist_bg=Histogram(3,self.nbins)
            self.extract_histograms(seg_img,region,hist_fg,hist_bg)
            self.hist_fg_p_bins=(1-self.hist_lr)*self.hist_fg_p_bins+self.hist_lr*hist_fg.p_bins
            self.hist_bg_p_bins=(1-self.hist_lr)*self.hist_bg_p_bins+self.hist_lr*hist_bg.p_bins

            hist_fg.p_bins=self.hist_fg_p_bins
            hist_bg.p_bins=self.hist_bg_p_bins
            mask=self.segment_region(seg_img,self._center,self.template_size,self.base_target_sz,self.current_scale_factor,
                                     hist_fg,hist_bg)
            init_mask_padded=np.zeros_like(mask)
            pm_x0=int(np.floor(mask.shape[1]/2-region[2]/2))
            pm_y0=int(np.floor(mask.shape[0]/2-region[3]/2))
            init_mask_padded[pm_y0:pm_y0+int(np.round(region[3])),pm_x0:pm_x0+int(np.round(region[2]))]=1
            mask=mask*init_mask_padded
            mask=cv2.resize(mask,(self.yf.shape[1],self.yf.shape[0]))
            if self.mask_normal(mask,self.target_dummy_area) is True:
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3),anchor=(1,1))
                mask=cv2.dilate(mask,kernel)
            else:
                mask=self.target_dummy_mask
            pass
        else:
            mask=self.target_dummy_mask

        #cv2.imshow('Mask', (mask * 255).astype(np.uint8))
        #cv2.waitKey(1)

        f = self.get_csr_features(current_frame, self._center, self.current_scale_factor,
                                  self.template_size, self.rescale_template_size, self.cell_size)
        f = f * self._window[:, :, None]
        H_new=self.create_csr_filter(f,self.yf,mask)
        if self.use_channel_weights:
            response=np.real(ifft2(fft2(f)*np.conj(H_new)))
            chann_w = np.max(response.reshape(response.shape[0] * response.shape[1], -1), axis=0)*channel_discr
            chann_w=chann_w/np.sum(chann_w)
            self.chann_w=(1-self.channels_weight_lr)*self.chann_w+self.channels_weight_lr*chann_w
            self.chann_w=self.chann_w/np.sum(self.chann_w)
        self.H=(1-self.interp_factor)*self.H+self.interp_factor*H_new

        return region


    def get_csr_features(self,img,center,scale,template_sz,resize_sz,cell_size):
        center=(int(center[0]),int(center[1]))
        patch=cv2.getRectSubPix(img,patchSize=(int(scale*template_sz[0]),int(scale*template_sz[1])),
                                center=center)
        patch=cv2.resize(patch,resize_sz).astype(np.uint8)
        hog_feature=extract_hog_feature(patch,cell_size)[:,:,:18]
        # gray feature is included in the cn features
        cn_feature=extract_cn_feature(patch,cell_size)
        features=np.concatenate((hog_feature,cn_feature),axis=2)
        return features

    def get_patch(self,img,center,scale,template_size):
        w = int(np.floor(scale*template_size[0]))
        h = int(np.floor(scale*template_size[1]))
        xs = (np.floor(center[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(center[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        valid_pixels_mask=np.ones((h,w),dtype=np.uint8)
        valid_pixels_mask[:,xs<0]=0
        valid_pixels_mask[:,xs>=img.shape[1]]=0
        valid_pixels_mask[ys<0,:]=0
        valid_pixels_mask[ys>=img.shape[0],:]=0
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        cropped = img[ys, :][:, xs]
        return valid_pixels_mask,cropped

    def create_csr_filter(self,img,Y,P):
        """
        create csr filter
        create filter with Augmented Lagrangian iterative optimization method
        :param img: image patch (already normalized)
        :param Y: gaussian shaped labels (note that the peak must be at the top-left corner)
        :param P: padding mask
        :return: filter
        """
        mu=5
        beta=3
        mu_max=20
        max_iter=4
        lambda_=mu/100
        F=fft2(img)
        Sxy=F*np.conj(Y)[:,:,None]
        Sxx=F*np.conj(F)
        # mask filter
        H=fft2(ifft2(Sxy/(Sxx+lambda_))*P[:,:,None])
        # initialize lagrangian multiplier
        L=np.zeros_like(H)
        iter=1
        while True:
            G=(Sxy+mu*H-L)/(Sxx+mu)
            H=fft2(np.real(P[:,:,None]*ifft2(mu*G+L)/(mu+lambda_)))
            # stop optimization after fixed number of steps
            if iter>=max_iter:
                break
            L+=mu*(G-H)
            mu=min(mu_max,beta*mu)
            iter+=1
        return H

    def binarize_softmask(self,M,binary_threshold=0.5):
        """
        binarize softmask
        binarize mask so that mask is first put on the [0,1] interval
        :param M: input mask
        :param binary_threshold:
        :return: interval mask
        """
        max_val=np.max(M)
        if max_val<=0:
            max_val=1
        M=M/max_val
        M[M>binary_threshold]=1
        M[M<=binary_threshold]=0
        return M

    def get_location_prior(self,roi,target_sz,img_sz):
        """
        :param roi: top_left and bottom_right point
        :param target_sz:
        :param img_sz:
        :return:
        """
        w,h=img_sz
        x1=int(round(max(min(roi[0]-1,w-1),0)))
        y1=int(round(max(min(roi[1]-1,h-1),0)))
        x2=int(round(min(max(roi[2]-1,0),w-1)))
        y2=int(round(min(max(roi[3]-1,0),h-1)))
        # make it rotationaly invariant
        target_size=min(target_sz[0],target_sz[1])
        target_sz=(target_size,target_size)

        kernel_size_width=1/(0.5*target_sz[0]*1.4142+1)
        kernel_size_height=1/(0.5*target_sz[1]*1.4142+1)
        cx=x1+0.5*(x2-x1)
        cy=y1+0.5*(y2-y1)

        kernel_weight=np.zeros((1+int(np.floor(y2-y1)),1+int(np.floor(-(x1-cx)+x2-cx))))
        ys=np.arange(y1,y2+1)
        xs=np.arange(x1,x2+1)
        xs,ys=np.meshgrid(xs,ys)
        kernel_weight[ys,xs]=kernel_profile_epanechnikov(((cx-xs)*kernel_size_width)**2+((cy-ys)*kernel_size_height)**2)
        max_val=np.max(kernel_weight)
        fg_prior=kernel_weight/max_val
        fg_prior=np.clip(fg_prior,a_min=0.5,a_max=0.9)
        return fg_prior


    def mask_normal(self,mask_bin,obj_area,lower_thresh=0.05):
        area_m=np.sum(mask_bin>0)
        if np.isnan(area_m) or area_m<obj_area*lower_thresh:
            return False
        return True

    def subpixel_peak(self,p):
        delta=0.5*(p[2]-p[0])/(2*p[1]-p[2]-p[0])
        if not np.isfinite(delta):
            delta=0
        return delta

    def normalize_img(self,img):
        min_val,max_val=np.min(img),np.max(img)
        if max_val>min_val:
            out=(img-min_val)/(max_val-min_val)
        else:
            out=np.zeros_like(img)
        return out

    def extract_histograms(self,img,roi,hf,hb):
        x,y,w,h=roi
        x1=int(min(max(0,x),img.shape[1]-1))
        y1=int(min(max(0,y),img.shape[0]-1))
        x2=int(min(max(0,x+w),img.shape[1]-1))
        y2=int(min(max(0,y+h),img.shape[0]-1))
        # calculate coordinates of the background region

        offset_x=(x2-x1+1)//3
        offset_y=(y2-y1+1)//3
        outer_y1=int(max(0,y1-offset_y))
        outer_y2=int(min(img.shape[0],y2+offset_y+1))
        outer_x1=int(max(0,x1-offset_x))
        outer_x2=int(min(img.shape[1],x2+offset_x+1))

        self.p_b=1-((x2-x1+1)*(y2-y1+1))/((outer_x2-outer_x1+1)*(outer_y2-outer_y1+1))
        hf.extract_foreground_histogram(img,None,False,(x1,y1),(x2,y2))
        hb.extract_background_histogram(img,(x1,y1),(x2,y2),(outer_x1,outer_y1),(outer_x2,outer_y2))


    def segment_region(self,img,center,template_size,target_size,scale_factor,hist_fg,hist_bg):
        valid_pixels_mask,patch=self.get_patch(img,center,scale_factor,template_size)
        scaled_target_sz=(target_size[0]*scale_factor,target_size[1]*scale_factor)
        fg_prior=self.get_location_prior((0,0,patch.shape[1],patch.shape[0]),scaled_target_sz,(patch.shape[1],patch.shape[0]))

        probs=Segment.compute_posteriors(patch, fg_prior, 1 - fg_prior,hist_fg,hist_bg, tl=(0, 0),
                                         br=(patch.shape[1],patch.shape[0]), p_b=self.p_b)

        mask=valid_pixels_mask*probs[0]
        mask=self.binarize_softmask(mask)
        return mask


class Histogram:
    def __init__(self,num_dimensions,num_bins_perdimension=8):
        self.num_dimensions=num_dimensions
        self.num_bins_perdimension=num_bins_perdimension
        self.p_size=int(np.floor(num_bins_perdimension**num_dimensions))
        self.p_bins=np.zeros((self.p_size,))
        self.p_dim_id_coef=np.power(num_bins_perdimension,(num_dimensions-1-np.arange(num_dimensions))).astype(np.int64)


    def extract_foreground_histogram(self,img_channels,weights,use_mat_weights,tl,br):
        img=img_channels[:,:,0]
        x1,y1=tl
        x2,y2=br
        if use_mat_weights is not True:
            cx=x1+(x2-x1)/2
            cy=y1+(y2-y1)/2
            kernel_size_width=1/(0.5*(x2-x1)*1.4142+1)
            kernel_size_height=1/(0.5*(y2-y1)*1.4142+1)
            kernel_weight=np.zeros_like(img,dtype=np.float32)
            ys=np.arange(y1,y2+1)
            xs=np.arange(x1,x2+1)
            xs,ys=np.meshgrid(xs,ys)
            kernel_weight[ys,xs]=kernel_profile_epanechnikov(((cx-xs)*kernel_size_width)**2+((cy-ys)*kernel_size_height)**2)
            weights=kernel_weight
        range_perbin_inverse=self.num_bins_perdimension/256
        """
        sum=0
        for y in range(y1,y2+1):
            for x in range(x1,x2+1):
                id=0
                for dim in range(self.num_dimensions):
                    id+=self.p_dim_id_coef[dim]*int(np.floor(range_perbin_inverse*img_channels[y,x,dim]))
                self.p_bins[id]+=weights[y,x]
                sum+=weights[y,x]
        """
        ids=np.sum(self.p_dim_id_coef[None,None,:]*(np.floor(
            range_perbin_inverse*img_channels[y1:y2+1,x1:x2+1]).astype(np.int64)),axis=2)
        self.p_bins[ids]+=weights[y1:y2+1,x1:x2+1]
        self.p_bins=self.p_bins/np.sum(self.p_bins)



    def extract_background_histogram(self,img_channels,tl,br,outer_tl,outer_br):
        range_per_bin_inverse=self.num_bins_perdimension/256
        #sum=0
        x1,y1=tl
        x2,y2=br
        outer_x1,outer_y1=outer_tl
        outer_x2,outer_y2=outer_br
        """
        for y in range(outer_y1,outer_y2):
            for x in range(outer_x1,outer_x2):
                if x>=x1 and x<=x2 and y>=y1 and y<=y2:
                    continue
                id=0
                for dim in range(self.num_dimensions):
                    id+=self.p_dim_id_coef[dim]*int(np.floor(range_per_bin_inverse*img_channels[y,x,dim]))
                self.p_bins[id]+=1
                sum+=1
        sum=1./sum
        """
        mask=np.ones((outer_y2-outer_y1,outer_x2-outer_x1))
        mask[y1-outer_y1:y2+1-outer_y1,x1-outer_x1:x2+1-outer_x1]=-1
        ids = np.sum(self.p_dim_id_coef[None, None, :] * (np.floor(
            range_per_bin_inverse *mask[:,:,None]*img_channels[outer_y1:outer_y2, outer_x1:outer_x2]).astype(np.int64)), axis=2)
        # only statistic valid val
        self.p_bins[ids[ids>=0]]+=1
        self.p_bins=self.p_bins/np.sum(self.p_bins)


    def back_project(self,img_channels):
        range_per_bin_inverse = self.num_bins_perdimension / 256
        """
        img=img_channels[:,:,0]
        back_project=np.zeros_like(img,dtype=np.float32)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                id=0
                for dim in range(self.num_dimensions):
                    id+=self.p_dim_id_coef[dim]*int(np.floor(range_per_bin_inverse*img_channels[y,x,dim]))
                back_project[y,x]=self.p_bins[int(id)]
        """
        ids = np.sum(self.p_dim_id_coef[None, None, :] * (np.floor(
            range_per_bin_inverse * img_channels).astype(np.int64)), axis=2)
        back_project=self.p_bins[ids]
        return back_project



class Segment:

    @staticmethod
    def compute_posteriors(img_channels, fg_prior, bg_prior, hist_target, hist_backgroud, tl, br,p_b):

        x1, y1 = tl
        x2, y2 = br
        x1 = int(min(max(x1, 0), img_channels.shape[1] - 1))
        y1 = int(min(max(y1, 0), img_channels.shape[0] - 1))
        x2 = int(max(min(x2, img_channels.shape[1] - 1), 0))
        y2 = int(max(min(y2, img_channels.shape[0] - 1), 0))
        #p_b = 5. / 3.
        #p_o = 1. / (p_b + 1)
        p_o=1-p_b
        factor = min(1, np.sqrt(1000 / ((x2 - x1) * (y2 - y1))))
        new_size = (int((x2 - x1) * factor), int((y2 - y1) * factor))
        img_channels_roi_inner = np.zeros((new_size[1], new_size[0], img_channels.shape[2]), dtype=np.float32)
        for i in range(img_channels.shape[2]):
            img_channels_roi_inner[:, :, i] = cv2.resize(img_channels[y1:y2 + 1, x1:x2 + 1, i], new_size)
        if len(fg_prior.shape) < 2:
            fg_prior_scaled = 0.5 * np.ones((new_size[1], new_size[0]))
        else:
            fg_prior_scaled = cv2.resize(fg_prior[y1:y2 + 1, x1:x2 + 1], new_size)

        if len(bg_prior.shape) < 2:
            bg_prior_scaled = 0.5 * np.ones((new_size[1], new_size[0]))
        else:
            bg_prior_scaled = cv2.resize(bg_prior[y1:y2 + 1, x1:x2 + 1], new_size)

        foreground_likelihood = hist_target.back_project(img_channels_roi_inner) * fg_prior_scaled
        background_likelihood = hist_backgroud.back_project(img_channels_roi_inner) * bg_prior_scaled
        prob_o = p_o * foreground_likelihood / (p_o * foreground_likelihood + p_b * background_likelihood+1e-20)
        prob_b = 1 - prob_o
        sized_probs = Segment._get_regularized_segmentatioin(prob_o, prob_b, fg_prior_scaled, bg_prior_scaled)
        first = cv2.resize(sized_probs[0], (x2 + 1 - x1, y2 + 1 - y1))
        second = cv2.resize(sized_probs[1], (x2 + 1 - x1, y2 + 1 - y1))
        return first, second

    @staticmethod
    def _get_regularized_segmentatioin(prob_o,prob_b,prior_o,prior_b):
        hsize=int(np.floor(max(1,prob_b.shape[1]*3/50+0.5)))
        lambda_size=2*hsize+1
        lambda_=np.zeros((lambda_size,lambda_size))
        std2=(hsize/3)**2
        ys=np.arange(-hsize,hsize+1)
        xs=np.arange(-hsize,hsize+1)
        xs,ys=np.meshgrid(xs,ys)
        lambda_[ys+hsize,xs+hsize]=Segment._gaussian(xs**2,ys**2,std2)

        #lambda_=gaussian2d_labels((lambda_size,lambda_size),hsize/3)
        # set center of kernel to 0
        lambda_[hsize,hsize]=0
        lambda_=lambda_/np.sum(lambda_)

        # create lambda2 kernel
        import copy
        lambda_2=copy.deepcopy(lambda_)
        lambda_2[hsize,hsize]=1.
        terminate_thr=1e-1
        log_like=1e20
        max_iter=50
        Qsum_o=None
        Qsum_b=None
        for i in range(max_iter):
            P_Io=prior_o*prob_o+1.192*1e-7
            P_Ib=prior_b*prob_b+1.192*1e-7
            Si_o=cv2.filter2D(prior_o,-1,lambda_,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Si_b=cv2.filter2D(prior_b,-1,lambda_,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Si_o=Si_o*prior_o
            Si_b=Si_b*prior_b
            norm_Si=1/(Si_o+Si_b)
            Si_o=Si_o*norm_Si
            Si_b=Si_b*norm_Si
            Ssum_o=cv2.filter2D(Si_o,-1,lambda_2,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Ssum_b=cv2.filter2D(Si_b,-1,lambda_2,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)

            Qi_o=cv2.filter2D(P_Io,-1,lambda_,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Qi_b=cv2.filter2D(P_Ib,-1,lambda_,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Qi_o=Qi_o*P_Io
            Qi_b=Qi_b*P_Ib
            norm_Qi=1/(Qi_b+Qi_o)
            Qi_o=Qi_o*norm_Qi
            Qi_b=Qi_b*norm_Qi

            Qsum_o=cv2.filter2D(Qi_o,-1,lambda_2,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)
            Qsum_b=cv2.filter2D(Qi_b,-1,lambda_2,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_REFLECT)

            prior_o=(Qsum_o+Ssum_o)*0.25
            prior_b=(Qsum_b+Ssum_b)*0.25
            normPI=1/(prior_b+prior_o)
            prior_o=prior_o*normPI
            prior_b=prior_b*normPI

            logQo=cv2.log(Qsum_o)
            logQb=cv2.log(Qsum_b)
            mean=np.sum(logQo+logQb)
            loglike_new=-mean/(2*Qsum_o.shape[0]*Qsum_o.shape[1])
            if abs(log_like-loglike_new)<terminate_thr:
                break
            log_like=loglike_new
        return  Qsum_o,Qsum_b

    @staticmethod
    def _gaussian(x2,y2,std2):
        return np.exp(-(x2+y2)/(2*std2))/(2*np.pi*std2)




















