"""
Python re-implementation of "Accurate Scale Estimation for Robust Visual Tracking"
@inproceedings{DSST,
  author = {Danelljan, Martin and H?ger, Gustav and Khan, Fahad and Felsberg, Michael},
  title = {{Accurate Scale Estimation for Robust Visual Tracking}},
  booktitle = {Proceedings of the British Machine Vision Conference 2014},
  year = {2014},
  publisher = {BMVA Press},
}
"""
import numpy as np
import cv2
from .base import BaseCF
from .feature import extract_hog_feature
from lib.utils import gaussian2d_labels,cos_window
from lib.fft_tools import fft2,ifft2
from .scale_estimator import DSSTScaleEstimator,LPScaleEstimator

class DSST(BaseCF):
    def __init__(self,config):
        super(DSST).__init__()
        self.interp_factor = config.interp_factor
        self.sigma = config.sigma
        self.lambda_ = config.lambda_
        self.output_sigma_factor=config.output_sigma_factor
        self.scale_type=config.scale_type
        self.scale_config=config.scale_config
        self.padding =config.padding
        self.config=config


    def init(self,first_frame,bbox):
        first_frame=first_frame.astype(np.float32)
        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        self.crop_size = (int(w*(1+self.padding)), int(h*(1+self.padding)))
        self.base_target_size=(self.w, self.h)
        self.target_sz=(self.w,self.h)
        self._window=cos_window(self.crop_size)
        output_sigma=np.sqrt(self.w*self.h)*self.output_sigma_factor
        self.y=gaussian2d_labels(self.crop_size,output_sigma)
        self._init_response_center = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape)
        self.yf=fft2(self.y)
        self.current_scale_factor=1.


        xl=self.get_translation_sample(first_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        self.xlf=fft2(xl)
        self.hf_den=np.sum(self.xlf*np.conj(self.xlf),axis=2)
        self.hf_num=self.yf[:,:,None]*np.conj(self.xlf)

        if self.scale_type=='normal':
            self.scale_estimator = DSSTScaleEstimator(self.target_sz, config=self.scale_config)
            self.scale_estimator.init(first_frame, self._center, self.base_target_size, self.current_scale_factor)
            self._num_scales = self.scale_estimator.num_scales
            self._scale_step = self.scale_estimator.scale_step

            self._min_scale_factor = self._scale_step ** np.ceil(
                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(
                first_frame.shape[:2] / np.array([self.base_target_size[1], self.base_target_size[0]]))) / np.log(
                self._scale_step))
        elif self.scale_type=='LP':
            self.scale_estimator=LPScaleEstimator(self.target_sz,config=self.scale_config)
            self.scale_estimator.init(first_frame,self._center,self.base_target_size,self.current_scale_factor)

    def update(self,current_frame,vis=False):
        xt=self.get_translation_sample(current_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        xtf=fft2(xt)
        response=np.real(ifft2(np.sum(self.hf_num*xtf,axis=2)/(self.hf_den+self.lambda_)))
        if vis is True:
            self.score=response
            self.win_sz=self.crop_size
        curr=np.unravel_index(np.argmax(response,axis=None),response.shape)
        dy=(curr[0]-self._init_response_center[0])*self.current_scale_factor
        dx=(curr[1]-self._init_response_center[1])*self.current_scale_factor
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (x_c, y_c)

        self.current_scale_factor = self.scale_estimator.update(current_frame, self._center, self.base_target_size,
                                                                self.current_scale_factor)
        if self.scale_type == 'normal':
            self.current_scale_factor = np.clip(self.current_scale_factor, a_min=self._min_scale_factor,
                                                a_max=self._max_scale_factor)

        xl=self.get_translation_sample(current_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        xlf=fft2(xl)
        new_hf_num=self.yf[:,:,None]*np.conj(xlf)
        new_hf_den=np.sum(xlf*np.conj(xlf),axis=2)

        self.hf_den=(1-self.interp_factor)*self.hf_den+self.interp_factor*new_hf_den
        self.hf_num=(1-self.interp_factor)*self.hf_num+self.interp_factor*new_hf_num


        self.target_sz=(self.base_target_size[0]*self.current_scale_factor,
                        self.base_target_size[1]*self.current_scale_factor)

        return [self._center[0]-self.target_sz[0]/2,self._center[1]-self.target_sz[1]/2,
                self.target_sz[0],self.target_sz[1]]

    def get_translation_sample(self,im,center,model_sz,scale_factor,cos_window):
        patch_sz=(int(model_sz[0]*scale_factor),int(model_sz[1]*scale_factor))
        im_patch=cv2.getRectSubPix(im,patch_sz,center)
        if model_sz[0]>patch_sz[1]:
            interpolation=cv2.INTER_LINEAR
        else:
            interpolation=cv2.INTER_AREA
        im_patch=cv2.resize(im_patch,model_sz,interpolation=interpolation)
        out=self.get_feature_map(im_patch)
        out=self._get_windowed(out,cos_window)
        return out


    def get_feature_map(self,im_patch):
        gray=(cv2.cvtColor(im_patch,cv2.COLOR_BGR2GRAY))[:,:,np.newaxis]/255-0.5
        hog_feature= extract_hog_feature(im_patch, cell_size=1)[:, :, :27]
        return np.concatenate((gray,hog_feature),axis=2)

    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        windowed = cos_window[:,:,None] * img
        return windowed




