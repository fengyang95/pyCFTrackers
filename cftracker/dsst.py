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

class DSST(BaseCF):
    def __init__(self, interp_factor=0.025, sigma=0.2, lambda_=0.01,output_sigma_factor=1./16,
                 scale_sigma_factor=1./4,num_of_scales=33,scale_step=1.02,scale_model_max_area=512,
                 padding=1):
        super(DSST).__init__()
        self.interp_factor = interp_factor
        self.sigma = sigma
        self.lambda_ = lambda_
        self.output_sigma_factor=output_sigma_factor
        self.scale_sigma_factor=scale_sigma_factor
        self.num_of_scales=num_of_scales
        self.scale_step=scale_step
        self.scale_model_max_area=scale_model_max_area
        self.padding=padding

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

        self.scale_sigma=self.num_of_scales/np.sqrt(33)*self.scale_sigma_factor
        ss=np.arange(1,self.num_of_scales+1)-np.ceil(self.num_of_scales/2)
        ys=np.exp(-0.5*(ss**2)/(self.scale_sigma**2))
        self.ysf=np.fft.fft(ys)

        if self.num_of_scales%2==0:
            scale_window=np.hanning(self.num_of_scales+1)
            self.scale_window=scale_window[1:]
        else:
            self.scale_window=np.hanning(self.num_of_scales)
        ss=np.arange(1,self.num_of_scales+1)
        self.scale_factors=self.scale_step**(np.ceil(self.num_of_scales/2)-ss)

        self.scale_model_factor=1.
        if (self.w*self.h)>self.scale_model_max_area:
            self.scale_model_factor=np.sqrt(self.scale_model_max_area/(self.w*self.h))

        self.scale_model_sz=(int(np.floor(self.w*self.scale_model_factor)),int(np.floor(self.h*self.scale_model_factor)))

        self.current_scale_factor=1.

        self.min_scale_factor=self.scale_step**(int(np.ceil(np.log(max(5/self.crop_size[0],5/self.crop_size[1]))/
                                                        np.log(self.scale_step))))
        self.max_scale_factor=self.scale_step**(int(np.floor((np.log(min(first_frame.shape[1]/self.w,first_frame.shape[0]/self.h))/
                                                          np.log(self.scale_step)))))


        xl=self.get_translation_sample(first_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        self.xlf=fft2(xl)
        self.hf_den=np.sum(self.xlf*np.conj(self.xlf),axis=2)
        self.hf_num=self.yf[:,:,None]*np.conj(self.xlf)

        xs=self.get_scale_sample(first_frame,self._center)
        xsf=np.fft.fft(xs,axis=1)
        self.sf_num=self.ysf*np.conj(xsf)
        self.sf_den=np.sum(xsf*np.conj(xsf),axis=0)


    def update(self,current_frame,vis=False):
        xt=self.get_translation_sample(current_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        xtf=fft2(xt)
        response=np.real(ifft2(np.sum(self.hf_num*xtf,axis=2)/(self.hf_den+self.lambda_)))
        if vis is True:
            self.score=response
        curr=np.unravel_index(np.argmax(response,axis=None),response.shape)
        dy=(curr[0]-self._init_response_center[0])*self.current_scale_factor
        dx=(curr[1]-self._init_response_center[1])*self.current_scale_factor
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (x_c, y_c)
        xs=self.get_scale_sample(current_frame, self._center)
        xsf=np.fft.fft(xs,axis=1)
        scale_response=np.real(np.fft.ifft(np.sum(self.sf_num*xsf,axis=0)/(self.sf_den+self.lambda_)))
        recovered_scale=np.argmax(scale_response)
        self.current_scale_factor=self.current_scale_factor*self.scale_factors[recovered_scale]
        self.current_scale_factor=np.clip(self.current_scale_factor,a_min=self.min_scale_factor,a_max=self.max_scale_factor)


        xl=self.get_translation_sample(current_frame,self._center,self.crop_size,self.current_scale_factor,self._window)
        xlf=fft2(xl)
        new_hf_num=self.yf[:,:,None]*np.conj(xlf)
        new_hf_den=np.sum(xlf*np.conj(xlf),axis=2)
        new_xs=self.get_scale_sample(current_frame,self._center)
        new_xsf=np.fft.fft(new_xs,axis=1)
        new_sf_num=self.ysf*np.conj(new_xsf)
        new_sf_den=np.sum(new_xsf*np.conj(new_xsf),axis=0)

        self.hf_den=(1-self.interp_factor)*self.hf_den+self.interp_factor*new_hf_den
        self.hf_num=(1-self.interp_factor)*self.hf_num+self.interp_factor*new_hf_num
        self.sf_den=(1-self.interp_factor)*self.sf_den+self.interp_factor*new_sf_den
        self.sf_num=(1-self.interp_factor)*self.sf_num+self.interp_factor*new_sf_num

        self.target_sz=(int(self.base_target_size[0]*self.current_scale_factor),
                        int(self.base_target_size[1]*self.current_scale_factor))

        return [int(self._center[0]-self.target_sz[0]/2),int(self._center[1]-self.target_sz[1]/2),
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

    def get_scale_sample(self,im,center):
        n_scales=len(self.scale_factors)
        out=None
        for s in range(n_scales):
            patch_sz=(int(self.base_target_size[0]*self.current_scale_factor*self.scale_factors[s]),
                      int(self.base_target_size[1]*self.current_scale_factor*self.scale_factors[s]))
            im_patch=cv2.getRectSubPix(im,patch_sz,center)
            if self.scale_model_sz[0] > patch_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            im_patch_resized=cv2.resize(im_patch,self.scale_model_sz,interpolation=interpolation).astype(np.uint8)
            tmp=extract_hog_feature(im_patch_resized, cell_size=4)
            if out is None:
                out=tmp.flatten()*self.scale_window[s]
            else:
                out=np.c_[out,tmp.flatten()*self.scale_window[s]]
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




