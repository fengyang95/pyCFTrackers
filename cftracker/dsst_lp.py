"""
Replacing the scale estimator in DSST with Log-Polar estimator described in LDES
@article{li2017robust,
  title={Robust Estimation of Similarity Transformation for Visual Object Tracking},
  author={Li, Yang and Zhu, Jianke and Hoi, Steven CH and Song, Wenjie and Wang, Zhefeng and Liu, Hantang},
  journal={arXiv preprint arXiv:1712.05231},
  year={2017}
}
"""
import numpy as np
import cv2
from .base import BaseCF
from .feature import extract_hog_feature
from lib.utils import gaussian2d_labels,cos_window
from lib.fft_tools import fft2,ifft2

class DSST_LP(BaseCF):
    def __init__(self, interp_factor=0.025, sigma=0.2, lambda_=0.01,output_sigma_factor=1./16,
                 learning_rate_scale=0.015,scale_sz_window = (128, 128),
                 padding=1):
        super(DSST_LP).__init__()
        self.interp_factor = interp_factor
        self.sigma = sigma
        self.lambda_ = lambda_
        self.output_sigma_factor=output_sigma_factor
        self.learning_rate_scale=learning_rate_scale
        self.scale_sz_window=scale_sz_window
        self.padding=padding
        self.sc=1.

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

        patch=self.get_sub_window(first_frame,self._center,self.crop_size,self.sc)
        xl=self.get_feature_map(patch)
        xl=xl*self._window[:,:,None]
        self.xlf=fft2(xl)
        self.hf_den=np.sum(self.xlf*np.conj(self.xlf),axis=2)
        self.hf_num=self.yf[:,:,None]*np.conj(self.xlf)

        avg_dim = (w+h) / 2.5
        self.scale_sz = ((w + avg_dim) / self.sc,
                         (h + avg_dim) / self.sc)
        self.scale_sz0 = self.scale_sz
        self.cos_window_scale = cos_window((self.scale_sz_window[0], self.scale_sz_window[1]))
        self.mag = self.cos_window_scale.shape[0] / np.log(np.sqrt((self.cos_window_scale.shape[0] ** 2 +
                                                                    self.cos_window_scale.shape[1] ** 2) / 4))

        # scale lp
        patchL=cv2.getRectSubPix(first_frame,(int(np.floor(self.sc*self.scale_sz[0])),
                                                   int(np.floor(self.sc*self.scale_sz[1]))),self._center)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        self.model_patchLp = extract_hog_feature(patchLp, cell_size=4)


    def update(self,current_frame,vis=False):
        patch=self.get_sub_window(current_frame,self._center,self.crop_size,self.sc)
        xt=self.get_feature_map(patch)
        xt=xt*self._window[:,:,None]
        xtf=fft2(xt)
        response=np.real(ifft2(np.sum(self.hf_num*xtf,axis=2)/(self.hf_den+self.lambda_)))
        if vis is True:
            self.score=response
        curr=np.unravel_index(np.argmax(response,axis=None),response.shape)
        dy=(curr[0]-self._init_response_center[0])*self.sc
        dx=(curr[1]-self._init_response_center[1])*self.sc
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (x_c, y_c)

        patch=self.get_sub_window(current_frame,self._center,self.crop_size,self.sc)
        xl=self.get_feature_map(patch)
        xl=xl*self._window[:,:,None]
        xlf=fft2(xl)
        new_hf_num=self.yf[:,:,None]*np.conj(xlf)
        new_hf_den=np.sum(xlf*np.conj(xlf),axis=2)

        patchL = cv2.getRectSubPix(current_frame, (int(np.floor(self.sc * self.scale_sz[0])),
                                         int(np.floor(self.sc * self.scale_sz[1]))), self._center)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        # convert into logpolar
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patchLp = extract_hog_feature(patchLp, cell_size=4)
        tmp_sc, _, _ = self.estimate_scale(self.model_patchLp, patchLp, self.mag)
        tmp_sc=np.clip(tmp_sc,a_min=0.6,a_max=1.4)
        self.sc=self.sc*tmp_sc


        self.hf_den=(1-self.interp_factor)*self.hf_den+self.interp_factor*new_hf_den
        self.hf_num=(1-self.interp_factor)*self.hf_num+self.interp_factor*new_hf_num
        self.model_patchLp=(1-self.learning_rate_scale)*self.model_patchLp+self.learning_rate_scale*patchLp


        self.target_sz=(self.base_target_size[0]*self.sc,
                        self.base_target_size[1]*self.sc)

        return [self._center[0]-self.target_sz[0]/2,self._center[1]-self.target_sz[1]/2,
                self.target_sz[0],self.target_sz[1]]

    def get_sub_window(self,im,center,model_sz,scale_factor):
        patch_sz=(int(model_sz[0]*scale_factor),int(model_sz[1]*scale_factor))
        im_patch=cv2.getRectSubPix(im,patch_sz,center)
        if model_sz[0]>patch_sz[1]:
            interpolation=cv2.INTER_LINEAR
        else:
            interpolation=cv2.INTER_AREA
        im_patch=cv2.resize(im_patch,model_sz,interpolation=interpolation)
        return im_patch


    def get_feature_map(self,im_patch):
        gray=(cv2.cvtColor(im_patch,cv2.COLOR_BGR2GRAY))[:,:,np.newaxis]/255-0.5
        hog_feature= extract_hog_feature(im_patch, cell_size=1)[:, :, :27]
        return np.concatenate((gray,hog_feature),axis=2)

    def estimate_scale(self,model,obser,mag):
        def phase_correlation(src1,src2):
            s1f=fft2(src1)
            s2f=fft2(src2)
            num=s2f*np.conj(s1f)
            d=np.sqrt(num*np.conj(num))+2e-16
            Cf=np.sum(num/d,axis=2)
            C=np.real(ifft2(Cf))
            C=np.fft.fftshift(C,axes=(0,1))

            mscore=np.max(C)
            pty,ptx=np.unravel_index(np.argmax(C, axis=None), C.shape)
            slobe_y=slobe_x=1
            idy=np.arange(pty-slobe_y,pty+slobe_y+1).astype(np.int64)
            idx=np.arange(ptx-slobe_x,ptx+slobe_x+1).astype(np.int64)
            idy=np.clip(idy,a_min=0,a_max=C.shape[0]-1)
            idx=np.clip(idx,a_min=0,a_max=C.shape[1]-1)
            weight_patch=C[idy,:][:,idx]

            s=np.sum(weight_patch)+2e-16
            pty=np.sum(np.sum(weight_patch,axis=1)*idy)/s
            ptx=np.sum(np.sum(weight_patch,axis=0)*idx)/s
            pty=pty-(src1.shape[0])//2
            ptx=ptx-(src1.shape[1])//2
            return ptx,pty,mscore

        ptx,pty,mscore=phase_correlation(model,obser)
        rotate=pty*np.pi/(np.floor(obser.shape[1]/2))
        scale = np.exp(ptx/mag)
        return scale,rotate,mscore




