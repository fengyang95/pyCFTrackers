"""
Python re-implementation of "Adaptive Color Attributes for Real-Time Visual Tracking"
@inproceedings{Danelljan2014Adaptive,
  title={Adaptive Color Attributes for Real-Time Visual Tracking},
  author={Danelljan, Martin and Khan, Fahad Shahbaz and Felsberg, Michael and Weijer, Joost Van De},
  booktitle={IEEE Conference on Computer Vision & Pattern Recognition},
  year={2014},
}
"""
import numpy as np
import cv2
from .base import BaseCF
from lib.utils import gaussian2d_labels,cos_window
from lib.fft_tools import fft2,ifft2
from cftracker.feature import extract_cn_feature
from .config.cn_config import CNConfig
class CN(BaseCF):
    def __init__(self,config=CNConfig()):
        super(CN).__init__()
        self.interp_factor = config.interp_factor
        self.sigma = config.sigma
        self.lambda_ = config.lambda_
        self.padding=config.padding
        self.output_sigma_factor=config.output_sigma_factor


    def get_sub_window(self,im,pos,sz):
        patch=cv2.getRectSubPix(im,patchSize=sz,center=pos).astype(np.uint8)
        feature=extract_cn_feature(patch,cell_size=1)
        return feature


    def init(self,first_frame,bbox):
        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        self._window=cos_window((int(w*(1+self.padding)),int(h*(1+self.padding))))
        self.crop_size=(self._window.shape[1],self._window.shape[0])
        s=np.sqrt(w*h)*self.output_sigma_factor
        self.y=gaussian2d_labels(self.crop_size,s)
        self.yf=fft2(self.y)
        self._init_response_center=np.unravel_index(np.argmax(self.y,axis=None),self.y.shape)
        self.x=self.get_sub_window(first_frame, self._center, self.crop_size)
        self.x=self._window[:,:,None]*self.x

        kf=fft2(self._dgk(self.x,self.x))
        self.alphaf_num=(self.yf)*kf
        self.alphaf_den=kf*(kf+self.lambda_)


    def update(self,current_frame,vis=False):
        z=self.get_sub_window(current_frame,self._center,self.crop_size)
        z=self._window[:,:,None]*z
        kf=fft2(self._dgk(self.x,z))
        responses=np.real(ifft2(self.alphaf_num*kf.conj()/(self.alphaf_den)))
        if vis is True:
            self.score=responses
        curr=np.unravel_index(np.argmax(responses,axis=None),responses.shape)
        dy=self._init_response_center[0]-curr[0]
        dx=self._init_response_center[1]-curr[1]
        x_c, y_c = self._center
        x_c -= dx
        y_c -= dy
        self._center = (x_c, y_c)
        new_x=self.get_sub_window(current_frame,self._center,self.crop_size)
        new_x=new_x*self._window[:,:,None]

        kf = fft2(self._dgk(new_x,new_x))
        new_alphaf_num=self.yf*kf
        new_alphaf_den=kf*(kf+self.lambda_)
        self.alphaf_num=(1-self.interp_factor)*self.alphaf_num+self.interp_factor*new_alphaf_num
        self.alphaf_den=(1-self.interp_factor)*self.alphaf_den+self.interp_factor*new_alphaf_den
        self.x = (1 - self.interp_factor) * self.x + self.interp_factor * new_x
        return [self._center[0]-self.w/2,self._center[1]-self.h/2,self.w,self.h]

    def _dgk(self, x1, x2):
        xf = fft2(x1)
        yf = fft2(x2)
        xx=(x1.flatten().T).dot(x1.flatten())
        yy=(x2.flatten().T).dot(x2.flatten())
        xyf=xf*np.conj(yf)
        if len(xyf.shape)==2:
            xyf=xyf[:,:,np.newaxis]
        xy = np.real(ifft2(np.sum(xyf, axis=2)))
        d =xx + yy- 2 * xy
        k = np.exp(-1 / self.sigma ** 2 * np.clip(d,a_min=0,a_max=None) / np.size(x1))
        return k



