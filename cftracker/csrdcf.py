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
from lib.utils import gaussian2d_labels,cos_window
from lib.fft_tools import fft2,ifft2
class CSK(BaseCF):
    def __init__(self, interp_factor=0.075, sigma=0.2, lambda_=0.01):
        super(CSK).__init__()
        self.interp_factor = interp_factor
        self.sigma = sigma
        self.lambda_ = lambda_

    def init(self,first_frame,bbox):
        if len(first_frame.shape)==3:
            assert first_frame.shape[2]==3
            first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        first_frame=first_frame.astype(np.float32)
        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        self._window=cos_window((2*w,2*h))
        self.crop_size=(2*w,2*h)
        self.x=cv2.getRectSubPix(first_frame,(2*w,2*h),self._center)/255-0.5
        self.x=self.x*self._window
        s=np.sqrt(w*h)/16
        self.y=gaussian2d_labels((2*w,2*h),s)
        self._init_response_center=np.unravel_index(np.argmax(self.y,axis=None),self.y.shape)
        self.alphaf=self._training(self.x,self.y)

    def update(self,current_frame,vis=False):
        if len(current_frame.shape)==3:
            assert current_frame.shape[2]==3
            current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        current_frame=current_frame.astype(np.float32)
        z=cv2.getRectSubPix(current_frame,(2*self.w,2*self.h),self._center)/255-0.5
        z=z*self._window
        self.z=z
        responses=self._detection(self.alphaf,self.x,z)
        if vis is True:
            self.score=responses
        curr=np.unravel_index(np.argmax(responses,axis=None),responses.shape)
        dy=curr[0]-self._init_response_center[0]
        dx=curr[1]-self._init_response_center[1]
        x_c, y_c = self._center
        x_c -= dx
        y_c -= dy
        self._center = (x_c, y_c)
        new_x=cv2.getRectSubPix(current_frame,(2*self.w,2*self.h),self._center)/255-0.5
        new_x=new_x*self._window
        self.alphaf=self.interp_factor*self._training(new_x,self.y)+(1-self.interp_factor)*self.alphaf
        self.x=self.interp_factor*new_x+(1-self.interp_factor)*self.x
        return [int(self._center[0]-self.w/2),int(self._center[1]-self.h/2),self.w,self.h]





