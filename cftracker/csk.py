"""
Python re-implementation of "Exploiting the Circulant Structure of
Tracking-by-detection with Kernels"
@book{Henriques2012Exploiting,
  title={Exploiting the Circulant Structure of Tracking-by-Detection with Kernels},
  author={Henriques, Jo?o F. and Rui, Caseiro and Martins, Pedro and Batista, Jorge},
  year={2012},
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
        self._window=cos_window((int(round(2*w)),int(round(2*h))))
        self.crop_size=(int(round(2*w)),int(round(2*h)))
        self.x=cv2.getRectSubPix(first_frame,(int(round(2*w)),int(round(2*h))),self._center)/255-0.5
        self.x=self.x*self._window
        s=np.sqrt(w*h)/16
        self.y=gaussian2d_labels((int(round(2*w)),int(round(2*h))),s)
        self._init_response_center=np.unravel_index(np.argmax(self.y,axis=None),self.y.shape)
        self.alphaf=self._training(self.x,self.y)

    def update(self,current_frame,vis=False):
        if len(current_frame.shape)==3:
            assert current_frame.shape[2]==3
            current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        current_frame=current_frame.astype(np.float32)
        z=cv2.getRectSubPix(current_frame,(int(round(2*self.w)),int(round(2*self.h))),self._center)/255-0.5
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
        return [self._center[0]-self.w/2,self._center[1]-self.h/2,self.w,self.h]


    def _dgk(self, x1, x2):
        c = np.fft.fftshift(ifft2(fft2(x1)* np.conj(fft2(x2))))
        d = np.dot(x1.flatten().conj(), x1.flatten()) + np.dot(x2.flatten().conj(), x2.flatten()) - 2 * c
        k = np.exp(-1 / self.sigma ** 2 * np.clip(d,a_min=0,a_max=None) / np.size(x1))
        return k

    def _training(self, x, y):
        k = self._dgk(x, x)
        alphaf = fft2(y) / (fft2(k) + self.lambda_)
        return alphaf

    def _detection(self, alphaf, x, z):
        k = self._dgk(x, z)
        responses = np.real(ifft2(alphaf * fft2(k)))
        return responses


