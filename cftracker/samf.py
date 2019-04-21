"""
Python re-implementation of "A Scale Adaptive Kernel Correlation Filter Tracker with Feature Integration"
@inproceedings{li2014scale,
  title={A scale adaptive kernel correlation filter tracker with feature integration},
  author={Li, Yang and Zhu, Jianke},
  booktitle={European conference on computer vision},
  pages={254--265},
  year={2014},
  organization={Springer}
}
"""
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature

class SAMF(BaseCF):
    def __init__(self,kernel='gaussian'):
        super(SAMF).__init__()
        self.padding = 1.5
        self.lambda_ = 1e-4
        self.output_sigma_factor=0.1
        self.interp_factor=0.01
        self.kernel_sigma=0.5
        self.cell_size=4
        self.kernel=kernel
        self.resize=False

    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        if w*h>=100**2:
            self.resize=True
            x0,y0,w,h=x0/2,y0/2,w/2,h/2
            first_frame=cv2.resize(first_frame,dsize=None,fx=0.5,fy=0.5).astype(np.uint8)

        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))# for vis
        self._center = (x0 + w / 2,y0 + h / 2)
        self.w, self.h = w, h
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        self._window = cos_window(self.window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        self.search_size=np.linspace(0.985,1.015,7)

        self.target_sz=(w,h)
        #param0=[self._center[0],self._center[1],1,
        #        0,1/(self.crop_size[1]/self.crop_size[0]),
        #        0]
        #param0=self.affparam2mat(param0)
        #patch=self.warpimg(first_frame.astype(np.float32),param0,self.crop_size).astype(np.uint8)
        patch = cv2.getRectSubPix(first_frame,self.crop_size, self._center)
        patch = cv2.resize(patch, dsize=self.crop_size)
        hc_features=self.get_features(patch,self.cell_size)
        hc_features=hc_features*self._window[:,:,None]
        xf=fft2(hc_features)
        kf=self._kernel_correlation(xf,xf,kernel=self.kernel)
        self.model_alphaf=self.yf/(kf+self.lambda_)
        self.model_xf=xf

    def update(self,current_frame,vis=False):
        if self.resize:
            current_frame=cv2.resize(current_frame,dsize=None,fx=0.5,fy=0.5).astype(np.uint8)
        response=None
        for i in range(len(self.search_size)):
            tmp_sz=(self.target_sz[0]*(1+self.padding)*self.search_size[i],
                    self.target_sz[1]*(1+self.padding)*self.search_size[i])
            #param0=[self._center[0],self._center[1],tmp_sz[0]/self.crop_size[0],
            #        0,tmp_sz[1]/self.crop_size[0]/(self.crop_size[1]/self.crop_size[0]),
            #        0]
            #param0=self.affparam2mat(param0)
            #patch=self.warpimg(current_frame.astype(np.float32),param0,self.crop_size).astype(np.uint8)
            patch=cv2.getRectSubPix(current_frame,(int(np.round(tmp_sz[0])),int(np.round(tmp_sz[1]))),self._center)
            patch = cv2.resize(patch, self.crop_size)
            hc_features=self.get_features(patch,self.cell_size)
            hc_features=hc_features*self._window[:,:,None]
            zf=fft2(hc_features)

            kzf=self._kernel_correlation(zf,self.model_xf,kernel=self.kernel)
            if response is None:
                response=np.real(ifft2(self.model_alphaf*kzf))[:,:,np.newaxis]
            else:
                response=np.concatenate((response,np.real(ifft2(self.model_alphaf*kzf))[:,:,np.newaxis]),
                                        axis=2)
        delta_y,delta_x,sz_id = np.unravel_index(np.argmax(response, axis=None), response.shape)
        self.sz_id=sz_id

        if vis is True:
            self.score=response[:,:,self.sz_id]
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        if delta_y+1>self.window_size[1]/2:
            delta_y=delta_y-self.window_size[1]
        if delta_x+1>self.window_size[0]/2:
            delta_x=delta_x-self.window_size[0]

        self.target_sz = (self.target_sz[0] * self.search_size[self.sz_id],
                          self.target_sz[1] * self.search_size[self.sz_id])
        tmp_sz=(self.target_sz[0]*(1+self.padding),
                self.target_sz[1]*(1+self.padding))
        current_size_factor=tmp_sz[0]/self.crop_size[0]
        x,y=self._center
        x+=current_size_factor*self.cell_size*delta_x
        y+=current_size_factor*self.cell_size*delta_y
        self._center=(x,y)

        #param0 = [self._center[0], self._center[1], tmp_sz[0] / self.crop_size[0],
        #          0, tmp_sz[1] / self.crop_size[0] / (self.crop_size[1] / self.crop_size[0]),
        #          0]
        #param0 = self.affparam2mat(param0)
        #patch = self.warpimg(current_frame.astype(np.float32), param0, self.crop_size).astype(np.uint8)
        patch = cv2.getRectSubPix(current_frame, (int(np.round(tmp_sz[0])), int(np.round(tmp_sz[1]))), self._center)
        patch=cv2.resize(patch,self.crop_size)
        hc_features=self.get_features(patch, self.cell_size)
        hc_features=self._window[:,:,None]*hc_features
        xf = fft2(hc_features)
        kf=self._kernel_correlation(xf,xf,kernel=self.kernel)
        alphaf=self.yf/(kf+self.lambda_)
        self.model_alphaf=(1-self.interp_factor)*self.model_alphaf+self.interp_factor*alphaf
        self.model_xf=(1-self.interp_factor)*self.model_xf+self.interp_factor*xf

        bbox=[(self._center[0] - self.target_sz[0] / 2), (self._center[1] - self.target_sz[1] / 2),
                self.target_sz[0], self.target_sz[1]]
        if self.resize is True:
            bbox=[ele*2 for ele in bbox]
        return bbox

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.kernel_sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def get_features(self,img,cell_size):
        hog_feature=extract_hog_feature(img,cell_size)
        cn_feature=extract_cn_feature(img,cell_size)
        return np.concatenate((hog_feature,cn_feature),axis=2)

    """
     def warpimg(self,img,p,sz):
        w,h=sz
        x,y=np.meshgrid(np.arange(w)-w/2+0.5,np.arange(h)-h/2)
        pos=np.reshape(np.concatenate((np.ones((w*h,1)),x.ravel()[:,np.newaxis],y.ravel()[:,np.newaxis]),axis=1).dot(
            np.array([[p[0],p[1]],[p[2],p[4]],[p[3],p[5]]])),(h,w,2),order='C')
        c=img.shape[2]
        wimg=np.zeros((h,w,c))
        for i in range(c):
            wimg[:,:,i]=self.interp2(img[:,:,i],pos[:,:,1],pos[:,:,0])
        wimg[np.isnan(wimg)]=0
        return wimg

    def interp2(self,img,Xq,Yq):
        wimg=map_coordinates(img,[Xq.ravel(),Yq.ravel()],order=1,mode='constant')
        wimg=wimg.reshape(Xq.shape)
        return wimg

    def affparam2mat(self,p):
        
        # converts 6 affine parameters to a 2x3 matrix
        # :param p [dx,dy,sc,th,sr,phi]'
        # :return: q [q(1),q(3),q(4);q(2),q(5),q(6)]
        _,_,s,th,r,phi=p
        cth,sth=np.cos(th),np.sin(th)
        cph,sph=np.cos(phi),np.sin(phi)
        ccc=cth*cph*cph
        ccs=cth*cph*sph
        css=cth*sph*sph
        scc=sth*cph*cph
        scs=sth*cph*sph
        sss=sth*sph*sph
        q0=p[0]
        q1=p[1]
        q2=s*(ccc+scs+r*(css-scs))
        q3=s*(r*(ccs-scc)-ccs-sss)
        q4=s*(scc-ccs+r*(ccs+sss))
        q5=s*(r*(ccc+scs)-scs+css)
        return [q0,q1,q2,q3,q4,q5]

    def mex_resize(self, img, sz,method='auto'):
        sz = (int(sz[0]), int(sz[1]))
        src_sz = (img.shape[1], img.shape[0])
        if method=='antialias':
            interpolation=cv2.INTER_AREA
        elif method=='linear':
            interpolation=cv2.INTER_LINEAR
        else:
            if sz[1] > src_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
        img = cv2.resize(img, sz, interpolation=interpolation)
        return img
    """




