import numpy as np
import cv2
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

    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))# for vis
        self._center = (x0 + w / 2,y0 + h / 2)
        self.w, self.h = w, h
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        self._window = cos_window(self.window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        self.sz_id=0

        self.search_size=[1,0.985,0.990,0.995,1.005,1.01,1.015]

        self.target_sz=(w,h)
        param0=[self._center[0],self._center[1],self.crop_size[0]/self.crop_size[0],
                0,self.crop_size[1]/self.crop_size[0]/(self.crop_size[1]/self.crop_size[0]),
                0]
        param0=self.affparam2mat(param0)
        patch=self.warpimg(first_frame.astype(np.float32),param0,self.crop_size).astype(np.uint8)
        xf=fft2(self.get_features(patch,self.cell_size))
        xf=self._window[:,:,None]*xf
        kf=self._kernel_correlation(xf,xf,kernel=self.kernel)
        self.model_alphaf=self.yf/(kf+self.lambda_)
        self.model_xf=xf

    def update(self,current_frame,vis=False):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        response=None
        for i in range(len(self.search_size)):
            tmp_sz=(int(np.floor(self.target_sz[0]*(1+self.padding)*self.search_size[i])),
                    int(np.floor(self.target_sz[1]*(1+self.padding)*self.search_size[i])))
            param0=[self._center[0],self._center[1],tmp_sz[0]/self.crop_size[0],
                    0,tmp_sz[1]/self.crop_size[0]/(self.crop_size[1]/self.crop_size[0]),
                    0]
            param0=self.affparam2mat(param0)
            patch=self.warpimg(current_frame.astype(np.float32),param0,self.crop_size).astype(np.uint8)
            zf=fft2(self.get_features(patch,self.cell_size))
            zf=self._window[:,:,None]*zf
            kzf=self._kernel_correlation(zf,self.model_xf,kernel=self.kernel)
            if response is None:
                response=np.real(ifft2(self.model_alphaf*kzf))[:,:,np.newaxis]
            else:
                response=np.concatenate((response,np.real(ifft2(self.model_alphaf*kzf))[:,:,np.newaxis]),
                                        axis=2)
        delta_y,delta_x,sz_id = np.unravel_index(np.argmax(response, axis=None), response.shape)
        if vis is True:
            self.score=response[:,:,sz_id]
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
        self.sz_id=sz_id
        if delta_y+1>self.window_size[1]/2:
            delta_y=delta_y-self.window_size[1]
        if delta_x+1>self.window_size[0]/2:
            delta_x=delta_x-self.window_size[0]
        tmp_sz=(int(np.floor(self.target_sz[0]*(1+self.padding)*self.search_size[self.sz_id])),
                int(np.floor(self.target_sz[1]*(1+self.padding)*self.search_size[self.sz_id])))
        current_size=tmp_sz[0]/self.crop_size[0]
        x,y=self._center
        x+=current_size*self.cell_size*delta_x
        y+=current_size*self.cell_size*delta_y
        self._center=(x,y)

        self.target_sz=(int(np.round(self.target_sz[0]*self.search_size[self.sz_id])),
                        int(np.round(self.target_sz[1]*self.search_size[self.sz_id])))

        tmp_sz=(int(np.floor(self.target_sz[0]*(1+self.padding))),
                int(np.floor(self.target_sz[1]*(1+self.padding))))

        param0 = [self._center[0], self._center[1], tmp_sz[0] / self.window_size[0],
                  0, tmp_sz[1] / self.window_size[0] / (self.window_size[1] / self.window_size[0]),
                  0]
        param0 = self.affparam2mat(param0)
        patch = self.warpimg(current_frame.astype(np.float32), param0, self.window_size).astype(np.uint8)


        xf = fft2(self.get_features(patch, self.cell_size))
        xf = self._window[:, :, None] * xf
        kf=self._kernel_correlation(xf,xf,kernel=self.kernel)
        alphaf=self.yf/(kf+self.lambda_)
        self.model_alphaf=(1-self.interp_factor)*self.model_alphaf+self.interp_factor*alphaf
        self.model_xf=(1-self.interp_factor)*self.model_xf+self.interp_factor*xf

        return [(self._center[0] - self.target_sz[0] / 2), (self._center[1] - self.target_sz[1] / 2),
                self.target_sz[0], self.target_sz[1]]

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

    def _crop(self,img,center,target_sz):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        w,h=target_sz
        """
        # the same as matlab code 
        w=int(np.floor((1+self.padding)*w))
        h=int(np.floor((1+self.padding)*h))
        xs=(np.floor(center[0])+np.arange(w)-np.floor(w/2)).astype(np.int64)
        ys=(np.floor(center[1])+np.arange(h)-np.floor(h/2)).astype(np.int64)
        xs[xs<0]=0
        ys[ys<0]=0
        xs[xs>=img.shape[1]]=img.shape[1]-1
        ys[ys>=img.shape[0]]=img.shape[0]-1
        cropped=img[ys,:][:,xs]
        """
        cropped=cv2.getRectSubPix(img,(int(np.floor((1+self.padding)*w)),int(np.floor((1+self.padding)*h))),center)
        return cropped


    def warpimg(self,img,p,sz):
        w,h=sz
        x,y=np.meshgrid(w,h)
        pos=np.reshape(np.concatenate((np.ones((w*h,1)),x[:,np.newaxis],y[np.newaxis]),axis=1).dot(
            np.array([[p[0],p[1]],[p[2],p[4]],[p[3],p[5]]])
        ),(h,w,2))
        c=img.shape[2]
        wimg=np.zeros((h,w,c))
        for i in range(c):
            wimg[:,:,i]=self.interp2(img[:,:,i],pos[:,:,0],pos[:,:,1])
        wimg[np.isnan(wimg)]=0
        return wimg

    def interp2(self,img,pos1,pos2):
        pass


    def affparam2mat(self,p):
        """
        converts 6 affine parameters to a 2x3 matrix
        :param p [dx,dy,sc,th,sr,phi]'
        :return: q [q(1),q(3),q(4);q(2),q(5),q(6)]
        """
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



