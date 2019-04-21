
import numpy as np
import cv2
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature

class KCF(BaseCF):
    def __init__(self, padding=1.5, features='gray', kernel='gaussian'):
        super(KCF).__init__()
        self.padding = padding
        self.lambda_ = 1e-4
        self.features = features
        self.w2c=None
        if self.features=='hog':
            self.interp_factor = 0.02
            self.sigma = 0.5
            self.cell_size=4
            self.output_sigma_factor=0.1
        elif self.features=='gray' or self.features=='color':
            self.interp_factor=0.075
            self.sigma=0.2
            self.cell_size=1
            self.output_sigma_factor=0.1
        elif self.features=='cn':
            self.interp_factor=0.075
            self.sigma=0.2
            self.cell_size=1
            self.output_sigma_factor=1./16
            self.padding=1
        else:
            raise NotImplementedError
        self.kernel=kernel


    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        if self.features=='gray':
            first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))# for vis
        self._center = (np.floor(x0 + w / 2),np.floor(y0 + h / 2))
        self.w, self.h = w, h
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        self._window = cos_window(self.window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        if self.features=='gray' or self.features=='color':
            first_frame = first_frame.astype(np.float32) / 255
            x=self._crop(first_frame,self._center,(w,h))
            x=x-np.mean(x)
        elif self.features=='hog':
            x=self._crop(first_frame,self._center,(w,h))
            x=cv2.resize(x,(self.window_size[0]*self.cell_size,self.window_size[1]*self.cell_size))
            x=extract_hog_feature(x, cell_size=self.cell_size)
        elif self.features=='cn':
            x = cv2.resize(first_frame, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            x=extract_cn_feature(x,self.cell_size)
        else:
            raise NotImplementedError

        self.xf = fft2(self._get_windowed(x, self._window))
        self.init_response_center = (0,0)
        self.alphaf = self._training(self.xf,self.yf)


    def update(self,current_frame,vis=False):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features=='color' or self.features=='gray':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z=z-np.mean(z)

        elif self.features=='hog':
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_hog_feature(z, cell_size=self.cell_size)
        elif self.features=='cn':
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_cn_feature(z, cell_size=self.cell_size)
        else:
            raise NotImplementedError

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)
        if vis is True:
            self.score=responses
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        curr =np.unravel_index(np.argmax(responses, axis=None),responses.shape)

        if curr[0]+1>self.window_size[1]/2:
            dy=curr[0]-self.window_size[1]
        else:
            dy=curr[0]
        if curr[1]+1>self.window_size[0]/2:
            dx=curr[1]-self.window_size[0]
        else:
            dx=curr[1]
        dy,dx=dy*self.cell_size,dx*self.cell_size
        x_c, y_c = self._center
        x_c+= dx
        y_c+= dy
        self._center = (np.floor(x_c), np.floor(y_c))

        if self.features=='color' or self.features=='gray':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))
        elif self.features=='hog':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            new_x= extract_hog_feature(new_x, cell_size=self.cell_size)
        elif self.features=='cn':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            new_x = extract_cn_feature(new_x,cell_size=self.cell_size)
        else:
            raise NotImplementedError
        new_xf = fft2(self._get_windowed(new_x, self._window))
        self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (1 - self.interp_factor) * self.alphaf
        self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return [(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h]

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf/(kf+self.lambda_)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)
        responses = np.real(ifft2(alphaf * kzf))
        return responses

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

    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        windowed = cos_window[:,:,None] * img
        return windowed
