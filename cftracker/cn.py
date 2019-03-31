import numpy as np
import h5py
from .base import BaseCF
from lib.utils import gaussian2d_labels,cos_window
from lib.fft_tools import fft2,ifft2
from cftracker.feature import extract_cn_feature,extract_cn_feature_pyECO
class CN(BaseCF):
    def __init__(self, interp_factor=0.075, sigma=0.2, lambda_=0.01,compression_learning_rate=0.15,
                 use_dimensionality_reduction=False,num_compressed_dim=2):
        super(CN).__init__()
        self.interp_factor = interp_factor
        self.sigma = sigma
        self.lambda_ = lambda_
        self.compression_learning_rate=compression_learning_rate
        self.num_compressed_dim=num_compressed_dim
        self.use_dimensionality_reduction=use_dimensionality_reduction
        self.projection_matrix=np.diag(np.ones((10,)))
        self.cn_type='matlab'


    def get_sub_window(self,im,pos,sz,cn_type='matlab'):
        if cn_type=='matlab':
            features=extract_cn_feature(im,pos,sz,self.w2c)
        elif cn_type=='pyECO':
            features=extract_cn_feature_pyECO(im,pos,sz)
        else:
            raise NotImplementedError
        return features[:, :,0:1], features[:, :, 1:]


    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        cos_window = cos_window[:, :, np.newaxis]
        cos_windows = np.repeat(cos_window, img.shape[2], axis=2)
        windowed = cos_windows * img
        return windowed

    def feature_projection(self,x_npca,x_pca,projection_matrix,cos_window):
        if self.use_dimensionality_reduction is True:
            if x_pca is None:
                z = x_npca
            else:
                h, w = cos_window.shape[:2]
                x_pca=x_pca.reshape(h*w,-1)
                num_pca_in, num_pca_out = projection_matrix.shape[:2]
                x_proj_pca = x_pca.dot(projection_matrix).reshape(h, w, num_pca_out)
                if x_npca is None:
                    z = x_proj_pca
                else:
                    if len(x_npca.shape) == 2:
                        x_npca = x_npca[:, :, np.newaxis]
                    z = np.concatenate((x_npca, x_proj_pca), axis=2)

        else:
            assert x_pca is not None or x_npca is not None
            if x_pca is None:
                z=x_npca
            elif x_npca is None:
                z=x_pca
            else:
                if len(x_npca.shape)==2:
                    x_npca=x_npca[:,:,np.newaxis]
                z=np.concatenate((x_npca,x_pca),axis=2)
        z = self._get_windowed(z, self._window)
        return z

    def init(self,first_frame,bbox):
        mat = h5py.File('w2crs.mat')
        self.w2c = mat['w2crs']
        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        self._window=cos_window((2*w,2*h))
        self.crop_size=(2*w,2*h)
        s=np.sqrt(w*h)/16
        self.y=gaussian2d_labels((2*w,2*h),s)
        self.yf=fft2(self.y)
        self._init_response_center=np.unravel_index(np.argmax(self.y,axis=None),self.y.shape)
        self.x_npca, self.x_pca=self.get_sub_window(first_frame, self._center, self.crop_size,cn_type=self.cn_type)
        if self.use_dimensionality_reduction:
            pass
        self.x=self.feature_projection(self.x_npca, self.x_pca, self.projection_matrix, self._window)

        kf=fft2(self._dgk(self.x,self.x))
        self.alphaf_num=np.conj(self.yf)*kf
        self.alphaf_den=np.conj(kf)*(kf+self.lambda_)


    def update(self,current_frame,vis=False):
        z_npca,z_pca=self.get_sub_window(current_frame,self._center,self.crop_size,cn_type=self.cn_type)
        z=self.feature_projection(z_npca,z_pca,self.projection_matrix,self._window)
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
        new_x_npca, new_x_pca = self.get_sub_window(current_frame, self._center, self.crop_size,cn_type=self.cn_type)
        self.x_ncpa= (1-self.interp_factor) * self.x_npca + self.interp_factor * new_x_npca
        if new_x_pca is not None:
            self.x_pca= (1 - self.interp_factor) * self.x_pca + self.interp_factor * new_x_pca

        if self.use_dimensionality_reduction:
            pass
        self.x=self.feature_projection(self.x_ncpa,self.x_pca,self.projection_matrix,self._window)
        new_x = self.feature_projection(new_x_npca, new_x_pca, self.projection_matrix, self._window)
        kf = fft2(self._dgk(new_x,new_x))
        new_alphaf_num=np.conj(self.yf)*kf
        new_alphaf_den=np.conj(kf)*(kf+self.lambda_)
        self.alphaf_num=(1-self.interp_factor)*self.alphaf_num+self.interp_factor*new_alphaf_num
        self.alphaf_den=(1-self.interp_factor)*self.alphaf_den+self.interp_factor*new_alphaf_den
        return [int(self._center[0]-self.w/2),int(self._center[1]-self.h/2),self.w,self.h]

    def _dgk(self, x, y):
        xf = fft2(x)
        yf = fft2(y)
        xx=(x.flatten().conj()).dot(x.flatten())
        yy=(y.flatten().conj()).dot(y.flatten())
        xyf=xf*np.conj(yf)
        if len(xyf.shape)==2:
            xyf=xyf[:,:,np.newaxis]
        xy=np.real(ifft2(np.sum(xyf,axis=2)))
        d =xx + yy- 2 * xy
        k = np.exp(-1 / self.sigma ** 2 * np.clip(d,a_min=0,a_max=None) / np.size(x))
        return k



