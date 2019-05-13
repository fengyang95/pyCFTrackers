"""
Python re-implemented of "Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking"
@inproceedings{li2018learning,
  title={Learning spatial-temporal regularized correlation filters for visual tracking},
  author={Li, Feng and Tian, Cheng and Zuo, Wangmeng and Zhang, Lei and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4904--4913},
  year={2018}
}
"""
import numpy as np
import cv2
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature
from .config import strdcf_hc_config
from .cf_utils import resp_newton,mex_resize,resize_dft2
from .scale_estimator import LPScaleEstimator,DSSTScaleEstimator

class STRCF(BaseCF):
    def __init__(self,config=strdcf_hc_config.STRDCFHCConfig()):
        super(STRCF).__init__()
        self.hog_cell_size = config.hog_cell_size
        #self.hog_compressed_dim = config.hog_compressed_dim
        self.hog_n_dim = config.hog_n_dim

        self.gray_cell_size = config.gray_cell_size
        self.cn_use_for_gray = config.cn_use_for_gray
        self.cn_cell_size = config.cn_cell_size
        self.cn_n_dim = config.cn_n_dim

        self.cell_size=self.hog_cell_size

        self.search_area_shape = config.search_area_shape
        self.search_area_scale=config.search_area_scale
        self.min_image_sample_size=config.min_image_sample_size
        self.max_image_sample_size=config.max_image_sample_size

        self.feature_downsample_ratio=config.feature_downsample_ratio
        self.reg_window_max=config.reg_window_max
        self.reg_window_min=config.reg_window_min
        self.alpha=config.alpha
        self.beta=config.beta
        self.p=config.p

        # detection parameters
        self.refinement_iterations=config.refinement_iterations
        self.newton_iterations=config.newton_iterations
        self.clamp_position=config.clamp_position

        # learning parameters
        self.output_sigma_factor=config.output_sigma_factor
        self.temporal_regularization_factor=config.temporal_regularization_factor

        # ADMM params
        self.admm_max_iterations=config.max_iterations
        self.init_penalty_factor=config.init_penalty_factor
        self.max_penalty_factor=config.max_penalty_factor
        self.penalty_scale_step=config.penalty_scale_step

        # scale parameters
        self.number_of_scales =config.number_of_scales
        self.scale_step=config.scale_step

        self.use_mex_resize=True

        self.scale_type=config.scale_type
        self.scale_config = config.scale_config

        self.normalize_power=config.normalize_power
        self.normalize_size=config.normalize_size
        self.normalize_dim=config.normalize_dim
        self.square_root_normalization=config.square_root_normalization
        self.config=config


    def init(self,first_frame,bbox):

        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self._center = (int(x0 + w / 2),int(y0 + h / 2))
        self.target_sz=(w,h)

        search_area=self.target_sz[0]*self.search_area_scale*self.target_sz[1]*self.search_area_scale
        self.sc=np.clip(1,a_min=np.sqrt(search_area/self.max_image_sample_size),a_max=np.sqrt(search_area/self.min_image_sample_size))

        self.base_target_sz=(self.target_sz[0]/self.sc,self.target_sz[1]/self.sc)

        if self.search_area_shape=='proportional':
            self.crop_size=(int(self.base_target_sz[0]*self.search_area_scale),int(self.base_target_sz[1]*self.search_area_scale))
        elif self.search_area_shape=='square':
            w=int(np.sqrt(self.base_target_sz[0]*self.base_target_sz[1])*self.search_area_scale)
            self.crop_size=(w,w)
        elif self.search_area_shape=='fix_padding':
            tmp=int(np.sqrt(self.base_target_sz[0]*self.search_area_scale+(self.base_target_sz[1]-self.base_target_sz[0])/4))+\
                (self.base_target_sz[0]+self.base_target_sz[1])/2
            self.crop_size=(self.base_target_sz[0]+tmp,self.base_target_sz[1]+tmp)
        else:
            raise ValueError
        output_sigma = np.sqrt(np.floor(self.base_target_sz[0]/self.cell_size)*np.floor(self.base_target_sz[1]*self.cell_size))*\
            self.output_sigma_factor

        self.crop_size = (int(round(self.crop_size[0] / self.cell_size) * self.cell_size),
                          int(round(self.crop_size[1] / self.cell_size) * self.cell_size))
        self.feature_map_sz = (self.crop_size[0] // self.cell_size, self.crop_size[1] // self.cell_size)
        y=gaussian2d_rolled_labels(self.feature_map_sz,output_sigma)

        self.cosine_window=(cos_window((y.shape[1],y.shape[0])))
        self.yf=fft2(y)
        reg_scale=(int(np.floor(self.base_target_sz[0]/self.feature_downsample_ratio)),
                   int(np.floor(self.base_target_sz[1] / self.feature_downsample_ratio)))
        use_sz = self.feature_map_sz

        #self.reg_window=self.create_reg_window(reg_scale,use_sz,self.p,self.reg_window_max,
        #                                      self.reg_window_min,self.alpha,self.beta)
        self.reg_window=self.create_reg_window_const(reg_scale,use_sz,self.reg_window_max,self.reg_window_min)

        self.ky = np.roll(np.arange(-int(np.floor((self.feature_map_sz[1] - 1) / 2)),
                                    int(np.ceil((self.feature_map_sz[1] - 1) / 2 + 1))),
                          -int(np.floor((self.feature_map_sz[1] - 1) / 2)))
        self.kx = np.roll(np.arange(-int(np.floor((self.feature_map_sz[0] - 1) / 2)),
                                    int(np.ceil((self.feature_map_sz[0] - 1) / 2 + 1))),
                          -int(np.floor((self.feature_map_sz[0] - 1) / 2)))

        # scale
        scale_exp=np.arange(-int(np.floor((self.number_of_scales-1)/2)),int(np.ceil((self.number_of_scales-1)/2)+1))

        self.scale_factors=self.scale_step**scale_exp


        if self.number_of_scales>0:
            self._min_scale_factor = self.scale_step ** np.ceil(
                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self.scale_step))
            self._max_scale_factor = self.scale_step ** np.floor(np.log(np.min(
                first_frame.shape[:2] / np.array([self.base_target_sz[1], self.base_target_sz[0]]))) / np.log(
                self.scale_step))
            #print(self._min_scale_factor)
            #print(self._max_scale_factor)

        if self.scale_type=='normal':
            self.scale_estimator = DSSTScaleEstimator(self.target_sz, config=self.scale_config)
            self.scale_estimator.init(first_frame, self._center, self.base_target_sz, self.sc)
            self._num_scales = self.scale_estimator.num_scales
            self._scale_step = self.scale_estimator.scale_step

            self._min_scale_factor = self._scale_step ** np.ceil(
                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(
                first_frame.shape[:2] / np.array([self.base_target_sz[1], self.base_target_sz[0]]))) / np.log(
                self._scale_step))
        elif self.scale_type=='LP':
            self.scale_estimator=LPScaleEstimator(self.target_sz,config=self.scale_config)
            self.scale_estimator.init(first_frame,self._center,self.base_target_sz,self.sc)


        patch = self.get_sub_window(first_frame, self._center, model_sz=self.crop_size,
                                    scaled_sz=(int(np.round(self.crop_size[0] * self.sc)),
                                               int(np.round(self.crop_size[1] * self.sc))))
        xl_hc = self.extrac_hc_feature(patch, self.cell_size)
        xlf_hc = fft2(xl_hc * self.cosine_window[:, :, None])
        f_pre_f_hc=np.zeros_like(xlf_hc)
        mu_hc=0
        self.f_pre_f_hc=self.ADMM(xlf_hc,f_pre_f_hc,mu_hc)


    def update(self,current_frame,vis=False):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        old_pos=(np.inf,np.inf)
        iter=1
        while iter<=self.refinement_iterations and (np.abs(old_pos[0]-self._center[0])>1e-2 or
                                                    np.abs(old_pos[1]-self._center[1])>1e-2):

            sample_scales=self.sc*self.scale_factors
            xt_hc = None
            sample_pos=(int(np.round(self._center[0])),int(np.round(self._center[1])))
            for scale in sample_scales:
                sub_window = self.get_sub_window(current_frame, sample_pos, model_sz=self.crop_size,
                                                 scaled_sz=(int(round(self.crop_size[0] * scale)),
                                                            int(round(self.crop_size[1] * scale))))
                hc_features=self.extrac_hc_feature(sub_window, self.cell_size)[:,:,:,np.newaxis]
                if xt_hc is None:
                    xt_hc = hc_features
                else:
                    xt_hc = np.concatenate((xt_hc, hc_features), axis=3)
            xtw_hc=xt_hc*self.cosine_window[:,:,None,None]
            xtf_hc=fft2(xtw_hc)
            responsef_hc=np.sum(np.conj(self.f_pre_f_hc)[:,:,:,None]*xtf_hc,axis=2)
            responsef=responsef_hc
            response = np.real(ifft2(responsef))


            disp_row,disp_col,sind=resp_newton(response,responsef,self.newton_iterations,self.ky,self.kx,self.feature_map_sz)

            #row, col, sind = np.unravel_index(np.argmax(response, axis=None), response.shape)

            #disp_row = (row+ int(np.floor(self.feature_map_sz[1] - 1) / 2)) % self.feature_map_sz[1] - int(
            #    np.floor((self.feature_map_sz[1] - 1) / 2))
            #disp_col = (col + int(np.floor(self.feature_map_sz[0] - 1) / 2)) % self.feature_map_sz[0] - int(
            #    np.floor((self.feature_map_sz[0] - 1) / 2))

            if vis is True:
                self.score = response[:, :, sind].astype(np.float32)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)


            dx, dy = (disp_col * self.cell_size*self.sc*self.scale_factors[sind]), (disp_row * self.cell_size*self.sc*self.scale_factors[sind])
            scale_change_factor=self.scale_factors[sind]
            old_pos = self._center
            self._center = (sample_pos[0] +dx, sample_pos[1] + dy)
            self.sc=self.sc*scale_change_factor
            self.sc = np.clip(self.sc, self._min_scale_factor, self._max_scale_factor)
            self.sc = self.scale_estimator.update(current_frame, self._center, self.base_target_sz,
                                                  self.sc)
            if self.scale_type == 'normal':
                self.sc = np.clip(self.sc, a_min=self._min_scale_factor,
                                                    a_max=self._max_scale_factor)
            iter+=1
        sample_pos=(int(np.round(self._center[0])),int(np.round(self._center[1])))
        patch = self.get_sub_window(current_frame, sample_pos, model_sz=self.crop_size,
                                         scaled_sz=(int(np.round(self.crop_size[0] * self.sc)),
                                                    int(np.round(self.crop_size[1] * self.sc))))
        xl_hc = self.extrac_hc_feature(patch, self.cell_size)
        xlw_hc = xl_hc * self.cosine_window[:, :, None]
        xlf_hc = fft2(xlw_hc)
        mu = self.temporal_regularization_factor
        self.f_pre_f_hc=self.ADMM(xlf_hc,self.f_pre_f_hc,mu)
        target_sz=(self.base_target_sz[0]*self.sc,self.base_target_sz[1]*self.sc)
        return [(self._center[0] - (target_sz[0]) / 2), (self._center[1] -(target_sz[1]) / 2), target_sz[0],target_sz[1]]

    def extrac_hc_feature(self,patch,cell_size,normalization=False):
        hog_features=extract_hog_feature(patch,cell_size)
        cn_features=extract_cn_feature(patch,cell_size)
        hc_features=np.concatenate((hog_features,cn_features),axis=2)
        if normalization is True:
            hc_features=self._feature_normalization(hc_features)
        return hc_features

    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))

        """
        w,h=sz
        xs = (np.floor(center[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(center[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        im_patch = img[ys, :][:, xs]
        """
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = mex_resize(im_patch, model_sz)
        return im_patch.astype(np.uint8)

    def ADMM(self,xlf,f_pre_f,mu):
        model_xf = xlf
        f_f = np.zeros_like(model_xf)
        g_f = np.zeros_like(f_f)
        h_f = np.zeros_like(f_f)
        gamma =self.init_penalty_factor
        gamma_max =self.max_penalty_factor
        gamma_scale_step = self.penalty_scale_step
        T = self.feature_map_sz[0] * self.feature_map_sz[1]
        S_xx = np.sum(np.conj(model_xf) * model_xf, axis=2)
        Sf_pre_f = np.sum(np.conj(model_xf) * f_pre_f, axis=2)
        Sfx_pre_f = model_xf * Sf_pre_f[:, :, None]
        iter = 1
        while iter <= self.admm_max_iterations:
            B = S_xx + T * (gamma + mu)
            Sgx_f = np.sum(np.conj(model_xf) * g_f, axis=2)
            Shx_f = np.sum(np.conj(model_xf) * h_f, axis=2)

            tmp0 = (1 / (T * (gamma + mu)) * (self.yf[:, :, None] * model_xf)) - ((1 / (gamma + mu)) * h_f) + (
                    gamma / (gamma + mu)) * g_f + \
                   (mu / (gamma + mu)) * f_pre_f
            tmp1 = 1 / (T * (gamma + mu)) * (model_xf * ((S_xx * self.yf)[:, :, None]))
            tmp2 = mu / (gamma + mu) * Sfx_pre_f
            tmp3 = 1 / (gamma + mu) * (model_xf * (Shx_f[:, :, None]))
            tmp4 = gamma / (gamma + mu) * (model_xf * Sgx_f[:, :, None])
            f_f = tmp0 - (tmp1 + tmp2 - tmp3 +tmp4) / B[:, :, None]
            g_f = fft2(self.argmin_g(self.reg_window, gamma, (ifft2(gamma * (f_f + h_f)))))
            h_f = h_f + (gamma * (f_f - g_f))
            gamma = min(gamma_scale_step * gamma, gamma_max)
            iter += 1
        return f_f


    def argmin_g(self,w0, zeta, X):
        lhd = 1 / (w0 ** 2 + zeta)
        T = lhd[:, :, None] * X
        return T

    def create_reg_window(self,reg_scale,use_sz,p,reg_window_max,reg_window_min,alpha,beta):
        range_ = np.zeros((2, 2))
        for j in range(len(use_sz)):
            if use_sz[0]%2==1 and use_sz[1]%2==1:
                if int(reg_scale[j]) % 2 == 1:
                    range_[j, :] = np.array([-np.floor(use_sz[j] / 2), np.floor(use_sz[j] / 2)])
                else:
                    range_[j, :] = np.array([-(use_sz[j] / 2 - 1), (use_sz[j] / 2)])
            else:
                if int(reg_scale[j]) % 2 == 1:
                    range_[j, :] = np.array([-np.floor(use_sz[j] / 2), (np.floor((use_sz[j] - 1) / 2))])
                else:
                    range_[j, :] = np.array([-((use_sz[j] - 1) / 2),((use_sz[j] - 1) / 2)])
        wrs = np.arange(range_[1, 0], range_[1, 1] + 1)
        wcs = np.arange(range_[0, 0], range_[0, 1] + 1)
        wrs, wcs = np.meshgrid(wrs, wcs)
        res = (np.abs(wrs) / reg_scale[1]) ** p + (np.abs(wcs) / reg_scale[0]) ** p
        reg_window = reg_window_max / (1 + np.exp(-1. * alpha * (np.power(res, 1. / p) -beta))) +reg_window_min
        reg_window=reg_window.T
        return reg_window

    def create_reg_window_const(self, reg_scale, use_sz,reg_window_max, reg_window_min):
        reg_window=np.ones((use_sz[1],use_sz[0]))*reg_window_max
        range_=np.zeros((2,2))
        for j in range(2):
            range_[j,:]=np.array([0,reg_scale[j]-1])-np.floor(reg_scale[j]/2)
        cx=int(np.floor((use_sz[0]+1)/2))+(use_sz[0]+1)%2-1
        cy=int(np.floor((use_sz[1]+1)/2))+(use_sz[1]+1)%2-1
        range_h=np.arange(cy+range_[1,0],cy+range_[1,1]+1).astype(np.int)
        range_w=np.arange(cx+range_[0,0],cx+range_[0,1]+1).astype(np.int)
        range_h,range_w=np.meshgrid(range_h,range_w)
        reg_window[range_h,range_w]=reg_window_min
        return reg_window

    def _feature_normalization(self, x):
        if hasattr(self.config, 'normalize_power') and self.config.normalize_power > 0:
            if self.config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** self.config.normalize_size * (x.shape[2] ** self.config.normalize_dim) / (x ** 2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** self.config.normalize_size) * (x.shape[2] ** self.config.normalize_dim) / ((np.abs(x) ** (1. / self.config.normalize_power)).sum(axis=(0, 1, 2)))

        if self.config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)









