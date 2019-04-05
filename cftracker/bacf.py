"""
Python re-implementation of "Learning Background-Aware Correlation Filters for Visual Tracking"
@article{Galoogahi2017Learning,
  title={Learning Background-Aware Correlation Filters for Visual Tracking},
  author={Galoogahi, Hamed Kiani and Fagg, Ashton and Lucey, Simon},
  year={2017},
}
"""
import numpy as np
import cv2
from .base import BaseCF
from lib.utils import cos_window,gaussian2d_rolled_labels
from lib.fft_tools import fft2,ifft2
from .feature import extract_hog_feature
from .config.bacf_config import BACFConfig


class BACF(BaseCF):
    def __init__(self, config=BACFConfig()):
        super(BACF).__init__()
        self.cell_size=config.cell_size
        self.cell_selection_thresh=config.cell_selection_thresh
        self.search_area_shape = config.search_area_shape
        self.search_area_scale=config.search_area_scale
        self.filter_max_area = config.filter_max_area
        self.interp_factor=config.interp_factor
        self.output_sigma_factor = config.output_sigma_factor
        self.interpolate_response =config.interpolate_response
        self.newton_iterations =config.newton_iterations
        self.number_of_scales =config.number_of_scales
        self.scale_step = config.scale_step
        self.admm_iterations = config.admm_iterations
        self.admm_lambda = config.admm_lambda


    def init(self,first_frame,bbox):
        bbox = np.array(bbox).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.feature_ratio=self.cell_size
        self.search_area=(self.w/self.feature_ratio*self.search_area_scale)*\
                         (self.h/self.feature_ratio*self.search_area_scale)
        if self.search_area<self.cell_selection_thresh*self.filter_max_area:
            self.cell_size=int(min(self.feature_ratio,max(1,int(np.ceil(np.sqrt(
                self.w*self.search_area_scale/(self.cell_selection_thresh*self.filter_max_area)*\
                self.h*self.search_area_scale/(self.cell_selection_thresh*self.filter_max_area)
            ))))))
            self.feature_ratio=self.cell_size
            self.search_area = (self.w / self.feature_ratio * self.search_area_scale) * \
                               (self.h / self.feature_ratio * self.search_area_scale)

        if self.search_area>self.filter_max_area:
            self.current_scale_factor=np.sqrt(self.search_area/self.filter_max_area)
        else:
            self.current_scale_factor=1.

        self.base_target_sz=(self.w/self.current_scale_factor,self.h/self.current_scale_factor)
        self.target_sz=self.base_target_sz
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
        self.crop_size=(int(round(self.crop_size[0]/self.feature_ratio)*self.feature_ratio),int(round(self.crop_size[1]/self.feature_ratio)*self.feature_ratio))
        self.feature_map_sz=(self.crop_size[0]//self.feature_ratio,self.crop_size[1]//self.feature_ratio)
        output_sigma=np.sqrt(np.floor(self.base_target_sz[0]/self.feature_ratio)*np.floor(self.base_target_sz[1]/self.feature_ratio))*self.output_sigma_factor
        y=gaussian2d_rolled_labels(self.feature_map_sz, output_sigma)
        self.yf=fft2(y)
        if self.interpolate_response==1:
            self.interp_sz=(self.feature_map_sz[0]*self.feature_ratio,self.feature_map_sz[1]*self.feature_ratio)
        else:
            self.interp_sz=(self.feature_map_sz[0],self.feature_map_sz[1])
        self._window=cos_window(self.feature_map_sz)
        if self.number_of_scales>0:
            scale_exp=np.arange(-int(np.floor((self.number_of_scales-1)/2)),int(np.ceil((self.number_of_scales-1)/2))+1)
            self.scale_factors=self.scale_step**scale_exp
            self.min_scale_factor=self.scale_step**(np.ceil(np.log(max(5/self.crop_size[0],5/self.crop_size[1]))/np.log(self.scale_step)))
            self.max_scale_factor=self.scale_step**(np.floor(np.log(min(first_frame.shape[0]/self.base_target_sz[1],
                                                                        first_frame.shape[1]/self.base_target_sz[0]))/np.log(self.scale_step)))
        if self.interpolate_response>=3:
            self.ky=np.roll(np.arange(-int(np.floor((self.feature_map_sz[1]-1)/2)),int(np.ceil((self.feature_map_sz[1]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[1]-1)/2)))
            self.kx=np.roll(np.arange(-int(np.floor((self.feature_map_sz[0]-1)/2)),int(np.ceil((self.feature_map_sz[0]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[0]-1)/2))).T

        self.small_filter_sz=(int(np.floor(self.base_target_sz[0]/self.feature_ratio)),int(np.floor(self.base_target_sz[1]/self.feature_ratio)))

        pixels=self.get_sub_window(first_frame,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(np.round(self.crop_size[0]*self.current_scale_factor)),
                                              int(np.round(self.crop_size[1]*self.current_scale_factor))))
        feature=extract_hog_feature(pixels, cell_size=self.feature_ratio)
        self.model_xf=fft2(self._window[:,:,None]*feature)

        self.g_f=self.ADMM(self.model_xf)



    def update(self,current_frame,vis=False):
        x=None
        for scale_ind in range(self.number_of_scales):
            current_scale=self.current_scale_factor*self.scale_factors[scale_ind]
            sub_window=self.get_sub_window(current_frame,self._center,model_sz=self.crop_size,
                                        scaled_sz=(int(round(self.crop_size[0]*current_scale)),
                                    int(round(self.crop_size[1]*current_scale))))
            feature= extract_hog_feature(sub_window, self.cell_size)[:, :, :, np.newaxis]
            if x is None:
                x=feature
            else:
                x=np.concatenate((x,feature),axis=3)
        xtf=fft2(x*self._window[:,:,None,None])
        responsef=np.sum(np.conj(self.g_f)[:,:,:,None]*xtf,axis=2)

        if self.interpolate_response==2:
            self.interp_sz=(int(self.yf.shape[1]*self.feature_ratio*self.current_scale_factor),
                            int(self.yf.shape[0]*self.feature_ratio*self.current_scale_factor))
        responsef_padded=self.resize_dft2(responsef,self.interp_sz)
        response=np.real(ifft2(responsef_padded))
        if self.interpolate_response==3:
            raise ValueError
        elif self.interpolate_response==4:
            disp_row,disp_col,sind=self.resp_newton(response,responsef_padded,self.newton_iterations,
                                                    self.ky,self.kx,self.feature_map_sz)
            if vis is True:
                self.score=response[:,:,sind]
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
        else:
            row,col,sind=np.unravel_index(np.argmax(response,axis=None),response.shape)

            if vis is True:
                self.score=response[:,:,sind]
                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
            disp_row=(row-1+int(np.floor(self.interp_sz[1]-1)/2))%self.interp_sz[1]-int(np.floor((self.interp_sz[1]-1)/2))
            disp_col = (col - 1 + int(np.floor(self.interp_sz[0] - 1) / 2)) % self.interp_sz[0] - int(
                np.floor((self.interp_sz[0] - 1) / 2))

        if self.interpolate_response==0  or self.interpolate_response==3 or self.interpolate_response==4:
            factor=self.feature_ratio*self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==1:
            factor=self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==2:
            factor=self.scale_factors[sind]
        else:
            raise ValueError
        dx,dy=int(np.round(disp_col*factor)),int(np.round(disp_row*factor))
        self.current_scale_factor=self.current_scale_factor*self.scale_factors[sind]
        self.current_scale_factor=max(self.current_scale_factor,self.min_scale_factor)
        self.current_scale_factor=min(self.current_scale_factor,self.max_scale_factor)
        self._center=(self._center[0]+dx,self._center[1]+dy)

        pixels=self.get_sub_window(current_frame,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(round(self.crop_size[0]*self.current_scale_factor)),
                                              int(round(self.crop_size[1]*self.current_scale_factor))))
        feature=extract_hog_feature(pixels, cell_size=self.cell_size)
        #feature=cv2.resize(pixels,self.feature_map_sz)/255-0.5
        xf=fft2(feature*self._window[:,:,None])
        self.model_xf=(1-self.interp_factor)*self.model_xf+self.interp_factor*xf
        self.g_f = self.ADMM(self.model_xf)

        target_sz=(int(self.target_sz[0]*self.current_scale_factor),int(self.target_sz[1]*self.current_scale_factor))
        return [int(self._center[0]-target_sz[0]/2),int(self._center[1]-target_sz[1]/2),target_sz[0],target_sz[1]]

    def get_subwindow_no_window(self,img,pos,sz):
        h,w=sz[1],sz[0]
        xs = (np.floor(pos[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(pos[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        out = img[ys, :][:, xs]
        xs,ys=np.meshgrid(xs,ys)
        return xs,ys,out

    def ADMM(self,xf):
        g_f = np.zeros_like(xf)
        h_f = np.zeros_like(g_f)
        l_f = np.zeros_like(g_f)
        mu = 1
        beta = 10
        mumax = 10000
        i = 1
        T = self.feature_map_sz[0] * self.feature_map_sz[1]
        S_xx = np.sum(np.conj(xf) * xf, 2)
        while i <= self.admm_iterations:
            B = S_xx + (T * mu)
            S_lx = np.sum(np.conj(xf) * l_f, axis=2)
            S_hx = np.sum(np.conj(xf) * h_f, axis=2)
            tmp0 = (1 / (T * mu) * (self.yf[:, :, None] * xf)) - ((1 / mu) * l_f)+ h_f
            tmp1 = 1 / (T * mu) * (xf * ((S_xx * self.yf)[:, :, None]))
            tmp2 = 1 / mu * (xf * (S_lx[:, :, None]))
            tmp3 = xf * S_hx[:, :, None]
            # solve for g
            g_f = tmp0 - (tmp1 - tmp2 + tmp3) / B[:, :, None]
            # solve for h
            h = (T / ((mu * T) + self.admm_lambda)) * ifft2(mu * g_f + l_f)
            xs, ys, h = self.get_subwindow_no_window(h,
                                                     (int(self.feature_map_sz[0] / 2), int(self.feature_map_sz[1] / 2)),
                                                     self.small_filter_sz)
            t = np.zeros((self.feature_map_sz[1], self.feature_map_sz[0], h.shape[2]),dtype=np.complex64)
            t[ys,xs,:] = h
            h_f = fft2(t)
            l_f = l_f + (mu * (g_f - h_f))
            mu = min(beta * mu, mumax)
            i += 1
        return g_f


    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = self.mex_resize(im_patch, model_sz)
        return im_patch.astype(np.uint8)

    def resize_dft2(self, input_dft, desired_sz):
        h,w,num_imgs=input_dft.shape
        if desired_sz[0]!=w or desired_sz[1]!=h:
            minsz=(int(min(w, desired_sz[0])), int(min(h, desired_sz[1])))
            scaling=(desired_sz[0]*desired_sz[1]/(h*w))
            resized_dfft=np.zeros((desired_sz[1],desired_sz[0],num_imgs),dtype=np.complex64)
            mids=(int(np.ceil(minsz[0]/2)),int(np.ceil(minsz[1]/2)))
            mide=(int(np.floor((minsz[0]-1)/2))-1,int(np.floor((minsz[1]-1)/2))-1)
            resized_dfft[:mids[1],:mids[0],:]=scaling*input_dft[:mids[1],:mids[0],:]
            resized_dfft[:mids[1],-1-mide[0]:-1,:]=scaling*input_dft[:mids[1],-1-mide[0]:-1,:]
            resized_dfft[-1-mide[1]:-1,:mids[0],:]=scaling*input_dft[-1-mide[1]:-1,:mids[0],:]
            resized_dfft[-1-mide[1]:-1,-1-mide[0]:-1,:]=scaling*input_dft[-1-mide[1]:-1,-1-mide[0]:-1,:]
            return resized_dfft
        else:
            return input_dft

    def mex_resize(self, img, sz):
        sz = (int(sz[0]), int(sz[1]))
        src_sz = (img.shape[1], img.shape[0])
        if sz[0] > src_sz[1]:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        img = cv2.resize(img, sz, interpolation=interpolation)
        return img

    """
    I didn't know how to convert matlab function mtimesx to numpy
    Just finetune from 4kubo's implementation
    https://github.com/4kubo/bacf_python/blob/master/special_operation/resp_newton.py
    """
    def resp_newton(self,response, responsef, iterations, ky, kx, use_sz):
        n_scale = response.shape[2]
        index_max_in_row = np.argmax(response, 0)
        max_resp_in_row = np.max(response, 0)
        index_max_in_col = np.argmax(max_resp_in_row, 0)
        init_max_response = np.max(max_resp_in_row, 0)
        col = index_max_in_col.flatten(order="F")

        max_row_perm = index_max_in_row
        row = max_row_perm[col, np.arange(n_scale)]

        trans_row = (row - 1 + np.floor((use_sz[1] - 1) / 2)) % use_sz[1] \
                    - np.floor((use_sz[1] - 1) / 2) + 1
        trans_col = (col - 1 + np.floor((use_sz[0] - 1) / 2)) % use_sz[0] \
                    - np.floor((use_sz[0] - 1) / 2) + 1
        init_pos_y = np.reshape(2 * np.pi * trans_row / use_sz[1], (1, 1, n_scale))
        init_pos_x = np.reshape(2 * np.pi * trans_col / use_sz[0], (1, 1, n_scale))
        max_pos_y = init_pos_y
        max_pos_x = init_pos_x

        # pre-compute complex exponential
        iky = 1j * ky
        exp_iky = np.tile(iky[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * \
                  np.tile(max_pos_y, (1, ky.shape[0], 1))
        exp_iky = np.exp(exp_iky)

        ikx = 1j * kx
        exp_ikx = np.tile(ikx[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * \
                  np.tile(max_pos_x, (kx.shape[0], 1, 1))
        exp_ikx = np.exp(exp_ikx)

        # gradient_step_size = gradient_step_size / prod(use_sz)

        ky2 = ky * ky
        kx2 = kx * kx

        iter = 1
        while iter <= iterations:
            # Compute gradient
            ky_exp_ky = np.tile(ky[np.newaxis, :, np.newaxis], (1, 1, exp_iky.shape[2])) * exp_iky
            kx_exp_kx = np.tile(kx[:, np.newaxis, np.newaxis], (1, 1, exp_ikx.shape[2])) * exp_ikx
            y_resp = np.einsum('ilk,ljk->ijk', exp_iky, responsef)
            resp_x = np.einsum('ilk,ljk->ijk', responsef, exp_ikx)
            grad_y = -np.imag(np.einsum('ilk,ljk->ijk', ky_exp_ky, resp_x))
            grad_x = -np.imag(np.einsum('ilk,ljk->ijk', y_resp, kx_exp_kx))
            ival = 1j * np.einsum('ilk,ljk->ijk', exp_iky, resp_x)
            H_yy = np.tile(ky2[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * exp_iky
            H_yy = np.real(-np.einsum('ilk,ljk->ijk', H_yy, resp_x) + ival)

            H_xx = np.tile(kx2[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * exp_ikx
            H_xx = np.real(-np.einsum('ilk,ljk->ijk', y_resp, H_xx) + ival)
            H_xy = np.real(-np.einsum('ilk,ljk->ijk', ky_exp_ky, np.einsum('ilk,ljk->ijk', responsef, kx_exp_kx)))
            det_H = H_yy * H_xx - H_xy * H_xy

            # Compute new position using newtons method
            diff_y = (H_xx * grad_y - H_xy * grad_x) / det_H
            diff_x = (H_yy * grad_x - H_xy * grad_y) / det_H
            max_pos_y = max_pos_y - diff_y
            max_pos_x = max_pos_x - diff_x

            # Evaluate maximum
            exp_iky = np.tile(iky[np.newaxis, :, np.newaxis], (1, 1, n_scale)) * \
                      np.tile(max_pos_y, (1, ky.shape[0], 1))
            exp_iky = np.exp(exp_iky)

            exp_ikx = np.tile(ikx[:, np.newaxis, np.newaxis], (1, 1, n_scale)) * \
                      np.tile(max_pos_x, (kx.shape[0], 1, 1))
            exp_ikx = np.exp(exp_ikx)

            iter = iter + 1

        max_response = 1 / np.prod(use_sz) * \
                       np.real(np.einsum('ilk,ljk->ijk',
                                         np.einsum('ilk,ljk->ijk', exp_iky, responsef),
                                         exp_ikx))

        # check for scales that have not increased in score
        ind = max_response < init_max_response
        max_response[0, 0, ind.flatten()] = init_max_response[ind.flatten()]
        max_pos_y[0, 0, ind.flatten()] = init_pos_y[0, 0, ind.flatten()]
        max_pos_x[0, 0, ind.flatten()] = init_pos_x[0, 0, ind.flatten()]

        sind = int(np.nanargmax(max_response, 2))
        disp_row = (np.mod(max_pos_y[0, 0, sind] + np.pi, 2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[1]
        disp_col = (np.mod(max_pos_x[0, 0, sind] + np.pi, 2 * np.pi) - np.pi) / (2 * np.pi) * use_sz[0]

        return disp_row, disp_col, sind










