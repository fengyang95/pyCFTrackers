import numpy as np
import scipy
import cv2
from numpy.fft import fft, ifft
from scipy import signal
from lib.eco.fourier_tools import resize_dft
from .feature import extract_hog_feature
from lib.utils import cos_window
from lib.fft_tools import ifft2,fft2

class DSSTScaleEstimator:
    def __init__(self,target_sz,config):
        init_target_sz = np.array([target_sz[0],target_sz[1]])

        self.config=config
        num_scales = self.config.number_of_scales_filter
        scale_step = self.config.scale_step_filter
        scale_sigma = self.config.number_of_interp_scales * self.config.scale_sigma_factor

        scale_exp = np.arange(-np.floor(num_scales - 1)/2,
                              np.ceil(num_scales-1)/2+1,
                              dtype=np.float32) * self.config.number_of_interp_scales / num_scales
        scale_exp_shift = np.roll(scale_exp, (0, -int(np.floor((num_scales-1)/2))))

        interp_scale_exp = np.arange(-np.floor((self.config.number_of_interp_scales - 1) / 2),
                                     np.ceil((self.config.number_of_interp_scales - 1) / 2) + 1,
                                     dtype=np.float32)
        interp_scale_exp_shift = np.roll(interp_scale_exp, [0, -int(np.floor(self.config.number_of_interp_scales - 1) / 2)])

        self.scale_size_factors = scale_step ** scale_exp
        self.interp_scale_factors = scale_step ** interp_scale_exp_shift

        ys = np.exp(-0.5 * (scale_exp_shift ** 2) / (scale_sigma ** 2))
        self.yf = np.real(fft(ys))
        self.window = np.hanning(ys.shape[0]).T.astype(np.float32)
        # make sure the scale model is not to large, to save computation time


        self.num_scales = num_scales
        self.scale_step = scale_step

        if self.config.scale_model_factor ** 2 * np.prod(init_target_sz) > self.config.scale_model_max_area:
            scale_model_factor = np.sqrt(self.config.scale_model_max_area / np.prod(init_target_sz))
        else:
            scale_model_factor = self.config.scale_model_factor

        # set the scale model size
        self.scale_model_sz = np.maximum(np.floor(init_target_sz * scale_model_factor), np.array([8, 8]))
        self.max_scale_dim = self.config.s_num_compressed_dim == 'MAX'
        if self.max_scale_dim:
            self.s_num_compressed_dim = len(self.scale_size_factors)
        else:
            self.s_num_compressed_dim = self.config.s_num_compressed_dim



    def init(self,im,pos,base_target_sz,current_scale_factor):

        # self.scale_factors = np.array([1])
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)
        self.s_num = xs
        # compute projection basis
        if self.max_scale_dim:
            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
        else:
            U, _, _ = np.linalg.svd(self.s_num)
            self.basis = U[:, :self.s_num_compressed_dim]
            V, _, _ = np.linalg.svd(xs)
            scale_basis_den = V[:, :self.s_num_compressed_dim]
        self.basis = self.basis.T
        # compute numerator
        feat_proj = self.basis.dot(self.s_num) * self.window
        sf_proj = np.fft.fft(feat_proj, axis=1)
        self.sf_num = self.yf * np.conj(sf_proj)

        # update denominator
        xs = scale_basis_den.T.dot(xs)*self.window
        xsf = fft(xs, axis=1)
        new_sf_den = np.sum((xsf * np.conj(xsf)), 0)
        self.sf_den = new_sf_den


    def update(self, im, pos, base_target_sz, current_scale_factor):
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        # get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)

        # project
        xs = self.basis.dot(xs) * self.window

        # get scores
        xsf = np.fft.fft(xs, axis=1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + self.config.lamBda)
        interp_scale_response = np.real(ifft(resize_dft(scale_responsef, self.config.number_of_interp_scales)))
        recovered_scale_index = np.argmax(interp_scale_response)

        if self.config.do_poly_interp:
            # fit a quadratic polynomial to get a refined scale estimate
            id1 = (recovered_scale_index - 1) % self.config.number_of_interp_scales
            id2 = (recovered_scale_index + 1) % self.config.number_of_interp_scales
            poly_x = np.array([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index],
                               self.interp_scale_factors[id2]])
            poly_y = np.array(
                [interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]])
            poly_A = np.array([[poly_x[0] ** 2, poly_x[0], 1],
                               [poly_x[1] ** 2, poly_x[1], 1],
                               [poly_x[2] ** 2, poly_x[2], 1]], dtype=np.float32)
            poly = np.linalg.inv(poly_A).dot(poly_y.T)
            scale_change_factor = - poly[1] / (2 * poly[0])
        else:
            scale_change_factor = self.interp_scale_factors[recovered_scale_index]


        current_scale_factor=current_scale_factor*scale_change_factor

        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)
        self.s_num = (1 - self.config.scale_learning_rate) * self.s_num + self.config.scale_learning_rate * xs
        # compute projection basis
        if self.max_scale_dim:
            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
        else:
            U, _, _ = np.linalg.svd(self.s_num)
            self.basis = U[:, :self.s_num_compressed_dim]
            V,_,_=np.linalg.svd(xs)
            scale_basis_den=V[:,:self.s_num_compressed_dim]
        self.basis = self.basis.T

        # compute numerator
        feat_proj = self.basis.dot(self.s_num) * self.window
        sf_proj = np.fft.fft(feat_proj, axis=1)
        self.sf_num = self.yf * np.conj(sf_proj)

        # update denominator
        xs = scale_basis_den.T.dot(xs)*self.window
        xsf = np.fft.fft(xs, axis=1)
        new_sf_den = np.sum((xsf * np.conj(xsf)), 0)
        self.sf_den = (1 - self.config.scale_learning_rate) * self.sf_den + self.config.scale_learning_rate * new_sf_den
        return current_scale_factor


    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factors, scale_model_sz):
        scale_sample = []
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        for idx, scale in enumerate(scale_factors):
            patch_sz = np.floor(base_target_sz * scale)
            im_patch=cv2.getRectSubPix(im,(int(patch_sz[0]),int(patch_sz[1])),pos)
            if scale_model_sz[0] > patch_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA

            im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]),int(scale_model_sz[1])), interpolation=interpolation).astype(np.uint8)
            scale_sample.append(extract_hog_feature(im_patch_resized,cell_size=4).reshape((-1, 1)))
        scale_sample = np.concatenate(scale_sample, axis=1)
        return scale_sample


class LPScaleEstimator:
    def __init__(self,target_sz,config):
        self.learning_rate_scale=config.learning_rate_scale
        self.scale_sz_window = config.scale_sz_window
        self.target_sz=target_sz

    def init(self,im,pos,base_target_sz,current_scale_factor):
        w,h=base_target_sz
        avg_dim = (w + h) / 2.5
        self.scale_sz = ((w + avg_dim) / current_scale_factor,
                         (h + avg_dim) / current_scale_factor)
        self.scale_sz0 = self.scale_sz
        self.cos_window_scale = cos_window((self.scale_sz_window[0], self.scale_sz_window[1]))
        self.mag = self.cos_window_scale.shape[0] / np.log(np.sqrt((self.cos_window_scale.shape[0] ** 2 +
                                                                    self.cos_window_scale.shape[1] ** 2) / 4))

        # scale lp
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                 int(np.floor(current_scale_factor * self.scale_sz[1]))), pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        self.model_patchLp = extract_hog_feature(patchLp, cell_size=4)

    def update(self,im,pos,base_target_sz,current_scale_factor):
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                   int(np.floor(current_scale_factor* self.scale_sz[1]))),pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        # convert into logpolar
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patchLp = extract_hog_feature(patchLp, cell_size=4)
        tmp_sc, _, _ = self.estimate_scale(self.model_patchLp, patchLp, self.mag)
        tmp_sc = np.clip(tmp_sc, a_min=0.6, a_max=1.4)
        scale_factor=current_scale_factor*tmp_sc
        self.model_patchLp = (1 - self.learning_rate_scale) * self.model_patchLp + self.learning_rate_scale * patchLp
        return scale_factor

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

