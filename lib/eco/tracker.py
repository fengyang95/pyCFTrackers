import numpy as np
import cv2
import scipy
import time


from scipy import signal
# from numpy.fft import fftshift

from .config import gpu_config
from .features import GrayFeature,FHogFeature, TableFeature, mround, ResNet50Feature, VGG16Feature
from .fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff,\
        cubic_spline_fourier, compact_fourier_coeff, ifft2, fft2, sample_fs
from .optimize_score import optimize_score
from .sample_space_model import GMM
from .train import train_joint, train_filter
from .scale_filter import ScaleFilter
if gpu_config.use_gpu:
    import cupy as cp


class ECOTracker:
    def __init__(self, is_color,config):
        self._is_color = is_color
        self._frame_num = 0
        self._frames_since_last_train = 0
        if gpu_config.use_gpu:
            cp.cuda.Device(gpu_config.gpu_id).use()
        self.config=config

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]+2))[:, np.newaxis].dot(np.hanning(int(size[1]+2))[np.newaxis, :])
        cos_window = cos_window[1:-1, 1:-1][:, :, np.newaxis, np.newaxis].astype(np.float32)
        if gpu_config.use_gpu:
            cos_window = cp.asarray(cos_window)
        return cos_window

    def _get_interp_fourier(self, sz):
        """
            compute the fourier series of the interpolation function.
        """
        f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis] / sz[0]
        interp1_fs = np.real(cubic_spline_fourier(f1, self.config.interp_bicubic_a) / sz[0])
        f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :] / sz[1]
        interp2_fs = np.real(cubic_spline_fourier(f2, self.config.interp_bicubic_a) / sz[1])
        if self.config.interp_centering:
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis]
            interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * f1)
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :]
            interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * f2)

        if self.config.interp_windowing:
            win1 = np.hanning(sz[0]+2)[:, np.newaxis]
            win2 = np.hanning(sz[1]+2)[np.newaxis, :]
            interp1_fs = interp1_fs * win1[1:-1]
            interp2_fs = interp2_fs * win2[1:-1]
        if not gpu_config.use_gpu:
            return (interp1_fs[:, :, np.newaxis, np.newaxis],
                    interp2_fs[:, :, np.newaxis, np.newaxis])
        else:
            return (cp.asarray(interp1_fs[:, :, np.newaxis, np.newaxis]),
                    cp.asarray(interp2_fs[:, :, np.newaxis, np.newaxis]))

    def _get_reg_filter(self, sz, target_sz, reg_window_edge):
        """
            compute the spatial regularization function and drive the
            corresponding filter operation used for optimization
        """
        if self.config.use_reg_window:
            # normalization factor
            reg_scale = 0.5 * target_sz

            # construct grid
            wrg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wcg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wrs, wcs = np.meshgrid(wrg, wcg)

            # construct the regularization window
            reg_window = (reg_window_edge - self.config.reg_window_min) * (np.abs(wrs / reg_scale[0]) ** self.config.reg_window_power + \
                                                                          np.abs(wcs/reg_scale[1]) ** self.config.reg_window_power) + self.config.reg_window_min

            # compute the DFT and enforce sparsity
            reg_window_dft = fft2(reg_window) / np.prod(sz)
            reg_window_dft[np.abs(reg_window_dft) < self.config.reg_sparsity_threshold * np.max(np.abs(reg_window_dft.flatten()))] = 0

            # do the inverse transform, correct window minimum
            reg_window_sparse = np.real(ifft2(reg_window_dft))
            reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + self.config.reg_window_min
            reg_window_dft = np.fft.fftshift(reg_window_dft).astype(np.complex64)

            # find the regularization filter by removing the zeros
            row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
            col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
            mask = np.outer(row_idx, col_idx)
            reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        else:
            # else use a scaled identity matrix
            reg_filter = self.config.reg_window_min
        if not gpu_config.use_gpu:
            return reg_filter.T
        else:
            return cp.asarray(reg_filter.T)

    def _init_proj_matrix(self, init_sample, compressed_dim, proj_method):
        """
            init the projection matrix
        """
        if gpu_config.use_gpu:
            xp = cp.get_array_module(init_sample[0])
        else:
            xp = np
        x = [xp.reshape(x, (-1, x.shape[2])) for x in init_sample]
        x = [z - z.mean(0) for z in x]
        proj_matrix_ = []
        if self.config.proj_init_method == 'pca':
            for x_, compressed_dim_  in zip(x, compressed_dim):
                proj_matrix, _, _ = xp.linalg.svd(x_.T.dot(x_))
                proj_matrix = proj_matrix[:, :compressed_dim_]
                proj_matrix_.append(proj_matrix)
        elif self.config.proj_init_method == 'rand_uni':
            for x_, compressed_dim_ in zip(x, compressed_dim):
                proj_matrix = xp.random.uniform(size=(x_.shape[1], compressed_dim_))
                proj_matrix /= xp.sqrt(xp.sum(proj_matrix**2, axis=0, keepdims=True))
                proj_matrix_.append(proj_matrix)
        return proj_matrix_

    def _proj_sample(self, x, P):
        if gpu_config.use_gpu:
            xp = cp.get_array_module(x[0])
        else:
            xp = np
        return [xp.matmul(P_.T, x_) for x_, P_ in zip(x, P)]

    def init(self, frame, bbox, total_frame=np.inf):
        """
            frame -- image
            bbox -- need xmin, ymin, width, height
        """
        self._pos = np.array([bbox[1]+(bbox[3]-1)/2., bbox[0]+(bbox[2]-1)/2.], dtype=np.float32)
        self._target_sz = np.array([bbox[3], bbox[2]])
        self._num_samples = min(self.config.num_samples, total_frame)
        xp = cp if gpu_config.use_gpu else np

        # calculate search area and initial scale factor
        search_area = np.prod(self._target_sz * self.config.search_area_scale)
        if search_area > self.config.max_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / self.config.max_image_sample_size)
        elif search_area < self.config.min_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / self.config.min_image_sample_size)
        else:
            self._current_scale_factor = 1.

        # target size at the initial scale
        self._base_target_sz = self._target_sz / self._current_scale_factor

        # target size, taking padding into account
        if self.config.search_area_shape == 'proportional':
            self._img_sample_sz = np.floor(self._base_target_sz * self.config.search_area_scale)
        elif self.config.search_area_shape == 'square':
            self._img_sample_sz = np.ones((2), dtype=np.float32) * np.sqrt(np.prod(self._base_target_sz * self.config.search_area_scale))
        else:
            raise("unimplemented")

        features = [feature for feature in self.config.features
                    if ("use_for_color" in feature and feature["use_for_color"] == self._is_color) or
                    "use_for_color" not in feature]

        self._features = []
        cnn_feature_idx = -1
        for idx, feature in enumerate(features):
            if feature['fname'] == 'cn' or feature['fname'] == 'ic':
                self._features.append(TableFeature(**feature))
            elif feature['fname'] == 'fhog':
                self._features.append(FHogFeature(**feature))
            elif feature['fname']=='gray':
                self._features.append(GrayFeature(**feature))
            elif feature['fname'].startswith('cnn'):
                cnn_feature_idx = idx
                netname = feature['fname'].split('-')[1]
                if netname == 'resnet50':
                    self._features.append(ResNet50Feature(**feature))
                elif netname == 'vgg16':
                    self._features.append(VGG16Feature(**feature))
            else:
                raise("unimplemented features")
        self._features = sorted(self._features, key=lambda x:x.min_cell_size)

        # calculate image sample size
        if cnn_feature_idx >= 0:
            self._img_sample_sz = self._features[cnn_feature_idx].init_size(self._img_sample_sz)
        else:
            cell_size = [x.min_cell_size for x in self._features]
            self._img_sample_sz = self._features[0].init_size(self._img_sample_sz, cell_size)

        for idx, feature in enumerate(self._features):
            if idx != cnn_feature_idx:
                feature.init_size(self._img_sample_sz)

        if self.config.use_projection_matrix:
            sample_dim = [ x for feature in self._features for x in feature._compressed_dim ]
        else:
            sample_dim = [ x for feature in self._features for x in feature.num_dim ]

        feature_dim = [ x for feature in self._features for x in feature.num_dim ]

        feature_sz = np.array([x for feature in self._features for x in feature.data_sz ], dtype=np.int32)

        # number of fourier coefficients to save for each filter layer, this will be an odd number
        filter_sz = feature_sz + (feature_sz + 1) % 2

        # the size of the label function DFT. equal to the maximum filter size
        self._k1 = np.argmax(filter_sz, axis=0)[0]
        self._output_sz = filter_sz[self._k1]

        self._num_feature_blocks = len(feature_dim)

        # get the remaining block indices
        self._block_inds = list(range(self._num_feature_blocks))
        self._block_inds.remove(self._k1)

        # how much each feature block has to be padded to the obtain output_sz
        self._pad_sz = [((self._output_sz - filter_sz_) / 2).astype(np.int32) for filter_sz_ in filter_sz]

        # compute the fourier series indices and their transposes
        self._ky = [np.arange(-np.ceil(sz[0]-1)/2, np.floor((sz[0]-1)/2)+1, dtype=np.float32)
                        for sz in filter_sz]
        self._kx = [np.arange(-np.ceil(sz[1]-1)/2, 1, dtype=np.float32)
                        for sz in filter_sz]

        # construct the gaussian label function using poisson formula
        sig_y = np.sqrt(np.prod(np.floor(self._base_target_sz))) * self.config.output_sigma_factor * (self._output_sz / self._img_sample_sz)
        yf_y = [np.sqrt(2 * np.pi) * sig_y[0] / self._output_sz[0] * np.exp(-2 * (np.pi * sig_y[0] * ky_ / self._output_sz[0])**2)
                    for ky_ in self._ky]
        yf_x = [np.sqrt(2 * np.pi) * sig_y[1] / self._output_sz[1] * np.exp(-2 * (np.pi * sig_y[1] * kx_ / self._output_sz[1])**2)
                    for kx_ in self._kx]
        self._yf = [yf_y_.reshape(-1, 1) * yf_x_ for yf_y_, yf_x_ in zip(yf_y, yf_x)]
        if gpu_config.use_gpu:
            self._yf = [cp.asarray(yf) for yf in self._yf]
            self._ky = [cp.asarray(ky) for ky in self._ky]
            self._kx = [cp.asarray(kx) for kx in self._kx]

        # construct cosine window
        self._cos_window = [self._cosine_window(feature_sz_) for feature_sz_ in feature_sz]

        # compute fourier series of interpolation function
        self._interp1_fs = []
        self._interp2_fs = []
        for sz in filter_sz:
            interp1_fs, interp2_fs = self._get_interp_fourier(sz)
            self._interp1_fs.append(interp1_fs)
            self._interp2_fs.append(interp2_fs)

        # get the reg_window_edge parameter
        reg_window_edge = []
        for feature in self._features:
            if hasattr(feature, 'reg_window_edge'):
                reg_window_edge.append(feature.reg_window_edge)
            else:
                reg_window_edge += [self.config.reg_window_edge for _ in range(len(feature.num_dim))]

        # construct spatial regularization filter
        self._reg_filter = [self._get_reg_filter(self._img_sample_sz, self._base_target_sz, reg_window_edge_)
                                for reg_window_edge_ in reg_window_edge]

        # compute the energy of the filter (used for preconditioner)
        if not gpu_config.use_gpu:
            self._reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]
        else:
            self._reg_energy = [cp.real(cp.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]

        if self.config.use_scale_filter:
            self._scale_filter = ScaleFilter(self._target_sz,config=self.config)
            self._num_scales = self._scale_filter.num_scales
            self._scale_step = self._scale_filter.scale_step
            self._scale_factor = self._scale_filter.scale_factors
        else:
            # use the translation filter to estimate the scale
            self._num_scales = self.config.number_of_scales
            self._scale_step = self.config.scale_step
            scale_exp = np.arange(-np.floor((self._num_scales-1)/2), np.ceil((self._num_scales-1)/2)+1)
            self._scale_factor = self._scale_step**scale_exp

        if self._num_scales > 0:
            # force reasonable scale changes
            self._min_scale_factor = self._scale_step ** np.ceil(np.log(np.max(5 / self._img_sample_sz)) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(frame.shape[:2] / self._base_target_sz)) / np.log(self._scale_step))

        # set conjugate gradient options
        init_CG_opts = {'CG_use_FR': True,
                        'tol': 1e-6,
                        'CG_standard_alpha': True
                       }
        self._CG_opts = {'CG_use_FR': self.config.CG_use_FR,
                         'tol': 1e-6,
                         'CG_standard_alpha': self.config.CG_standard_alpha
                        }
        if self.config.CG_forgetting_rate == np.inf or self.config.learning_rate >= 1:
            self._CG_opts['init_forget_factor'] = 0.
        else:
            self._CG_opts['init_forget_factor'] = (1 - self.config.learning_rate) ** self.config.CG_forgetting_rate

        # init ana allocate
        self._gmm = GMM(self._num_samples,config=self.config)
        self._samplesf = [[]] * self._num_feature_blocks

        for i in range(self._num_feature_blocks):
            if not gpu_config.use_gpu:
                self._samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                                              sample_dim[i], self.config.num_samples), dtype=np.complex64)
            else:
                self._samplesf[i] = cp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                                              sample_dim[i], self.config.num_samples), dtype=cp.complex64)

        # allocate
        self._num_training_samples = 0

        # extract sample and init projection matrix
        sample_pos = mround(self._pos)
        sample_scale = self._current_scale_factor
        xl = [x for feature in self._features
                for x in feature.get_features(frame, sample_pos, self._img_sample_sz, self._current_scale_factor) ]  # get features

        if gpu_config.use_gpu:
            xl = [cp.asarray(x) for x in xl]

        xlw = [x * y for x, y in zip(xl, self._cos_window)]                                                          # do windowing
        xlf = [cfft2(x) for x in xlw]                                                                                # fourier series
        xlf = interpolate_dft(xlf, self._interp1_fs, self._interp2_fs)                                               # interpolate features,
        xlf = compact_fourier_coeff(xlf)                                                                             # new sample to be added
        shift_sample_ = 2 * np.pi * (self._pos - sample_pos) / (sample_scale * self._img_sample_sz)
        xlf = shift_sample(xlf, shift_sample_, self._kx, self._ky)
        self._proj_matrix = self._init_proj_matrix(xl, sample_dim, self.config.proj_init_method)
        xlf_proj = self._proj_sample(xlf, self._proj_matrix)
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
            self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)
        self._num_training_samples += 1

        if self.config.update_projection_matrix:
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # train_tracker
        self._sample_energy = [xp.real(x * xp.conj(x)) for x in xlf_proj]

        # init conjugate gradient param
        self._CG_state = None
        if self.config.update_projection_matrix:
            init_CG_opts['maxit'] = np.ceil(self.config.init_CG_iter / self.config.init_GN_iter)
            self._hf = [[[]] * self._num_feature_blocks for _ in range(2)]
            feature_dim_sum = float(np.sum(feature_dim))
            proj_energy = [2 * xp.sum(xp.abs(yf_.flatten())**2) / feature_dim_sum * xp.ones_like(P)
                    for P, yf_ in zip(self._proj_matrix, self._yf)]
        else:
            self._CG_opts['maxit'] = self.config.init_CG_iter
            self._hf = [[[]] * self._num_feature_blocks]

        # init the filter with zeros
        for i in range(self._num_feature_blocks):
            self._hf[0][i] = xp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                int(sample_dim[i]), 1), dtype=xp.complex64)

        if self.config.update_projection_matrix:
            # init Gauss-Newton optimization of the filter and projection matrix
            self._hf, self._proj_matrix = train_joint(
                                                  self._hf,
                                                  self._proj_matrix,
                                                  xlf,
                                                  self._yf,
                                                  self._reg_filter,
                                                  self._sample_energy,
                                                  self._reg_energy,
                                                  proj_energy,
                                                  init_CG_opts,self.config)
            # re-project and insert training sample
            xlf_proj = self._proj_sample(xlf, self._proj_matrix)
            # self._sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, 0:1] = xlf_proj[i]

            # udpate the gram matrix since the sample has changed
            if self.config.distance_matrix_update_type == 'exact':
                # find the norm of the reprojected sample
                new_train_sample_norm = 0.
                for i in range(self._num_feature_blocks):
                    new_train_sample_norm += 2 * xp.real(xp.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                self._gmm._gram_matrix[0, 0] = new_train_sample_norm
        self._hf_full = full_fourier_coeff(self._hf)

        if self.config.use_scale_filter and self._num_scales > 0:
            self._scale_filter.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)
        self._frame_num += 1

    def update(self, frame, train=True, vis=False):
        # target localization step
        xp = cp if gpu_config.use_gpu else np
        pos = self._pos
        old_pos = np.zeros((2))
        for _ in range(self.config.refinement_iterations):
            # if np.any(old_pos != pos):
            if not np.allclose(old_pos, pos):
                old_pos = pos.copy()
                # extract fatures at multiple resolutions
                sample_pos = mround(pos)
                sample_scale = self._current_scale_factor * self._scale_factor
                xt = [x for feature in self._features
                        for x in feature.get_features(frame, sample_pos, self._img_sample_sz, sample_scale) ]  # get features
                if gpu_config.use_gpu:
                    xt = [cp.asarray(x) for x in xt]
                xt_proj = self._proj_sample(xt, self._proj_matrix)                                             # project sample
                xt_proj = [feat_map_ * cos_window_
                        for feat_map_, cos_window_ in zip(xt_proj, self._cos_window)]                          # do windowing
                xtf_proj = [cfft2(x) for x in xt_proj]                                                         # compute the fourier series
                xtf_proj = interpolate_dft(xtf_proj, self._interp1_fs, self._interp2_fs)                       # interpolate features to continuous domain

                # compute convolution for each feature block in the fourier domain, then sum over blocks
                scores_fs_feat = [[]] * self._num_feature_blocks
                scores_fs_feat[self._k1] = xp.sum(self._hf_full[self._k1] * xtf_proj[self._k1], 2)
                scores_fs = scores_fs_feat[self._k1]

                # scores_fs_sum shape: height x width x num_scale
                for i in self._block_inds:
                    scores_fs_feat[i] = xp.sum(self._hf_full[i] * xtf_proj[i], 2)
                    scores_fs[self._pad_sz[i][0]:self._output_sz[0]-self._pad_sz[i][0],
                              self._pad_sz[i][1]:self._output_sz[0]-self._pad_sz[i][1]] += scores_fs_feat[i]

                # optimize the continuous score function with newton's method.
                trans_row, trans_col, scale_idx = optimize_score(scores_fs, self.config.newton_iterations)

                # show score
                if vis:
                    if gpu_config.use_gpu:
                       xp = cp
                    self.score = xp.fft.fftshift(sample_fs(scores_fs[:,:,scale_idx],
                            tuple((10*self._output_sz).astype(np.uint32))))
                    if gpu_config.use_gpu:
                       self.score = cp.asnumpy(self.score)
                    self.crop_size = self._img_sample_sz * self._current_scale_factor

                # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                translation_vec = np.array([trans_row, trans_col]) * (self._img_sample_sz / self._output_sz) * \
                                    self._current_scale_factor * self._scale_factor[scale_idx]
                scale_change_factor = self._scale_factor[scale_idx]

                # udpate position
                pos = sample_pos + translation_vec

                if self.config.clamp_position:
                    pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), pos))

                # do scale tracking with scale filter
                if self._num_scales > 0 and self.config.use_scale_filter:
                    scale_change_factor = self._scale_filter.track(frame, pos, self._base_target_sz,
                           self._current_scale_factor)

                # udpate the scale
                self._current_scale_factor *= scale_change_factor

                # adjust to make sure we are not to large or to small
                if self._current_scale_factor < self._min_scale_factor:
                    self._current_scale_factor = self._min_scale_factor
                elif self._current_scale_factor > self._max_scale_factor:
                    self._current_scale_factor = self._max_scale_factor

        # model udpate step
        if self.config.learning_rate > 0:
            # use the sample that was used for detection
            sample_scale = sample_scale[scale_idx]
            xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_proj]

            # shift the sample so that the target is centered
            shift_sample_ = 2 * np.pi * (pos - sample_pos) / (sample_scale * self._img_sample_sz)
            xlf_proj = shift_sample(xlf_proj, shift_sample_, self._kx, self._ky)

        # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior weight are also updated
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
                self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)

        if self._num_training_samples < self._num_samples:
            self._num_training_samples += 1

        if self.config.learning_rate > 0:
            for i in range(self._num_feature_blocks):
                if merged_sample_id >= 0:
                    self._samplesf[i][:, :, :, merged_sample_id:merged_sample_id+1] = merged_sample[i]
                if new_sample_id >= 0:
                    self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # training filter
        if self._frame_num < self.config.skip_after_frame or \
                self._frames_since_last_train >= self.config.train_gap:
            # print("Train filter: ", self._frame_num)
            new_sample_energy = [xp.real(xlf * xp.conj(xlf)) for xlf in xlf_proj]
            self._CG_opts['maxit'] = self.config.CG_iter
            self._sample_energy = [(1 - self.config.learning_rate) * se + self.config.learning_rate * nse
                                   for se, nse in zip(self._sample_energy, new_sample_energy)]

            # do conjugate gradient optimization of the filter
            self._hf, self._CG_state = train_filter(
                                                 self._hf,
                                                 self._samplesf,
                                                 self._yf,
                                                 self._reg_filter,
                                                 self._gmm.prior_weights,
                                                 self._sample_energy,
                                                 self._reg_energy,
                                                 self._CG_opts,
                                                 self._CG_state,
                                                    self.config)
            # reconstruct the ful fourier series
            self._hf_full = full_fourier_coeff(self._hf)
            self._frames_since_last_train = 0
        else:
            self._frames_since_last_train += 1
        if self.config.use_scale_filter:
            self._scale_filter.update(frame, pos, self._base_target_sz, self._current_scale_factor)

        # udpate the target size
        self._target_sz = self._base_target_sz * self._current_scale_factor

        # save position and calculate fps
        bbox = (pos[1] - self._target_sz[1]/2, # xmin
                pos[0] - self._target_sz[0]/2, # ymin
                pos[1] + self._target_sz[1]/2, # xmax
                pos[0] + self._target_sz[0]/2) # ymax
        self._pos = pos
        self._frame_num += 1
        return bbox
