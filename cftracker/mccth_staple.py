"""Python re-implementation of "Multi-Cue Correlation Filters for Robust Visual Tracking"
@inproceedings{wang2018multi,
  title={Multi-cue correlation filters for robust visual tracking},
  author={Wang, Ning and Zhou, Wengang and Tian, Qi and Hong, Richang and Wang, Meng and Li, Houqiang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4844--4853},
  year={2018}
}

I implemented its hand-crafted feature version
"""

import numpy as np
import cv2
from .base import BaseCF
from .feature import extract_hog_feature, extract_cn_feature
from lib.utils import cos_window
from lib.fft_tools import fft2, ifft2
from .scale_estimator import LPScaleEstimator


def mod_one(a, b):
    y = np.mod(a - 1, b) + 1
    return y


def cal_ious(bboxes1, bboxes2):
    intersect_tl_x = np.max((bboxes1[:, 0], bboxes2[:, 0]), axis=0)
    intersect_tl_y = np.max((bboxes1[:, 1], bboxes2[:, 1]), axis=0)
    intersect_br_x = np.min((bboxes1[:, 0] + bboxes1[:, 2], bboxes2[:, 0] + bboxes2[:, 2]), axis=0)
    intersect_br_y = np.min((bboxes1[:, 1] + bboxes1[:, 3], bboxes2[:, 1] + bboxes2[:, 3]), axis=0)
    intersect_w = intersect_br_x - intersect_tl_x
    intersect_w[intersect_w < 0] = 0
    intersect_h = intersect_br_y - intersect_tl_y
    intersect_h[intersect_h < 0] = 0
    intersect_areas = intersect_h * intersect_w
    ious = intersect_areas / (bboxes1[:, 2] * bboxes1[:, 3] + bboxes2[:, 2] * bboxes2[:, 3] - intersect_areas)
    return ious


# max val at the bottom right loc
def gaussian2d_rolled_labels_staple(sz, sigma):
    halfx, halfy = int(np.floor((sz[0] - 1) / 2)), int(np.floor((sz[1] - 1) / 2))
    x_range = np.arange(-halfx, halfx + 1)
    y_range = np.arange(-halfy, halfy + 1)
    i, j = np.meshgrid(y_range, x_range)
    i_mod_range = mod_one(i, sz[1])
    j_mod_range = mod_one(j, sz[0])
    labels = np.zeros((sz[1], sz[0]))
    labels[i_mod_range - 1, j_mod_range - 1] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return labels


def crop_filter_response(response_cf, response_sz):
    h, w = response_cf.shape[:2]
    half_width = int(np.floor(response_sz[0] / 2))
    half_height = int(np.floor(response_sz[1] / 2))
    range_i, range_j = np.arange(-half_height, half_height + 1), np.arange(-half_width, half_width + 1)
    i, j = np.meshgrid(mod_one(range_i, h), mod_one(range_j, w))
    new_responses = response_cf[i - 1, j - 1]
    return new_responses.T


def get_center_likelihood(likelihood_map, m):
    h, w = likelihood_map.shape[:2]
    n1 = h - m[1] + 1
    n2 = w - m[0] + 1
    sat = cv2.integral(likelihood_map)
    i, j = np.arange(n1), np.arange(n2)
    i, j = np.meshgrid(i, j)
    sat1 = sat[i, j]
    sat2 = np.roll(sat, -m[1], axis=0)
    sat2 = np.roll(sat2, -m[0], axis=1)
    sat2 = sat2[i, j]
    sat3 = np.roll(sat, -m[1], axis=0)
    sat3 = sat3[i, j]
    sat4 = np.roll(sat, -m[0], axis=1)
    sat4 = sat4[i, j]
    center_likelihood = (sat1 + sat2 - sat3 - sat4) / (m[0] * m[1])
    return center_likelihood.T


class Expert:
    def  __init__(self):
        self.xt = None
        self.hf_den = None
        self.hf_num = None
        self.response = None
        self.pos = None
        self.rect_positions = []
        self.centers = []
        self.smoothes = []
        self.smooth_score = None
        self.smooth_scores = []
        self.rob_scores = []


class MCCTHStaple(BaseCF):
    def __init__(self, config):
        super(MCCTHStaple).__init__()
        self.cell_size = config.hog_cell_size
        self.fixed_area = config.fixed_area
        self.n_bins = config.n_bins
        self.learning_rate_pwp = config.interp_factor_pwp
        self.inner_padding = config.inner_padding
        self.output_sigma_factor = config.output_sigma_factor
        self.lambda_ = config.lambda_
        self.learning_rate_cf = config.interp_factor_cf
        self.merge_factor = config.merge_factor
        self.den_per_channel = config.den_per_channel
        self.config = config

        self.scale_adaptation = config.scale_adaptation
        self.padding=config.padding

        self.period = config.period
        self.update_thresh = config.update_thresh
        self.expert_num = config.expert_num

        self.scale_config = config.scale_config

        weight_num = np.arange(self.period)
        self.weight = 1.1 ** weight_num
        self.mean_score = [0]
        self.psr_score = [0]
        self.id_ensemble = []
        self.frame_idx = -1
        self.experts = []
        for i in range(7):
            self.experts.append(Expert())
            self.id_ensemble.append(1)

    def init(self, first_frame, bbox):

        self.frame_idx += 1
        first_frame = first_frame.astype(np.float32)
        bbox = np.array(bbox).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.crop_size = (int(w * (1 + self.padding)), int(h * (1 + self.padding)))
        self.target_sz = (self.w, self.h)

        self.bin_mapping = self.get_bin_mapping(self.n_bins)
        avg_dim = (w + h) / 2
        self.bg_area = (round(w + avg_dim), round(h + avg_dim))
        self.fg_area = (int(round(w - avg_dim * self.inner_padding)), int(round(h - avg_dim * self.inner_padding)))
        self.bg_area = (
            int(min(self.bg_area[0], first_frame.shape[1] - 1)), int(min(self.bg_area[1], first_frame.shape[0] - 1)))

        self.bg_area = (self.bg_area[0] - (self.bg_area[0] - self.target_sz[0]) % 2,
                        self.bg_area[1] - (self.bg_area[1] - self.target_sz[1]) % 2)
        self.fg_area = (self.fg_area[0] + (self.bg_area[0] - self.fg_area[0]) % 2,
                        self.fg_area[1] + (self.bg_area[1] - self.fg_area[1]) % 2)
        self.area_resize_factor = np.sqrt(self.fixed_area / (self.bg_area[0] * self.bg_area[1]))
        self.norm_bg_area = (
            round(self.bg_area[0] * self.area_resize_factor), round(self.bg_area[1] * self.area_resize_factor))

        self.cf_response_size = (int(np.floor(self.norm_bg_area[0] / self.cell_size)),
                                 int(np.floor(self.norm_bg_area[1] / self.cell_size)))
        norm_target_sz_w = 0.75 * self.norm_bg_area[0] - 0.25 * self.norm_bg_area[1]
        norm_target_sz_h = 0.75 * self.norm_bg_area[1] - 0.25 * self.norm_bg_area[0]
        self.norm_target_sz = (round(norm_target_sz_w), round(norm_target_sz_h))
        norm_pad = (int(np.floor((self.norm_bg_area[0] - norm_target_sz_w) / 2)),
                    int(np.floor((self.norm_bg_area[1] - norm_target_sz_h) / 2)))
        radius = min(norm_pad[0], norm_pad[1])
        self.norm_delta_area = (2 * radius + 1, 2 * radius + 1)
        self.norm_pwp_search_area = (self.norm_target_sz[0] + self.norm_delta_area[0] - 1,
                                     self.norm_target_sz[1] + self.norm_delta_area[1] - 1)

        patch_padded = self.get_sub_window(first_frame, self._center, self.norm_bg_area, self.bg_area)
        self.new_pwp_model = True
        self.bg_hist, self.fg_hist = self.update_hist_model(self.new_pwp_model, patch_padded, self.bg_area,
                                                            self.fg_area,
                                                            self.target_sz, self.norm_bg_area, self.n_bins,
                                                            )
        self.new_pwp_model = False
        self._window = cos_window(self.cf_response_size)
        output_sigma = np.sqrt(
            self.norm_target_sz[0] * self.norm_target_sz[1]) * self.output_sigma_factor / self.cell_size
        self.y = gaussian2d_rolled_labels_staple(self.cf_response_size, output_sigma)
        self._init_response_center = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape)
        # print(self._init_response_center)
        self.yf = fft2(self.y)

        if self.scale_adaptation is True:
            self.scale_factor = 1
            self.base_target_sz = self.target_sz
            self.scale_estimator = LPScaleEstimator(self.target_sz, config=self.scale_config)
            self.scale_estimator.init(first_frame, self._center, self.base_target_sz, self.scale_factor)


        im_patch_bg = self.get_sub_window(first_frame, self._center, self.norm_bg_area, self.bg_area)
        xt = self.get_feature_map(im_patch_bg, self.cell_size)
        xt = self._window[:, :, None] * xt
        xt_cn, xt_hog1, xt_hog2 = self.split_features(xt)
        self.experts[0].xt = xt_cn
        self.experts[1].xt = xt_hog1
        self.experts[2].xt = xt_hog2
        self.experts[3].xt = np.concatenate((xt_hog1, xt_cn), axis=2)
        self.experts[4].xt = np.concatenate((xt_hog2, xt_cn), axis=2)
        self.experts[5].xt = np.concatenate((xt_hog1, xt_hog2), axis=2)
        self.experts[6].xt = xt

        for i in range(self.expert_num):
            xtf = fft2(self.experts[i].xt)
            self.experts[i].hf_den = np.conj(xtf) * xtf / (self.cf_response_size[0] * self.cf_response_size[1])
            self.experts[i].hf_num = np.conj(self.yf)[:, :, None] * xtf / (
                        self.cf_response_size[0] * self.cf_response_size[1])


        self.rect_position_padded = None
        self.avg_dim = avg_dim
        for i in range(self.expert_num):
            self.experts[i].rect_positions.append([self._center[0] - self.target_sz[0] / 2,
                                                   self._center[1] - self.target_sz[1] / 2,
                                                   self.target_sz[0], self.target_sz[1]])
            self.experts[i].rob_scores.append(1)
            self.experts[i].smoothes.append(0)
            self.experts[i].smooth_scores.append(1)
            self.experts[i].centers.append([self._center[0], self._center[1]])

    def update(self, current_frame, vis=False):
        self.frame_idx += 1
        im_patch_cf = self.get_sub_window(current_frame, self._center, self.norm_bg_area, self.bg_area)
        pwp_search_area = (round(self.norm_pwp_search_area[0] / self.area_resize_factor),
                           round(self.norm_pwp_search_area[1] / self.area_resize_factor))
        im_patch_pwp = self.get_sub_window(current_frame, self._center, self.norm_pwp_search_area, pwp_search_area)
        likelihood_map = self.get_colour_map(im_patch_pwp, self.bg_hist, self.fg_hist, self.bin_mapping)
        likelihood_map[np.isnan(likelihood_map)] = 0.
        self.norm_target_sz = (int(self.norm_target_sz[0]), int(self.norm_target_sz[1]))
        response_pwp = get_center_likelihood(likelihood_map, self.norm_target_sz)

        xt = self.get_feature_map(im_patch_cf, self.cell_size)
        xt = self._window[:, :, None] * xt
        xt_cn, xt_hog1, xt_hog2 = self.split_features(xt)
        self.experts[0].xt = xt_cn
        self.experts[1].xt = xt_hog1
        self.experts[2].xt = xt_hog2
        self.experts[3].xt = np.concatenate((xt_hog1, xt_cn), axis=2)
        self.experts[4].xt = np.concatenate((xt_hog2, xt_cn), axis=2)
        self.experts[5].xt = np.concatenate((xt_hog1, xt_hog2), axis=2)
        self.experts[6].xt = xt

        center = ((self.norm_delta_area[0] - 1) / 2, (self.norm_delta_area[1] - 1) / 2)

        for i in range(self.expert_num):
            xtf = fft2(self.experts[i].xt)
            hf = self.experts[i].hf_num / (np.sum(self.experts[i].hf_den, axis=2) + self.lambda_)[:, :, None]
            response_cf = np.real(ifft2(np.sum(np.conj(hf) * xtf, axis=2)))
            response_sz = (self.floor_odd(self.norm_delta_area[0] / self.cell_size),
                           self.floor_odd(self.norm_delta_area[1] / self.cell_size))
            response_cf = cv2.resize(crop_filter_response(response_cf, response_sz), self.norm_delta_area,
                                     cv2.INTER_NEAREST)
            response_cf[np.isnan(response_cf)] = 0.
            self.experts[i].response = (1 - self.merge_factor) * response_cf + self.merge_factor * response_pwp

            row, col = np.unravel_index(np.argmax(self.experts[i].response, axis=None), self.experts[i].response.shape)
            dy = (row - center[1]) / self.area_resize_factor
            dx = (col - center[0]) / self.area_resize_factor

            self.experts[i].pos = (self._center[0] + dx, self._center[1] + dy)
            cx, cy, w, h = self.experts[i].pos[0], self.experts[i].pos[1], self.target_sz[0], self.target_sz[1]
            self.experts[i].rect_positions.append([cx - w / 2, cy - h / 2, w, h])
            self.experts[i].centers.append([cx, cy])

            pre_center = self.experts[i].centers[self.frame_idx - 1]
            smooth = np.sqrt((cx - pre_center[0]) ** 2 + (cy - pre_center[1]) ** 2)
            self.experts[i].smoothes.append(smooth)
            self.experts[i].smooth_scores.append(np.exp(-smooth ** 2 / (2 * self.avg_dim ** 2)))

        if self.frame_idx >= self.period - 1:
            for i in range(self.expert_num):
                self.experts[i].rob_scores.append(self.robustness_eva(self.experts, i, self.frame_idx,
                                                                      self.period, self.weight, self.expert_num))

                self.id_ensemble[i] = self.experts[i].rob_scores[self.frame_idx]
            self.mean_score.append(np.sum(np.array(self.id_ensemble)) / self.expert_num)
            idx = np.argmax(np.array(self.id_ensemble))
            self._center = self.experts[idx].pos
            self.response = self.experts[idx].response
        else:
            for i in range(self.expert_num):
                self.experts[i].rob_scores.append(1)
            self._center = self.experts[6].pos
            self.response = self.experts[6].response
            self.mean_score.append(0)

        if vis is True:
            self.score = self.response

        # adaptive update
        score1 = self.cal_psr(self.experts[0].response)
        score2 = self.cal_psr(self.experts[1].response)
        score3 = self.cal_psr(self.experts[2].response)
        self.psr_score.append((score1 + score2 + score3) / 3)

        if self.frame_idx >= self.period - 1:
            final_score = self.mean_score[self.frame_idx] * self.psr_score[self.frame_idx]
            ave_score = np.sum(np.array(self.mean_score)[self.period-1:self.frame_idx + 1] *
                               np.array(self.psr_score[self.period-1:self.frame_idx + 1])) / (
                                    self.frame_idx + 1 - self.period+1)
            threshold = self.update_thresh * ave_score
            if final_score > threshold:
                self.learning_rate_pwp = self.config.interp_factor_pwp
                self.learning_rate_cf = self.config.interp_factor_cf
            else:
                self.learning_rate_pwp = 0
                self.learning_rate_cf = (final_score / threshold) ** 3 * self.config.interp_factor_cf

        if self.scale_adaptation:
            self.scale_factor = self.scale_estimator.update(current_frame, self._center, self.base_target_sz,
                                                                    self.scale_factor)
            self.target_sz = (round(self.base_target_sz[0] * self.scale_factor), round(self.base_target_sz[1] * self.scale_factor))
            avg_dim = (self.target_sz[0] + self.target_sz[1]) / 2
            bg_area = (round(self.target_sz[0] + avg_dim), round(self.target_sz[1] + avg_dim))
            fg_area = (round(self.target_sz[0] - avg_dim * self.inner_padding),
                       round(self.target_sz[1] - avg_dim * self.inner_padding))
            bg_area = (min(bg_area[0], current_frame.shape[1] - 1), min(bg_area[1], current_frame.shape[0] - 1))
            self.bg_area = (
                bg_area[0] - (bg_area[0] - self.target_sz[0]) % 2, bg_area[1] - (bg_area[1] - self.target_sz[1]) % 2)
            self.fg_area = (
                fg_area[0] + (self.bg_area[0] - fg_area[0]) % 2, fg_area[1] + (self.bg_area[1] - fg_area[1]) % 2)
            self.area_resize_factor = np.sqrt(self.fixed_area / (self.bg_area[0] * self.bg_area[1]))

        im_patch_bg = self.get_sub_window(current_frame, self._center, self.norm_bg_area, self.bg_area)
        xt = self.get_feature_map(im_patch_bg, self.cell_size)
        xt = self._window[:, :, None] * xt
        xt_cn, xt_hog1, xt_hog2 = self.split_features(xt)
        self.experts[0].xt = xt_cn
        self.experts[1].xt = xt_hog1
        self.experts[2].xt = xt_hog2
        self.experts[3].xt = np.concatenate((xt_hog1, xt_cn), axis=2)
        self.experts[4].xt = np.concatenate((xt_hog2, xt_cn), axis=2)
        self.experts[5].xt = np.concatenate((xt_hog1, xt_hog2), axis=2)
        self.experts[6].xt = xt

        for i in range(self.expert_num):
            xtf = fft2(self.experts[i].xt)
            hf_den = np.conj(xtf) * xtf / (self.cf_response_size[0] * self.cf_response_size[1])
            hf_num = np.conj(self.yf)[:, :, None] * xtf / (self.cf_response_size[0] * self.cf_response_size[1])
            self.experts[i].hf_den = (1 - self.learning_rate_cf) * self.experts[i].hf_den + self.learning_rate_cf * hf_den
            self.experts[i].hf_num = (1 - self.learning_rate_cf) * self.experts[i].hf_num + self.learning_rate_cf * hf_num
        if self.learning_rate_pwp != 0:
            im_patch_bg = self.get_sub_window(current_frame, self._center, self.norm_bg_area, self.bg_area)
            self.bg_hist, self.fg_hist = self.update_hist_model(self.new_pwp_model,
                                                                im_patch_bg, self.bg_area, self.fg_area, self.target_sz,
                                                                self.norm_bg_area, self.n_bins)

        return [self._center[0] - self.target_sz[0] / 2, self._center[1] - self.target_sz[1] / 2,
                self.target_sz[0], self.target_sz[1]]

    def floor_odd(self, x):
        return 2 * int(np.floor((x - 1) / 2)) + 1


    def get_feature_map(self, im_patch, cell_size):
        hog_feature = extract_hog_feature(im_patch, cell_size=cell_size)
        cn_feature = extract_cn_feature(im_patch, cell_size)
        if cell_size > 1:
            im_patch = self.mex_resize(im_patch, (self._window.shape[1], self._window.shape[0])).astype(np.uint8)
        gray = cv2.cvtColor(im_patch, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] / 255 - 0.5
        features = np.concatenate((cn_feature, gray, hog_feature), axis=2)
        return features

    def split_features(self, features):
        cn_feature, hog_feature1, hog_feature2 = features[:, :, :11], features[:, :, 11:27], features[:, :, 27:]
        return cn_feature, hog_feature1, hog_feature2

    def update_hist_model(self, new_model, patch, bg_area, fg_area, target_sz, norm_area,
                          n_bins):
        pad_offset1 = ((bg_area[0] - target_sz[0]) / 2, (bg_area[1] - target_sz[1]) / 2)
        assert pad_offset1[0] == round(pad_offset1[0]) and pad_offset1[1] == round(pad_offset1[1])
        bg_mask = np.ones((int(bg_area[1]), int(bg_area[0])))
        pad_offset1 = (int(max(1, pad_offset1[0])), int(max(1, pad_offset1[1])))
        bg_mask[pad_offset1[1]:-pad_offset1[1], pad_offset1[0]:-pad_offset1[0]] = 0.

        pad_offset2 = ((bg_area[0] - fg_area[0]) / 2, (bg_area[1] - fg_area[1]) / 2)
        assert pad_offset2[0] == round(pad_offset2[0]) and pad_offset2[1] == round(pad_offset2[1])
        fg_mask = np.zeros((int(bg_area[1]), int(bg_area[0])))
        pad_offset2 = (int(max(1, pad_offset2[0])), int(max(1, pad_offset2[1])))
        fg_mask[pad_offset2[1]:-pad_offset2[1], pad_offset2[0]:-pad_offset2[0]] = 1.
        fg_mask = self.mex_resize(fg_mask, norm_area)
        bg_mask = self.mex_resize(bg_mask, norm_area)
        bg_hist_new = self.compute_histogram(patch, bg_mask, n_bins)
        fg_hist_new = self.compute_histogram(patch, fg_mask, n_bins)

        if new_model is not True:
            bg_hist_new = (1 - self.learning_rate_pwp) * self.bg_hist + self.learning_rate_pwp * bg_hist_new
            fg_hist_new = (1 - self.learning_rate_pwp) * self.fg_hist + self.learning_rate_pwp * fg_hist_new
        return bg_hist_new, fg_hist_new

    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = self.mex_resize(im_patch, model_sz).astype(np.uint8)
        return im_patch

    def mex_resize(self, img, sz, method='auto'):
        sz = (int(sz[0]), int(sz[1]))
        src_sz = (img.shape[1], img.shape[0])
        if method == 'antialias':
            interpolation = cv2.INTER_AREA
        elif method == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:
            if sz[1] > src_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
        img = cv2.resize(img, sz, interpolation=interpolation)
        return img

    def compute_histogram(self, patch, mask, n_bins):
        h, w, d = patch.shape
        assert h == mask.shape[0] and w == mask.shape[1]
        mask = mask.astype(np.uint8)
        histogram = cv2.calcHist([patch], [0, 1, 2], mask, [n_bins, n_bins, n_bins],
                                 [0, 256, 0, 256, 0, 256]) / np.count_nonzero(mask)
        return histogram

    def get_colour_map(self, patch, bg_hist, fg_hist, bin_mapping):
        frame_bin = cv2.LUT(patch, bin_mapping).astype(np.int64)
        P_fg = fg_hist[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]
        P_bg = bg_hist[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]
        P_O = P_fg / (P_fg + P_bg)
        return P_O

    def get_bin_mapping(self, num_bins):
        bin_mapping = np.zeros((256,))
        for i in range(bin_mapping.shape[0]):
            bin_mapping[i] = (np.floor(i / (256 / num_bins)))
        return bin_mapping.astype(np.uint8)

    def cal_psr(self, response):
        cf_max = np.max(response)
        cf_avg = np.mean(response)
        cf_sigma = np.std(response)
        return (cf_max - cf_avg) / cf_sigma

    def robustness_eva(self, experts, num, frame_idx, period, weight, expert_num):
        overlap_score = np.zeros((period, expert_num))
        for i in range(expert_num):
            bboxes1 = np.array(experts[i].rect_positions)[frame_idx - period + 1:frame_idx + 1]
            bboxes2 = np.array(experts[num].rect_positions)[frame_idx - period + 1:frame_idx + 1]
            overlaps = cal_ious(bboxes1, bboxes2)
            overlap_score[:, i] = np.exp(-(1 - overlaps) ** 2 / 2)
        avg_overlap = np.sum(overlap_score, axis=1) / expert_num
        expert_avg_overlap = np.sum(overlap_score, axis=0) / period
        var_overlap = np.sqrt(np.sum((overlap_score - expert_avg_overlap[np.newaxis, :]) ** 2, axis=1) / expert_num)
        norm_factor = 1 / np.sum(np.array(weight))
        weight_avg_overlap = norm_factor * (weight.dot(avg_overlap))
        weight_var_overlap = norm_factor * (weight.dot(var_overlap))
        pair_score = weight_avg_overlap / (weight_var_overlap + 0.008)
        smooth_score = experts[num].smooth_scores[frame_idx - period + 1:frame_idx + 1]
        self_score = norm_factor * np.sum(np.array(smooth_score) * weight)
        eta = 0.1
        reliability = eta * pair_score + (1 - eta) * self_score
        return reliability
