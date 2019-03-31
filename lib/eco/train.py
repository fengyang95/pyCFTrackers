import numpy as np
import warnings

from scipy.signal import convolve
from .fourier_tools import symmetrize_filter
from .config import gpu_config
from .cuda_tools import convolve2d

if gpu_config.use_gpu:
    import cupy as cp


def diag_precond(hf, M_diag):
    ret = []
    for x, y in zip(hf, M_diag):
        ret.append([x_ / y_ for x_, y_ in zip(x, y)])
    return ret

def inner_product_filter(xf, yf):
    """
        computes the inner product between two filters
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(xf[0][0])
    else:
        xp = np
    ip = 0
    for i in range(len(xf[0])):
        ip += 2 * xp.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - xp.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten())
    return xp.real(ip)

def inner_product_joint(xf, yf):
    """
        computes the joint inner product between two filters and projection matrices
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(xf[0][0])
    else:
        xp = np
    ip = 0
    for i in range(len(xf[0])):
        ip += 2 * xp.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - xp.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten())
        ip += xp.vdot(xf[1][i].flatten(), yf[1][i].flatten())
    return xp.real(ip)

def lhs_operation(hf, samplesf, reg_filter, sample_weights):
    """
        This is the left-hand-side operation in Conjugate Gradient
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(hf[0][0])
    else:
        xp = np
    num_features = len(hf[0])
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(num_features):
        filter_sz[i, :] = np.array(hf[0][i].shape[:2])

    # index for the feature block with the largest spatial size
    k1 = np.argmax(filter_sz[:, 0])

    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[0][k1].shape[0], hf[0][k1].shape[1]*2-1])

    # compute the operation corresponding to the data term in the optimization
    # implements: A.H diag(sample_weights) A f

    # sum over all features and feature blocks
    sh = xp.matmul(hf[0][k1].transpose(0, 1, 3, 2), samplesf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[0][i].shape[0], hf[0][i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += xp.matmul(hf[0][i].transpose(0, 1, 3, 2), samplesf[i])

    # weight all the samples
    sh = sample_weights.reshape(1, 1, 1, -1) * sh

    # multiply with the transpose
    hf_out = [[]] * num_features
    hf_out[k1] = xp.matmul(xp.conj(samplesf[k1]), sh.transpose(0, 1, 3, 2))
    for i in block_inds:
        hf_out[i] = xp.matmul(xp.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :].transpose(0, 1, 3, 2))

    # compute the operation corresponding to the regularization term (convolve each feature dimension
    # with the DFT of w, and the transposed operation) add the regularization part
    # W^H W f
    for i in range(num_features):
        reg_pad = min(reg_filter[i].shape[1] - 1, hf[0][i].shape[1]-1)

        # add part needed for convolution
        hf_conv = xp.concatenate([hf[0][i], xp.conj(xp.rot90(hf[0][i][:, -reg_pad-1:-1, :], 2))], axis=1)

        if not gpu_config.use_gpu:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                # do first convolution
                hf_conv = convolve(hf_conv, reg_filter[i][:,:,np.newaxis,np.newaxis])

                # do final convolution and put together result
                if reg_pad!=0:
                    hf_out[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:,:,np.newaxis,np.newaxis], 'valid')
                else:
                    hf_out[i] += convolve(hf_conv[:, :, :], reg_filter[i][:, :, np.newaxis, np.newaxis],
                                          'valid')

        else:
            # do first convolution
            hf_conv = convolve2d(hf_conv, reg_filter[i][:,:,cp.newaxis,cp.newaxis])

            # do final convolution and put together result
            if reg_pad!=0:
                hf_out[i] += convolve2d(hf_conv[:, :-reg_pad, :], reg_filter[i][:,:,cp.newaxis,cp.newaxis], 'valid')
            else:
                hf_out[i] += convolve2d(hf_conv[:, :, :], reg_filter[i][:, :, cp.newaxis, cp.newaxis], 'valid')
    return [hf_out]


def lhs_operation_joint(hf, samplesf, reg_filter, init_samplef, XH, init_hf, proj_reg):
    """
        This is the left-hand-side operation in Conjugate Gradient(coded tested no error)
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(hf[0][0])
    else:
        xp = np
    hf_out = [[[]] * len(hf[0]) for _ in range(len(hf))]

    # extract projection matrix and filter separately
    P = [xp.real(hf_) for hf_ in hf[1]]
    hf = hf[0]

    # get sizes
    num_features = len(hf)
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(num_features):
        filter_sz[i, :] = np.array(hf[i].shape[:2])

    # index for the feature block with the largest spatial size
    k1 = np.argmax(filter_sz[:, 0])

    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[k1].shape[0], hf[k1].shape[1]*2-1])

    # compute the operation corresponding to the data term in the optimization(blockwise matrix multiplications)
    # implements: A^H A f

    # sum over all features and feature blocks
    sh = xp.matmul(samplesf[k1].transpose(0, 1, 3, 2), hf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[i].shape[0], hf[i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += xp.matmul(samplesf[i].transpose(0, 1, 3, 2), hf[i])

    # multiply with the transpose
    hf_out1 = [[]] * num_features
    hf_out1[k1] = xp.matmul(xp.conj(samplesf[k1]), sh)
    for i in block_inds:
        hf_out1[i] = xp.matmul(xp.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :])

    # compute the operation corresponding to the regularization term
    # (convolve each feature dimension with the DFT of w, and the transposed
    # operation) add the regularization part
    for i in range(num_features):
        reg_pad = min(reg_filter[i].shape[1]-1, hf[i].shape[1]-1)
        # not sure
        # modified by fengyang 2019/3/30
        if reg_pad==0:
            reg_pad=1

        # add part needed for convolution
        hf_conv = xp.concatenate([hf[i], xp.conj(xp.rot90(hf[i][:, -reg_pad-1:-1, :], 2))], axis=1)

        if not gpu_config.use_gpu:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                # do first convolution
                hf_conv = convolve(hf_conv, reg_filter[i][:, :, np.newaxis, np.newaxis])
                # do final convolution and put together result
                # fix reg_pad==0
                if reg_pad!=0:
                    hf_out1[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:, :, np.newaxis, np.newaxis], 'valid')
                else:
                    hf_out1[i] += convolve(hf_conv[:, :, :], reg_filter[i][:, :, np.newaxis, np.newaxis],
                                           'valid')

        else:
            # do first convolution
            hf_conv = convolve2d(hf_conv, reg_filter[i][:,:,cp.newaxis,cp.newaxis])

            # do final convolution and put together result
            hf_out1[i] += convolve2d(hf_conv[:, :-reg_pad, :,:], reg_filter[i][:,:,cp.newaxis,cp.newaxis], 'valid')

    # stuff related to the projection matrix
    # B * P
    BP_list = [xp.matmul(init_hf_.transpose(0, 1, 3, 2), xp.matmul(P_.T, init_samplef_))
            for init_samplef_, P_, init_hf_ in zip(init_samplef, P, init_hf)]
    BP = BP_list[k1]
    for i in block_inds:
        BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += BP_list[i]

    # multiply with the transpose: A^H * BP
    hf_out[0][k1] = hf_out1[k1] + (xp.conj(samplesf[k1]) * BP)

    # B^H * BP
    fBP = [[]] * num_features
    fBP[k1] = (xp.conj(init_hf[k1]) * BP).reshape((-1, init_hf[k1].shape[2]))

    # compute proj matrix part: B^H * A_m * f
    shBP = [[]] * num_features
    shBP[k1] = (xp.conj(init_hf[k1]) * sh).reshape((-1, init_hf[k1].shape[2]))

    for i in block_inds:
        # multiply with the transpose: A^H * BP
        hf_out[0][i] = hf_out1[i] + (BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] * xp.conj(samplesf[i]))

        # B^H * BP
        fBP[i] = (xp.conj(init_hf[i]) * BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))

        # compute proj matrix part: B^H * A_m * f
        shBP[i] = (xp.conj(init_hf[i]) * sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))

    for i in range(num_features):
        fi = hf[i].shape[0] * (hf[i].shape[1] - 1) # index where the last frequency column starts

        # B^H * BP + \lambda \delta P
        hf_out2 = 2 * xp.real(XH[i].dot(fBP[i]) - XH[i][:, fi:].dot(fBP[i][fi:, :])) + proj_reg * P[i]

        # compute proj matrix part: B^H * A_m * f
        hf_out[1][i] = hf_out2 + 2 * xp.real(XH[i].dot(shBP[i]) - XH[i][:, fi:].dot(shBP[i][fi:, :]))
    return hf_out


def preconditioned_conjugate_gradient(A, b, opts, M1, M2, ip,x0, state=None):
    """
        performs preconditioned conjugate gradient
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(x0[0][0])
    else:
        xp = np
    maxit  = int(opts['maxit'])

    if 'init_forget_factor' not in opts:
        opts['init_forget_factor'] = 1

    x = x0
    # Load the CG state
    p = None
    rho = 1
    r_prev = None

    # load the CG state
    if state is None:
        state = {}
    else:
        if opts['init_forget_factor'] > 0:
            if 'p' in state:
                p = state['p']
            if 'rho' in state:
                rho = state['rho'] / opts['init_forget_factor']
            if 'r_prev' in state and not opts['CG_use_FR']:
                r_prev = state['r_prev']
    state['flag'] = 1

    r = []
    for z, y in zip(b, A(x)):
        r.append([z_- y_ for z_, y_ in zip(z, y)])

    resvec = []
    relres = []
    # loop over maxit iterations (unless convergence or failure)
    for ii in range(maxit):
        if M1 is not None:
            y = M1(r)
        else:
            y = r

        if M2 is not None:
            z = M2(y)
        else:
            z = y

        rho1 = rho
        rho = ip(r, z)
        if rho == 0 or xp.isinf(rho):
            state['flag'] = 4
            break

        if ii == 0 and p is None:
            p = z
        else:
            if opts['CG_use_FR']:
                #  use Fletcher-Reeves
                beta = rho / rho1
            else:
                # Use Polak-Ribiere
                rho2 = ip(r_prev, z)
                beta = (rho - rho2) / rho1
            if beta == 0 or xp.isinf(beta):
                state['flag'] = 4
                break
            beta = max(0, beta)
            tmp = []
            for zz, pp in zip(z, p):
                tmp.append([zz_ + beta * pp_ for zz_, pp_ in zip(zz, pp)])
            p = tmp

        q = A(p)
        pq = ip(p, q)
        if pq <= 0 or xp.isinf(pq):
            state['flag'] = 4
            break
        else:
            if opts['CG_standard_alpha']:
                alpha = rho / pq
            else:
                alpha = ip(p, r) / pq
        if xp.isinf(alpha):
            state['flag'] = 4
        # save old r if not using FR formula for beta
        if not opts['CG_use_FR']:
            r_prev = r

        # form new iterate
        tmp = []
        for xx, pp in zip(x, p):
            tmp.append([xx_ + alpha * pp_ for xx_, pp_ in zip(xx, pp)])
        x = tmp
        if ii < maxit:
            tmp = []
            for rr, qq in zip(r, q):
                tmp.append([rr_ - alpha * qq_ for rr_, qq_ in zip(rr, qq)])
            r = tmp

    # save the state
    state['p'] = p
    state['rho'] = rho
    if not opts['CG_use_FR']:
        state['r_prev'] = r_prev
    return x, resvec, state

def train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, CG_opts, CG_state,config):
    """
        do conjugate graident optimization of the filter
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(hf[0][0])
    else:
        xp = np
    # construct the right hand side vector (A^H weight yf)
    rhs_samplef = [xp.matmul(xf, sample_weights) for xf in samplesf]
    rhs_samplef = [(xp.conj(xf) * yf[:,:,xp.newaxis,xp.newaxis])
            for xf, yf in zip(rhs_samplef, yf)]

    # construct preconditioner
    diag_M = [(1 - config.precond_reg_param) * (config.precond_data_param * m + (1 - config.precond_data_param) * xp.mean(m, 2, keepdims=True)) + \
              config.precond_reg_param * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    hf, _, CG_state = preconditioned_conjugate_gradient(
            lambda x: lhs_operation(x, samplesf, reg_filter, sample_weights), # A
            [rhs_samplef],                                                    # b
            CG_opts,                                                          # opts
            lambda x: diag_precond(x, [diag_M]),                              # M1
            None,                                                             # M2
            inner_product_filter,
            [hf],
            CG_state)
    # res_norms = res_norms / xp.sqrt(inner_product_filter([rhs_samplef], [rhs_samplef]))
    return hf[0], CG_state #res_norms, CG_state

def train_joint(hf, proj_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, init_CG_opts,config):
    """
        initial Gauss-Newton optimization of the filter and projection matrix
    """
    if gpu_config.use_gpu:
        xp = cp.get_array_module(hf[0][0])
    else:
        xp = np
    # index for the start of the last column of frequencies
    lf_ind = [x.shape[0] * (x.shape[1]-1) for x in hf[0]]

    # construct stuff for the proj matrix part
    init_samplef = xlf
    init_samplef_H = [xp.conj(x.reshape((-1, x.shape[2]))).T for x in init_samplef]

    # construct preconditioner
    diag_M = [[], []]
    diag_M[0] = [(1 - config.precond_reg_param) * (config.precond_data_param * m + (1 - config.precond_data_param) * xp.mean(m, 2, keepdims=True)) + \
                 config.precond_reg_param * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    diag_M[1] = [config.precond_proj_param * (m + config.projection_reg) for m in proj_energy]

    rhs_samplef = [[]] * len(hf[0])
    # res_norms = []
    for iter_ in range(config.init_GN_iter):
        # project sample with new matrix
        init_samplef_proj = [xp.matmul(P.T, x) for x, P in zip(init_samplef, proj_matrix)]
        init_hf = hf[0]

        # construct the right hand side vector for filter part
        rhs_samplef[0] = [xp.conj(xf) * yf_[:,:,xp.newaxis,xp.newaxis] for xf, yf_ in zip(init_samplef_proj, yf)]

        # construct the right hand side vector for the projection matrix part
        fyf = [xp.reshape(xp.conj(f) * yf_[:,:,xp.newaxis,xp.newaxis], (-1, f.shape[2])) for f, yf_ in zip(hf[0], yf)] # matlab reshape
        rhs_samplef[1] =[2 * xp.real(XH.dot(fyf_) - XH[:, fi:].dot(fyf_[fi:, :])) - config.projection_reg * P
                         for P, XH, fyf_, fi in zip(proj_matrix, init_samplef_H, fyf, lf_ind)]

        # initialize the projection matrix increment to zero
        hf[1] = [xp.zeros_like(P) for P in proj_matrix]

        # do conjugate gradient
        hf, _, _ = preconditioned_conjugate_gradient(
                lambda x: lhs_operation_joint(x, init_samplef_proj, reg_filter, init_samplef, init_samplef_H, init_hf, config.projection_reg), # A
                rhs_samplef,                                                                                                                   # b
                init_CG_opts,                                                                                                                  # opts
                lambda x: diag_precond(x, diag_M),                                                                                             # M1
                None,                                                                                                                          # M2
                inner_product_joint,
                hf)

        # make the filter symmetric
        hf[0] = symmetrize_filter(hf[0])

        # add to the projection matrix
        proj_matrix = [x + y for x, y in zip(proj_matrix, hf[1])]

        # res_norms.append(res_norms_temp)

    # extract filter
    hf = hf[0]
    # res_norms = res_norms / xp.sqrt(inner_product_joint(rhs_samplef, rhs_samplef))
    return hf, proj_matrix, # res_norms
