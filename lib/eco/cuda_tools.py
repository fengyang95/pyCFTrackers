from .config import gpu_config
if gpu_config.use_gpu:
    import cupy as cp

# https://github.com/chainer/chainer/blob/master/chainer/utils/conv.py
def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the convolution operation.

    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1

def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False, dy=1, dx=1,
               out_h=None, out_w=None):
    """
        img NxCxHxW
        kh: kernel height
        kw: kernle width
        sy: stride y
        sx: stride x
        ph: padding height
        pw: padding width
    """
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    col = cp.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    cp.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)
    return col

def col2im_gpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = cp.empty((n, c, h, w), dtype=col.dtype)
    cp.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img

def convolve2d(in1, in2, mode='full'):
    """
        note only support H * W * N * 1 convolve 2d
    """
    in1 = in1.transpose(2, 3, 0, 1) # to N * C * H * W
    in2 = in2.transpose(2, 3, 0, 1)
    out_c, _, kh, kw = in2.shape
    n, _, h, w = in1.shape

    if mode == 'full':
        ph, pw = kh-1, kw-1
        out_h, out_w = h-kh+1+ph*2, w-kw+1+pw*2# TODO
    elif mode == 'valid':
        ph, pw = 0, 0
        out_h, out_w = h-kh+1, w-kw+1 # TODO
    else:
        raise NotImplementedError

    y = cp.empty((n, out_c, out_h, out_w), dtype=in1.dtype)

    col = im2col_gpu(in1, kh, kw, 1, 1, ph, pw)
    y = cp.tensordot(
            col, in2, ((1, 2, 3), (1, 2, 3))).astype(in1.dtype, copy=False)
    y = cp.rollaxis(y, 3, 1)
    return y.transpose(2, 3, 0, 1)

if __name__ == '__main__':
    import cupy as cp
    import numpy as np
    from scipy.signal import convolve

    a = np.random.randn(5, 5, 5, 1) + 1j*np.random.randn(5,5,5,1)
    b = np.random.randn(3, 3, 1, 1) + 1j*np.random.randn(3,3,1,1)
    y_cpu = convolve(a, b, 'valid')

    x = cp.asarray(a)
    w = cp.asarray(b)
    y_gpu = convolve2d(x, w, 'valid')

    np.allclose(y_gpu.get().squeeze(), y_cpu.squeeze(), atol=1e-6)

