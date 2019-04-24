import numpy as np
def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)

def cifft2(xf):
    x = np.real(ifft2(np.fft.ifftshift(np.fft.ifftshift(xf, 0),1))).astype(np.float32)
    return x

def cfft2(x):
    in_shape = x.shape
    if in_shape[0] % 2 == 1 and in_shape[1] % 2 == 1:
        xf = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
    else:
        out_shape = list(in_shape)
        out_shape[0] =  in_shape[0] + (in_shape[0] + 1) % 2
        out_shape[1] =  in_shape[1] + (in_shape[1] + 1) % 2
        out_shape = tuple(out_shape)
        xf = np.zeros(out_shape, dtype=np.complex64)
        xf[:in_shape[0], :in_shape[1]] = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
        if out_shape[0] != in_shape[0]:
            xf[-1,:] = np.conj(xf[0,::-1])
        if out_shape[1] != in_shape[1]:
            xf[:,-1] = np.conj(xf[::-1,0])
    return xf[:in_shape[0],:in_shape[1]]
