import numpy as np
from cftracker.strcf import STRCF
from cftracker.config import strdcf_hc_config

if __name__=='__main__':
    strcf=STRCF(config=strdcf_hc_config.STRDCFHCConfig())
    strcf.yf=np.array([[1,0,1],[0,1,0],[1,1,1]])
    strcf.reg_window=np.array([[1,1,0],[0,1,0],[1,0,1]])
    strcf.feature_map_sz=(3,3)
    xlf1=np.array([[3,3,3],[2,3,2],[1,2,1]])[:,:,np.newaxis]
    xlf2=np.array([[4,3,1.2],[4,35,3],[3,23,4]])[:,:,np.newaxis]
    xlf3=xlf1-xlf2
    xlf=np.concatenate((xlf1,xlf2,xlf3),axis=2)
    f_pre_f=np.zeros_like(xlf)
    mu=10

    res=strcf.ADMM(xlf,f_pre_f,mu)[:,:,1]
    print(res)