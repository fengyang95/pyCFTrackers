import cv2
import numpy as np
import lib.fhog as fhog
from lib.eco.features.features import FHogFeature,TableFeature

# pyECO pos->(y,x) format
def extract_hog_feature_pyECO(img, center, sz, cell_size=4):
    hog=FHogFeature(fname='fhog',cell_size=cell_size, compressed_dim=10, num_orients=9,clip=1.)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    fhog_feature=hog.get_features(img,(center[1],center[0]),sz,1.,normalization=True)[0][:,:,:,0]
    return fhog_feature

# pyECO pos——>(y,x)
def extract_cn_feature_pyECO(img, center, sz, cell_size=1):
    cn=TableFeature(fname='cn',cell_size=cell_size,compressed_dim=11,table_name="CNnorm",
                 use_for_color= True)
    sz=np.array([sz[0],sz[1]])
    cn_feature=cn.get_features(img,(center[1],center[0]),sz,1,normalization=False)[0][:,:,:,0]
    patch  = cn._sample_patch(img, (center[1],center[0]), sz, sz)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    gray = gray[:, :, np.newaxis]
    out = np.concatenate((gray,cn_feature), axis=2)
    return out

def extract_cn_feature(img, center, sz, w2c):
    patch=cv2.getRectSubPix(img,sz,center)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    gray = gray[:, :, np.newaxis]

    b, g, r = cv2.split(patch)
    index_im = ( r//8 + 32 * g//8 + 32 * 32 * b//8)
    h, w = patch.shape[:2]
    w2c=np.array(w2c)
    out=w2c.T[index_im.flatten()].reshape((h,w,w2c.shape[0]))
    out=np.concatenate((gray,out),axis=2)
    return out

def extract_hog_feature(img, cell_size=4):
    h,w=img.shape[:2]
    img=cv2.resize(img,(w+2*cell_size,h+2*cell_size))
    mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
    mapp = fhog.getFeatureMaps(img, cell_size, mapp)
    mapp = fhog.normalizeAndTruncate(mapp, 0.2)
    mapp = fhog.PCAFeatureMaps(mapp)
    size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
    FeaturesMap = mapp['map'].reshape(
        (size_patch[0],size_patch[1], size_patch[2]))
    return FeaturesMap


