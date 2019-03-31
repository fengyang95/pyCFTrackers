import os
import numpy as np
import matplotlib.pyplot as plt
from examples.pytracker import PyTracker
import json
from lib.utils import get_ground_truthes,get_thresh_success_pair,get_thresh_precision_pair,calAUC
from examples.otbdataset_config import OTBDatasetConfig
if __name__ == '__main__':
    data_dir = '../dataset/OTB100'
    data_names = os.listdir(data_dir)
    otb100_results = {}

    dataset_config=OTBDatasetConfig()

    for data_name in data_names:
        print('data name:', data_name)
        otb100_results[data_name]={}
        data_path = os.path.join(data_dir, data_name)
        img_dir = os.path.join(data_path,'img')
        tracker_kcf_gray=PyTracker(img_dir, tracker_type='KCF_GRAY',dataset_config=dataset_config)
        tracker_kcf_hog=PyTracker(img_dir,tracker_type='KCF_HOG',dataset_config=dataset_config)
        tracker_dcf_gray=PyTracker(img_dir,tracker_type='DCF_GRAY',dataset_config=dataset_config)
        tracker_dcf_hog=PyTracker(img_dir,tracker_type='DCF_HOG',dataset_config=dataset_config)
        tracker_mosse=PyTracker(img_dir,tracker_type='MOSSE',dataset_config=dataset_config)
        tracker_csk=PyTracker(img_dir,tracker_type='CSK',dataset_config=dataset_config)
        tracker_kcf_cn=PyTracker(img_dir,tracker_type='KCF_CN',dataset_config=dataset_config)
        tracker_kcf_pyECO_cn=PyTracker(img_dir,tracker_type='KCF_pyECO_CN',dataset_config=dataset_config)
        tracker_kcf_pyECO_hog=PyTracker(img_dir,tracker_type='KCF_pyECO_HOG',dataset_config=dataset_config)
        tracker_cn=PyTracker(img_dir,tracker_type='CN',dataset_config=dataset_config)
        gts=get_ground_truthes(data_path)
        if data_name in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[data_name][:2]
            if data_name!='David':
                gts = gts[start_frame - 1:end_frame]
        otb100_results[data_name]['gts']=[]
        for gt in gts:
            otb100_results[data_name]['gts'].append(list(gt.astype(np.int)))

        kcf_gray_preds=tracker_kcf_gray.tracking()
        otb100_results[data_name]['kcf_gray_preds']=[]
        for kcf_gray_pred in kcf_gray_preds:
            otb100_results[data_name]['kcf_gray_preds'].append(list(kcf_gray_pred.astype(np.int)))
        print('kcf gray done!')

        kcf_hog_preds=tracker_kcf_hog.tracking()
        otb100_results[data_name]['kcf_hog_preds'] = []
        for kcf_hog_pred in kcf_hog_preds:
            otb100_results[data_name]['kcf_hog_preds'].append(list(kcf_hog_pred.astype(np.int)))
        print('kcf hog done!')

        dcf_gray_preds=tracker_dcf_gray.tracking()
        otb100_results[data_name]['dcf_gray_preds'] = []
        for dcf_gray_pred in dcf_gray_preds:
            otb100_results[data_name]['dcf_gray_preds'].append(list(dcf_gray_pred.astype(np.int)))
        print('dcf gray done!')

        dcf_hog_preds=tracker_dcf_hog.tracking()
        otb100_results[data_name]['dcf_hog_preds'] = []
        for dcf_hog_pred in dcf_hog_preds:
            otb100_results[data_name]['dcf_hog_preds'].append(list(dcf_hog_pred.astype(np.int)))
        print('dcf hog done!')

        mosse_preds=tracker_mosse.tracking()
        otb100_results[data_name]['mosse'] = []
        for mosse_pred in mosse_preds:
            otb100_results[data_name]['mosse'].append(list(mosse_pred.astype(np.int)))
        print('mosse done!')

        csk_preds=tracker_csk.tracking()
        otb100_results[data_name]['csk'] = []
        for csk_pred in csk_preds:
            otb100_results[data_name]['csk'].append(list(csk_pred.astype(np.int)))
        print('csk done!')

        kcf_cn_preds=tracker_kcf_cn.tracking()
        otb100_results[data_name]['kcf_cn']=[]
        for kcf_cn_pred in kcf_cn_preds:
            otb100_results[data_name]['kcf_cn'].append(list(kcf_cn_pred.astype(np.int)))
        print('kcf_cn done!')


        kcf_pyECO_cn_preds = tracker_kcf_pyECO_cn.tracking()
        otb100_results[data_name]['kcf_pyECO_cn'] = []
        for kcf_cn_pred in kcf_cn_preds:
            otb100_results[data_name]['kcf_pyECO_cn'].append(list(kcf_cn_pred.astype(np.int)))
        print('kcf_pyECO_cn done!')

        kcf_pyECO_hog_preds=tracker_kcf_pyECO_hog.tracking()
        otb100_results[data_name]['kcf_pyECO_hog']=[]
        for kcf_pyECO_hog_pred in kcf_pyECO_hog_preds:
            otb100_results[data_name]['kcf_pyECO_hog'].append(list(kcf_pyECO_hog_pred.astype(np.int)))
        print('kcf_pyECO_hog done!')

        cn_preds=tracker_cn.tracking()
        otb100_results[data_name]['cn']=[]
        for cn_pred in cn_preds:
            otb100_results[data_name]['cn'].append(list(cn_pred.astype(np.int)))
        print('cn done!')

        threshes,precisions_kcf_gray=get_thresh_precision_pair(gts,kcf_gray_preds)
        _,precisions_kcf_hog=get_thresh_precision_pair(gts,kcf_hog_preds)
        _,precisions_dcf_gray=get_thresh_precision_pair(gts,dcf_gray_preds)
        _,precisions_dcf_hog=get_thresh_precision_pair(gts,dcf_hog_preds)
        _,precisions_mosse=get_thresh_precision_pair(gts,mosse_preds)
        _,precisions_csk=get_thresh_precision_pair(gts,csk_preds)
        _,precisions_kcf_cn=get_thresh_precision_pair(gts,kcf_cn_preds)
        _,precisions_kcf_pyECO_cn=get_thresh_precision_pair(gts,kcf_pyECO_cn_preds)
        _,precisions_kcf_pyECO_hog=get_thresh_precision_pair(gts,kcf_pyECO_hog_preds)
        _,precisions_cn=get_thresh_precision_pair(gts,cn_preds)
        idx20=[i for i, x in enumerate(threshes) if x==20][0]

        plt.plot(threshes, precisions_kcf_gray, label='KCF_GRAY '+str(precisions_kcf_gray[idx20])[:5])
        plt.plot(threshes,precisions_kcf_hog,label='KCF_HOG '+str(precisions_kcf_hog[idx20])[:5])
        plt.plot(threshes,precisions_dcf_gray,label='DCF_GRAY '+str(precisions_dcf_gray[idx20])[:5])
        plt.plot(threshes,precisions_dcf_hog,label='DCF_HOG '+str(precisions_dcf_hog[idx20])[:5])
        plt.plot(threshes,precisions_mosse,label='MOSSE '+str(precisions_mosse[idx20])[:5])
        plt.plot(threshes,precisions_csk,label='CSK '+str(precisions_csk[idx20])[:5])
        plt.plot(threshes,precisions_kcf_cn,label='KCF_CN '+str(precisions_kcf_cn[idx20])[:5])
        plt.plot(threshes,precisions_kcf_pyECO_cn,label='KCF_pyECO_CN '+str(precisions_kcf_pyECO_cn[idx20])[:5])
        plt.plot(threshes,precisions_kcf_pyECO_hog,label='KCF_pyECO_HOG '+str(precisions_kcf_pyECO_hog[idx20])[:5])
        plt.plot(threshes,precisions_cn,label='CN '+str(precisions_cn[idx20])[:5])
        plt.title(data_name+' Precision')
        plt.xlabel('thresh')
        plt.ylabel('precision')
        plt.legend()
        plt.savefig('../results/OTB100_cftrackers/'+data_name+'_precision.jpg')
        plt.clf()


        threshes,successes_kcf_gray=get_thresh_success_pair(gts, kcf_gray_preds)
        _,successes_kcf_hog=get_thresh_success_pair(gts,kcf_hog_preds)
        _,successes_dcf_gray=get_thresh_success_pair(gts,dcf_gray_preds)
        _,successes_dcf_hog=get_thresh_success_pair(gts,dcf_hog_preds)
        _,successes_mosse=get_thresh_success_pair(gts,mosse_preds)
        _,successes_csk=get_thresh_success_pair(gts,csk_preds)
        _,successes_kcf_cn=get_thresh_success_pair(gts,kcf_cn_preds)
        _,successes_kcf_pyECO_cn=get_thresh_success_pair(gts,kcf_pyECO_cn_preds)
        _,successes_kcf_pyECO_hog=get_thresh_success_pair(gts,kcf_pyECO_hog_preds)
        _,successes_cn=get_thresh_success_pair(gts,cn_preds)
        plt.plot(threshes,successes_kcf_cn,label='KCF_CN '+str(calAUC(successes_kcf_cn))[:5])
        plt.plot(threshes,successes_kcf_pyECO_cn,label='KCF_pyECO_CN '+str(calAUC(successes_kcf_pyECO_cn))[:5])
        plt.plot(threshes, successes_kcf_gray, label='KCF_GRAY '+str(calAUC(successes_kcf_gray))[:5])
        plt.plot(threshes,successes_kcf_hog,label='KCF_HOG '+str(calAUC(successes_kcf_hog))[:5])
        plt.plot(threshes,successes_dcf_gray,label='DCF_GRAY '+str(calAUC(successes_dcf_gray))[:5])
        plt.plot(threshes,successes_dcf_hog,label='DCF_HOG '+str(calAUC(successes_dcf_hog))[:5])
        plt.plot(threshes,successes_mosse,label='MOSSE '+str(calAUC(successes_mosse))[:5])
        plt.plot(threshes,successes_csk,label='CSK '+str(calAUC(successes_csk))[:5])
        plt.plot(threshes,successes_kcf_pyECO_hog,label='KCF_pyECO_HOG '+str(calAUC(successes_kcf_pyECO_hog))[:5])
        plt.plot(threshes,successes_cn,label='CN '+str(calAUC(successes_cn))[:5])

        plt.title(data_name+' Success')
        plt.xlabel('thresh')
        plt.ylabel('success')
        plt.legend()
        plt.savefig('../results/OTB100_cftrackers/'+data_name+'_success.jpg')
        plt.clf()

    json_content = json.dumps(otb100_results, default=str)
    f = open('otb100_results.json', 'w')
    f.write(json_content)
    f.close()



