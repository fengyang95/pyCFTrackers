import json
import numpy as np
import matplotlib.pyplot as plt
from examples.vis.OTB100_info import OTB100,OTB50,IV,SV,OCC,DEF,COLOR,GRAY,OTB2013
from lib.utils import get_thresh_precision_pair,get_thresh_success_pair,calAUC
def get_preds_by_name(preds_dict,key):
    valid_keys=['gts','kcf_gray_preds','kcf_hog_preds','dcf_gray_preds',
                'dcf_hog_preds','mosse','csk','eco_hc','kcf_cn','kcf_pyECO_cn',
                'kcf_pyECO_hog','cn','DSST','DAT','Staple']
    assert key in valid_keys
    str_preds=preds_dict[key]
    np_preds=[]
    for bbox in str_preds:
        bbox=[int(a) for a in bbox]
        np_preds.append(bbox)
    np_preds=np.array(np_preds)
    return np_preds


def draw_plot(results_json_path,datalist,dataset_name):
    f = open(results_json_path, 'r')
    results = json.load(f)
    precisions_kcf_gray_all = np.zeros((101,))
    precisions_kcf_hog_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_dcf_gray_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_dcf_hog_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_mosse_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_csk_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_eco_hc_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_kcf_cn_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_kcf_pyECO_cn_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_kcf_pyECO_hog_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_cn_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_dsst_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_dat_all=np.zeros_like(precisions_kcf_gray_all)
    precisions_staple_all=np.zeros_like(precisions_kcf_gray_all)

    successes_kcf_gray_all = np.zeros((101,))
    successes_kcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    successes_dcf_gray_all = np.zeros_like(successes_kcf_gray_all)
    successes_dcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    successes_mosse_all = np.zeros_like(successes_kcf_gray_all)
    successes_csk_all = np.zeros_like(successes_kcf_gray_all)
    successes_eco_hc_all = np.zeros_like(successes_kcf_gray_all)
    successes_kcf_cn_all=np.zeros_like(successes_kcf_gray_all)
    successes_kcf_pyECO_cn_all=np.zeros_like(successes_kcf_gray_all)
    successes_kcf_pyECO_hog_all=np.zeros_like(successes_kcf_gray_all)
    successes_cn_all=np.zeros_like(successes_kcf_gray_all)
    successes_dsst_all=np.zeros_like(successes_kcf_gray_all)
    successes_dat_all=np.zeros_like(successes_kcf_gray_all)
    successes_staple_all=np.zeros_like(successes_kcf_gray_all)
    num_videos=0
    for data_name in results.keys():
        if data_name not in datalist:
            continue
        #print(data_name)
        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        kcf_gray_preds = get_preds_by_name(data_all, 'kcf_gray_preds')
        kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
        dcf_gray_preds = get_preds_by_name(data_all, 'dcf_gray_preds')
        dcf_hog_preds = get_preds_by_name(data_all, 'dcf_hog_preds')
        mosse_preds = get_preds_by_name(data_all, 'mosse')
        csk_preds = get_preds_by_name(data_all, 'csk')
        eco_hc_preds = get_preds_by_name(data_all, 'eco_hc')
        kcf_cn_preds=get_preds_by_name(data_all,'kcf_cn')
        kcf_pyECO_cn_preds=get_preds_by_name(data_all,'kcf_pyECO_cn')
        kcf_pyECO_hog_preds=get_preds_by_name(data_all,'kcf_pyECO_hog')
        cn_preds=get_preds_by_name(data_all,'cn')
        dsst_preds=get_preds_by_name(data_all,'DSST')
        dat_preds=get_preds_by_name(data_all,'DAT')
        staple_preds=get_preds_by_name(data_all,'Staple')

        precisions_kcf_gray_all += np.array(get_thresh_precision_pair(gts, kcf_gray_preds)[1])
        precisions_kcf_hog_all += np.array(get_thresh_precision_pair(gts, kcf_hog_preds)[1])
        precisions_dcf_gray_all += np.array(get_thresh_precision_pair(gts, dcf_gray_preds)[1])
        precisions_dcf_hog_all += np.array(get_thresh_precision_pair(gts, dcf_hog_preds)[1])
        precisions_mosse_all += np.array(get_thresh_precision_pair(gts, mosse_preds)[1])
        precisions_csk_all += np.array(get_thresh_precision_pair(gts, csk_preds)[1])
        precisions_eco_hc_all += np.array(get_thresh_precision_pair(gts, eco_hc_preds)[1])
        precisions_kcf_cn_all+=np.array(get_thresh_precision_pair(gts,kcf_cn_preds)[1])
        precisions_kcf_pyECO_cn_all+=np.array(get_thresh_precision_pair(gts,kcf_pyECO_cn_preds)[1])
        precisions_kcf_pyECO_hog_all+=np.array(get_thresh_precision_pair(gts,kcf_pyECO_hog_preds)[1])
        precisions_cn_all+=np.array(get_thresh_precision_pair(gts,cn_preds)[1])
        precisions_dsst_all+=np.array(get_thresh_precision_pair(gts,dsst_preds)[1])
        precisions_dat_all+=np.array(get_thresh_precision_pair(gts,dat_preds)[1])
        precisions_staple_all+=np.array(get_thresh_precision_pair(gts,staple_preds)[1])

        successes_kcf_gray_all += np.array(get_thresh_success_pair(gts, kcf_gray_preds)[1])
        successes_kcf_hog_all += np.array(get_thresh_success_pair(gts, kcf_hog_preds)[1])
        successes_dcf_gray_all += np.array(get_thresh_success_pair(gts, dcf_gray_preds)[1])
        successes_dcf_hog_all += np.array(get_thresh_success_pair(gts, dcf_hog_preds)[1])
        successes_mosse_all += np.array(get_thresh_success_pair(gts, mosse_preds)[1])
        successes_csk_all += np.array(get_thresh_success_pair(gts, csk_preds)[1])
        successes_eco_hc_all += np.array(get_thresh_success_pair(gts, eco_hc_preds)[1])
        successes_kcf_cn_all+=np.array(get_thresh_success_pair(gts,kcf_cn_preds)[1])
        successes_kcf_pyECO_cn_all+=np.array(get_thresh_success_pair(gts,kcf_pyECO_cn_preds)[1])
        successes_kcf_pyECO_hog_all+=np.array(get_thresh_success_pair(gts,kcf_pyECO_hog_preds)[1])
        successes_cn_all+=np.array(get_thresh_success_pair(gts,cn_preds)[1])
        successes_dsst_all+=np.array(get_thresh_success_pair(gts,dsst_preds)[1])
        successes_dat_all+=np.array(get_thresh_success_pair(gts,dat_preds)[1])
        successes_staple_all+=np.array(get_thresh_success_pair(gts,staple_preds)[1])

    precisions_kcf_gray_all /= num_videos
    precisions_kcf_hog_all /= num_videos
    precisions_dcf_gray_all /= num_videos
    precisions_dcf_hog_all /= num_videos
    precisions_csk_all /= num_videos
    precisions_mosse_all /= num_videos
    precisions_eco_hc_all /= num_videos
    precisions_kcf_cn_all/=num_videos
    precisions_kcf_pyECO_cn_all/=num_videos
    precisions_kcf_pyECO_hog_all/=num_videos
    precisions_cn_all/=num_videos
    precisions_dsst_all/=num_videos
    precisions_dat_all/=num_videos
    precisions_staple_all/=num_videos

    successes_kcf_gray_all /= num_videos
    successes_kcf_hog_all /= num_videos
    successes_dcf_gray_all /= num_videos
    successes_dcf_hog_all /= num_videos
    successes_csk_all /= num_videos
    successes_mosse_all /= num_videos
    successes_eco_hc_all /= num_videos
    successes_kcf_cn_all/=num_videos
    successes_kcf_pyECO_cn_all/=num_videos
    successes_kcf_pyECO_hog_all/=num_videos
    successes_cn_all/=num_videos
    successes_dsst_all/=num_videos
    successes_dat_all/=num_videos
    successes_staple_all/=num_videos

    threshes_precision = np.linspace(0, 50, 101)
    threshes_success = np.linspace(0, 1, 101)

    idx20 = [i for i, x in enumerate(threshes_precision) if x == 20][0]
    plt.plot(threshes_precision, precisions_eco_hc_all, label='ECO-HC ' + str(precisions_eco_hc_all[idx20])[:5])
    #plt.plot(threshes_precision, precisions_kcf_gray_all, label='KCF_GRAY ' + str(precisions_kcf_gray_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_kcf_hog_all, label='KCF_HOG ' + str(precisions_kcf_hog_all[idx20])[:5])
    #plt.plot(threshes_precision, precisions_dcf_gray_all, label='DCF_GRAY ' + str(precisions_dcf_gray_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_dcf_hog_all, label='DCF_HOG ' + str(precisions_dcf_hog_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_mosse_all, label='MOSSE ' + str(precisions_mosse_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_csk_all, label='CSK ' + str(precisions_csk_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_cn_all,label='KCF_CN '+str(precisions_kcf_cn_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_pyECO_cn_all,label='KCF_pyECO_CN '+str(precisions_kcf_pyECO_cn_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_pyECO_hog_all,label='KCF_pyECO_HOG '+str(precisions_kcf_pyECO_hog_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_cn_all,label='CN '+str(precisions_cn_all[idx20])[:5])
    plt.plot(threshes_precision,precisions_dsst_all,label='DSST '+str(precisions_dsst_all[idx20])[:5])
    plt.plot(threshes_precision,precisions_dat_all,label='DAT '+str(precisions_dat_all[idx20])[:5])
    plt.plot(threshes_precision,precisions_staple_all,label='Staple '+str(precisions_staple_all[idx20])[:5])
    plt.title(dataset_name + ' Precision plot of OPE')
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(dataset_name + '_precision.jpg')
    plt.clf()
    # plt.show()

    plt.plot(threshes_success, successes_eco_hc_all, label='ECO-HC' + str(calAUC(successes_eco_hc_all))[:5])
    #plt.plot(threshes_success, successes_kcf_gray_all, label='KCF_GRAY ' + str(calAUC(successes_kcf_gray_all))[:5])
    plt.plot(threshes_success, successes_kcf_hog_all, label='KCF_HOG ' + str(calAUC(successes_kcf_hog_all))[:5])
    #plt.plot(threshes_success, successes_dcf_gray_all, label='DCF_GRAY ' + str(calAUC(successes_dcf_gray_all))[:5])
    plt.plot(threshes_success, successes_dcf_hog_all, label='DCF_HOG ' + str(calAUC(successes_dcf_hog_all))[:5])
    plt.plot(threshes_success, successes_mosse_all, label='MOSSE ' + str(calAUC(successes_mosse_all))[:5])
    plt.plot(threshes_success, successes_csk_all, label='CSK ' + str(calAUC(successes_csk_all))[:5])
    #plt.plot(threshes_success,successes_kcf_cn_all,label='KCF_CN '+str(calAUC(successes_kcf_cn_all))[:5])
    #plt.plot(threshes_success,successes_kcf_pyECO_cn_all,label='KCF_pyECO_CN '+str(calAUC(successes_kcf_pyECO_cn_all))[:5])
    #plt.plot(threshes_success,successes_kcf_pyECO_hog_all,label='KCF_pyECO_HOG '+str(calAUC(successes_kcf_pyECO_hog_all))[:5])
    #plt.plot(threshes_success,successes_cn_all,label='CN '+str(calAUC(successes_cn_all))[:5])
    plt.plot(threshes_success,successes_dsst_all,label='DSST '+str(calAUC(successes_dsst_all))[:5])
    plt.plot(threshes_success,successes_dat_all,label='DAT '+str(calAUC(successes_dat_all))[:5])
    plt.plot(threshes_success,successes_staple_all,label='Staple '+str(calAUC(successes_staple_all))[:5])
    plt.title(dataset_name + ' Success plot of OPE')
    plt.xlabel('Overlap Threshold')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig(dataset_name + '_success.jpg')
    plt.clf()
    print(dataset_name,':',num_videos)


if __name__=='__main__':
    result_json_path='../otb100_results.json'

    draw_plot(result_json_path,OTB100,'OTB100')
    draw_plot(result_json_path,OTB50,'OTB50')
    draw_plot(result_json_path,IV,'IV')
    draw_plot(result_json_path,SV,'SV')
    draw_plot(result_json_path,OCC,'OCC')
    draw_plot(result_json_path,DEF,'DEF')
    draw_plot(result_json_path,COLOR,'COLOR')
    draw_plot(result_json_path,GRAY,'GRAY')
    draw_plot(result_json_path,OTB2013,'OTB2013')






