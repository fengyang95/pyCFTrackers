import json
import numpy as np
import os
import cv2
from examples.vis.OTB100_info import OTB100,OTB50,IV,SV,OCC,DEF
from lib.utils import get_img_list
from examples.vis.draw_plot import get_preds_by_name
from lib.utils import get_thresh_success_pair,calAUC

def vis_results(results_json_path,dataset_dir,data_name):
    f = open(results_json_path, 'r')
    results = json.load(f)
    if not data_name in results.keys():
        raise ValueError
    cap=cv2.VideoCapture(data_name+'_vis.avi')
    data_all = results[data_name]
    gts = get_preds_by_name(data_all, 'gts')
    kcf_gray_preds = get_preds_by_name(data_all, 'kcf_gray_preds')
    kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
    dcf_gray_preds = get_preds_by_name(data_all, 'dcf_gray_preds')
    dcf_hog_preds = get_preds_by_name(data_all, 'dcf_hog_preds')
    mosse_preds = get_preds_by_name(data_all, 'mosse')
    dsst_preds=get_preds_by_name(data_all,'DSST')
    csk_preds = get_preds_by_name(data_all, 'csk')
    eco_hc_preds = get_preds_by_name(data_all, 'eco_hc')
    #kcf_cn_preds = get_preds_by_name(data_all, 'kcf_cn')
    #kcf_pyECO_cn_preds = get_preds_by_name(data_all, 'kcf_pyECO_cn')
    img_dir=os.path.join(dataset_dir,data_name)
    img_dir=os.path.join(img_dir,'img')
    img_list=get_img_list(img_dir)
    color_gt=(0,97,255)# gt 橙色
    color_mosse=(0,255,0)# mosse green
    color_csk=(0,0,255)# csk red
    color_eco=(100,0,100) #
    color_kcf=(255,0,0)
    color_dsst=(240,32,160)#
    writer=None
    for i in range(len(csk_preds)):
        flag,current_frame=cap.read()
        if flag is False:
            return
        gt=gts[i]
        mosse_pred=mosse_preds[i]
        csk_pred=csk_preds[i]
        eco_hc_pred=eco_hc_preds[i]
        dsst_pred=dsst_preds[i]

        show_frame=cv2.rectangle(current_frame,(gt[0],gt[1]),(gt[0]+gt[2],gt[1]+gt[3]),color_gt,thickness=2)

        show_frame=cv2.rectangle(show_frame,(mosse_pred[0],mosse_pred[1]),
                                 (mosse_pred[0]+mosse_pred[2],mosse_pred[1]+mosse_pred[3]),
                                 color_mosse)
        show_frame=cv2.rectangle(show_frame,(csk_pred[0],csk_pred[1]),
                                 (csk_pred[0]+csk_pred[2],csk_pred[1]+csk_pred[3]),
                                 color_csk)
        show_frame=cv2.rectangle(show_frame,(eco_hc_pred[0],eco_hc_pred[1]),
                                 (eco_hc_pred[0]+eco_hc_pred[2],eco_hc_pred[1]+eco_hc_pred[3]),
                                 color_eco)
        show_frame=cv2.rectangle(show_frame,(dsst_pred[0],dsst_pred[1]),
                                 (dsst_pred[0]+dsst_pred[2],dsst_pred[1]+dsst_pred[3]),
                                 color_dsst)

        threshes,kcf_success=get_thresh_success_pair(gts,kcf_hog_preds)
        _,csk_success=get_thresh_success_pair(gts,csk_preds)
        _,mosse_success=get_thresh_success_pair(gts,mosse_preds)
        _,dsst_success=get_thresh_success_pair(gts,dsst_preds)
        _,ecohc_success=get_thresh_success_pair(gts,eco_hc_preds)
        _,gts_success=get_thresh_success_pair(gts,gts)
        show_frame = cv2.putText(show_frame, 'MOSSE ' + str(calAUC(mosse_success))[:5], (50, 40),
                                 cv2.FONT_HERSHEY_COMPLEX, 1, color_mosse)
        show_frame=cv2.putText(show_frame,'KCF '+str(calAUC(kcf_success))[:5],(50,120),cv2.FONT_HERSHEY_COMPLEX,1,color_kcf)
        show_frame=cv2.putText(show_frame,'CSK '+str(calAUC(csk_success))[:5],(50,80),cv2.FONT_HERSHEY_COMPLEX,1,color_csk)
        show_frame=cv2.putText(show_frame,'ECO '+str(calAUC(dsst_success))[:5],(50,200),cv2.FONT_HERSHEY_COMPLEX,1,color_eco)
        show_frame=cv2.putText(show_frame,'DSST '+str(calAUC(ecohc_success))[:5],(50,160),cv2.FONT_HERSHEY_COMPLEX,1,color_dsst)
        show_frame=cv2.putText(show_frame,'GT',(50,240),cv2.FONT_HERSHEY_COMPLEX,1,color_gt)
        cv2.imshow('demo',show_frame)
        if writer is None:
            writer = cv2.VideoWriter(data_name+'_res.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5,
                                     (show_frame.shape[1], show_frame.shape[0]))
        writer.write(show_frame)
        cv2.waitKey(30)


if __name__=='__main__':
    result_json_path='../otb100_results.json'
    vis_results(result_json_path,'../../dataset/OTB100','Deer')






