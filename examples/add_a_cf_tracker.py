import os
import numpy as np
import matplotlib.pyplot as plt
from examples.pytracker import PyTracker
import json
from lib.utils import get_ground_truthes,get_thresh_success_pair,get_thresh_precision_pair,calAUC
from examples.otbdataset_config import OTBDatasetConfig

def add_a_tracker(results_json_path,tracker_type,dst_json_path):
    data_dir = '../dataset/OTB100'
    data_names = os.listdir(data_dir)
    f = open(results_json_path, 'r')
    otb100_results = json.load(f)
    dataset_config = OTBDatasetConfig()
    for data_name in data_names:
        print('data name:', data_name)

        data_path = os.path.join(data_dir, data_name)
        img_dir = os.path.join(data_path,'img')
        tracker=PyTracker(img_dir, tracker_type=tracker_type,dataset_config=dataset_config)

        gts=get_ground_truthes(data_path)
        if data_name in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[data_name][:2]
            if data_name!='David':
                gts = gts[start_frame - 1:end_frame]
        otb100_results[data_name]['gts']=[]
        for gt in gts:
            otb100_results[data_name]['gts'].append(list(gt.astype(np.int)))

        tracker_preds=tracker.tracking()
        otb100_results[data_name][tracker_type] = []
        for tracker_pred in tracker_preds:
            otb100_results[data_name][tracker_type].append(list(tracker_pred.astype(np.int)))

        threshes,precisions_tracker=get_thresh_precision_pair(gts,tracker_preds)
        idx20=[i for i, x in enumerate(threshes) if x==20][0]

        plt.plot(threshes, precisions_tracker, label=tracker_type+' '+str(precisions_tracker[idx20])[:5])
        plt.title(data_name+' Precision')
        plt.xlabel('thresh')
        plt.ylabel('precision')
        plt.legend()
        if not os.path.exists('../results/tmp'):
            print('test')
            os.mkdir('../results/tmp/')
        plt.savefig('../results/tmp/'+data_name+'_precision.jpg')
        plt.clf()


        threshes,successes_tracker=get_thresh_success_pair(gts, tracker_preds)

        plt.plot(threshes,successes_tracker,label=tracker_type+' '+str(calAUC(successes_tracker))[:5])

        plt.title(data_name+' Success')
        plt.xlabel('thresh')
        plt.ylabel('success')
        plt.legend()

        plt.savefig('../results/tmp/'+data_name+'_success.jpg')
        plt.clf()

    json_content = json.dumps(otb100_results, default=str)
    f = open(dst_json_path, 'w')
    f.write(json_content)
    f.close()

if __name__ == '__main__':
    add_a_tracker('otb100_results.json','KCF_pyECO_HOG','otb100_results.json')






