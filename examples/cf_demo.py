import os
from examples.pytracker import PyTracker
from lib.utils import get_ground_truthes,plot_precision,plot_success
from examples.otbdataset_config import OTBDatasetConfig
if __name__ == '__main__':
    data_dir='../dataset/test'
    data_names=sorted(os.listdir(data_dir))

    print(data_names)
    dataset_config=OTBDatasetConfig()
    for data_name in data_names:
        data_path=os.path.join(data_dir,data_name)
        gts = get_ground_truthes(data_path)
        if data_name in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[data_name][:2]
            if data_name!='David':
                gts=gts[start_frame-1:end_frame]
        img_dir = os.path.join(data_path,'img')
        tracker = PyTracker(img_dir,tracker_type='STRCF',dataset_config=dataset_config)
        poses=tracker.tracking(verbose=True,video_path=os.path.join('../results/CF',data_name+'_vis.avi'))
        plot_success(gts,poses,os.path.join('../results/CF',data_name+'_success.jpg'))
        plot_precision(gts,poses,os.path.join('../results/CF',data_name+'_precision.jpg'))
