from __future__ import division
import argparse
import logging
import cv2
from os import makedirs
from os.path import join, isdir

from lib.log_helper import init_log, add_file_handler

from cftracker.mosse import MOSSE
from cftracker.staple import Staple
from cftracker.dsst import DSST
from cftracker.samf import SAMF
from cftracker.kcf import KCF
from cftracker.csk import CSK
from cftracker.cn import CN
from cftracker.dat import DAT
from cftracker.eco import ECO
from cftracker.bacf import BACF
from cftracker.csrdcf import CSRDCF
from cftracker.ldes import LDES
from cftracker.mkcfup import MKCFup
from cftracker.strcf import STRCF
from cftracker.mccth_staple import MCCTHStaple
from cftracker.opencv_cftracker import OpenCVCFTracker
from cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config
from lib.eco.config import otb_deep_config,otb_hc_config

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--dataset', dest='dataset', default='OTB100',
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result',default=True)
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')

from lib.pysot.datasets import DatasetFactory

def track_otb(tracker_type,dataset):
    for video in dataset:
        regions = []
        print('video:',video.name)
        if tracker_type == 'MOSSE':
            tracker=MOSSE()
        elif tracker_type=='CSK':
            tracker=CSK()
        elif tracker_type=='CN':
            tracker=CN()
        elif tracker_type=='DSST':
            tracker=DSST(dsst_config.DSSTConfig())
        elif tracker_type=='SAMF':
            tracker=SAMF()
        elif tracker_type=='Staple':
            tracker=Staple(config=staple_config.StapleConfig())
        elif tracker_type=='Staple-CA':
            tracker=Staple(config=staple_config.StapleCAConfig())
        elif tracker_type=='KCF':
            tracker=KCF(features='hog',kernel='gaussian')
        elif tracker_type=='DCF':
            tracker=KCF(features='hog',kernel='linear')
        elif tracker_type=='DAT':
            tracker=DAT()
        elif tracker_type=='ECO-HC':
            tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif tracker_type=='ECO':
            tracker=ECO(config=otb_deep_config.OTBDeepConfig())
        elif tracker_type=='BACF':
            tracker=BACF()
        elif tracker_type=='CSRDCF':
            tracker=CSRDCF(csrdcf_config.CSRDCFConfig())
        elif  tracker_type=='CSRDCF-LP':
            tracker=CSRDCF(csrdcf_config.CSRDCFLPConfig())
        elif tracker_type == 'OPENCV_KCF':
            tracker = OpenCVCFTracker(name='KCF')
        elif tracker_type == 'OPENCV_MOSSE':
            tracker = OpenCVCFTracker(name='MOSSE')
        elif tracker_type == 'OPENCV-CSRDCF':
            tracker = OpenCVCFTracker(name='CSRDCF')
        elif tracker_type=='LDES':
            tracker=LDES(ldes_config.LDESOTBLinearConfig())
        elif tracker_type=='LDES-NoBGD':
            tracker=LDES(ldes_config.LDESOTBNoBGDLinearConfig())
        elif tracker_type=='DSST-LP':
            tracker=DSST(dsst_config.DSSTLPConfig())
        elif tracker_type=='MKCFup':
            tracker=MKCFup(mkcf_up_config.MKCFupConfig())
        elif tracker_type=='MKCFup-LP':
            tracker=MKCFup(mkcf_up_config.MKCFupLPConfig())
        elif tracker_type=='STRCF':
            tracker=STRCF()
        elif tracker_type=='MCCTH-Staple':
            tracker=MCCTHStaple(mccth_staple_config.MCCTHOTBConfig())
        else:
            raise NotImplementedError
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
            # init your tracker here
                tracker.init(img,tuple(gt_bbox))
                regions.append(gt_bbox)
                location=gt_bbox
            else:
                bbox=tracker.update(img)
                regions.append(bbox)
                location=bbox
            if args.visualization and idx >= 0:  # visualization (skip lost frame)
                im_show = img.copy()
                if idx == 0: cv2.destroyAllWindows()

                cv2.rectangle(im_show, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                      (0, 255, 0), 3)

                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
                cv2.putText(im_show, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow(video.name, im_show)
                cv2.waitKey(1)

        name=tracker_type


        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video.name))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x]) + '\n')


def main():
    global args, logger, v_id
    args = parser.parse_args()
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)
    trackers = ['STRCF']
    dataset = DatasetFactory.create_dataset(name='OTB100',
                                            dataset_root='../dataset/OTB100',
                                            load_img=True
                                            )

    for tracker_type in trackers:
        print(tracker_type)
        track_otb(tracker_type,dataset)

if __name__ == '__main__':
    main()
