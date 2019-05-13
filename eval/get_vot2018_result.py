from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from os import makedirs
from os.path import join, isdir

from lib.log_helper import init_log, add_file_handler
from lib.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from lib.benchmark_helper import load_dataset


from lib.pysot.utils import region
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

from lib.eco.config import vot18_deep_config,vot18_hc_config
from cftracker.config import ldes_config,dsst_config,csrdcf_config,staple_config,mkcf_up_config,mccth_staple_config

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--dataset', dest='dataset', default='VOT2018',
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test_vot2018.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result',default=True)
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')

def create_tracker(tracker_type):
    if tracker_type == 'MOSSE':
        tracker = MOSSE()
    elif tracker_type == 'CSK':
        tracker = CSK()
    elif tracker_type == 'CN':
        tracker = CN()
    elif tracker_type == 'DSST':
        tracker = DSST(dsst_config.DSSTConfig())
    elif tracker_type=='SAMF':
        tracker=SAMF()
    elif tracker_type == 'Staple':
        tracker = Staple(config=staple_config.StapleVOTConfig())
    #elif tracker_type=='Staple-CA':
    #    tracker=Staple(config=staple_config.StapleCAVOTConfig())
    elif tracker_type == 'KCF':
        tracker = KCF(features='hog', kernel='gaussian')
    elif tracker_type == 'DCF':
        tracker = KCF(features='hog', kernel='linear')
    elif tracker_type == 'DAT':
        tracker = DAT()
    elif tracker_type=='ECO-HC':
        tracker=ECO(config=vot18_hc_config.VOT18HCConfig())
    elif tracker_type=='ECO':
        tracker=ECO(config=vot18_deep_config.VOT18DeepConfig())
    elif tracker_type=='BACF':
        tracker=BACF()
    elif tracker_type=='CSRDCF':
        tracker=CSRDCF(csrdcf_config.CSRDCFConfig())
    elif tracker_type=='CSRDCF-LP':
        tracker=CSRDCF(csrdcf_config.CSRDCFLPConfig())
    elif tracker_type=='OPENCV_KCF':
        tracker=OpenCVCFTracker(name='KCF')
    elif tracker_type=='OPENCV_MOSSE':
        tracker=OpenCVCFTracker(name='MOSSE')
    elif tracker_type=='OPENCV-CSRDCF':
        tracker=OpenCVCFTracker(name='CSRDCF')
    elif tracker_type=='LDES':
        tracker=LDES(config=ldes_config.LDESVOTLinearConfig())
    elif tracker_type=='LDES-NoBGD':
        tracker=LDES(config=ldes_config.LDESVOTNoBGDLinearConfig())
    elif tracker_type=='DSST-LP':
        tracker=DSST(dsst_config.DSSTLPConfig())
    elif tracker_type=='MKCFup':
        tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
    elif tracker_type=='MKCFup-LP':
        tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
    elif tracker_type=='STRCF':
        tracker=STRCF()
    elif tracker_type=='MCCTH-Staple':
        tracker=MCCTHStaple(config=mccth_staple_config.MCCTHVOTConfig())
    else:
        raise NotImplementedError
    return tracker

def track_vot(tracker_type, video):

    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            tracker=create_tracker(tracker_type)
            if tracker_type=='LDES':
                tracker.init(im,gt[f])
                location=gt[f]
            else:
                cx, cy, w, h = get_axis_aligned_bbox(gt[f])
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                location=cxy_wh_2_rect(target_pos,target_sz)
                tracker.init(im,((cx-w/2),(cy-h/2),(w),(h)))
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:
            location=tracker.update(im)

            if 'VOT' in args.dataset:
                b_overlap = region.vot_overlap(gt[f],location, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1
            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    name = tracker_type


    video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
            fin.write(','.join([region.vot_float2str("%.4f", i) for i in x]) + '\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} Tracker:{}'.format(
        v_id, video['name'], toc, f / toc, lost_times,tracker_type))

    return lost_times, f / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    # setup dataset
    dataset = load_dataset(args.dataset)


    total_lost = 0  # VOT
    speed_list = []

    trackers = ['STRCF']

    for tracker_type in trackers:

        for v_id, video in enumerate(dataset.keys(), start=1):
            lost, speed = track_vot(tracker_type,dataset[video])
            total_lost += lost
            speed_list.append(speed)

        logger.info('Total Lost: {:d}'.format(total_lost))

        logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
