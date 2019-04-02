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
from cftracker.kcf import KCF
from cftracker.csk import CSK
from cftracker.cn import CN
from cftracker.dat import DAT
from cftracker.eco import ECO
from lib.eco.config import vot18_deep_config,vot_hc_config


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
        tracker = DSST()
    elif tracker_type == 'Staple':
        tracker = Staple()
    elif tracker_type == 'KCF':
        tracker = KCF(features='hog_uoip', kernel='gaussian')
    elif tracker_type == 'DCF':
        tracker = KCF(features='hog_uoip', kernel='linear')
    elif tracker_type == 'DAT':
        tracker = DAT()
    elif tracker_type=='ECO-HC':
        tracker=ECO(config=vot_hc_config.VOTHCConfig())
    elif tracker_type=='ECO':
        tracker=ECO(config=vot18_deep_config.VOT18DeepConfig())
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
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            location=cxy_wh_2_rect(target_pos,target_sz)
            tracker.init(im,(int(cx-w/2),int(cy-h/2),int(w),int(h)))
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:
            bbox=tracker.update(im)
            x,y,w,h=bbox
            location = cxy_wh_2_rect((x+w/2,y+h/2),(w,h))

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

    trackers = ['ECO']

    for tracker_type in trackers:

        for v_id, video in enumerate(dataset.keys(), start=1):
            lost, speed = track_vot(tracker_type,dataset[video])
            total_lost += lost
            speed_list.append(speed)

        logger.info('Total Lost: {:d}'.format(total_lost))

        logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
