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
from cftracker.kcf import KCF
from cftracker.csk import CSK
from cftracker.cn import CN
from cftracker.dat import DAT
from cftracker.eco import ECO

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
            tracker=DSST()
        elif tracker_type=='Staple':
            tracker=Staple()
        elif tracker_type=='KCF':
            tracker=KCF(features='hog_uoip',kernel='gaussian')
        elif tracker_type=='DCF':
            tracker=KCF(features='hog_uoip',kernel='linear')
        elif tracker_type=='DAT':
            tracker=DAT()
        elif tracker_type=='ECO-HC':
            tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif tracker_type=='ECO':
            tracker=ECO(config=otb_deep_config.OTBDeepConfig())
        else:
            raise NotImplementedError

        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
            # init your tracker here
                tracker.init(img,gt_bbox)
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
    trackers=['ECO']
    dataset = DatasetFactory.create_dataset(name='OTB100',
                                            dataset_root='../dataset/OTB100',
                                            load_img=True
                                            )

    for tracker_type in trackers:
        track_otb(tracker_type,dataset)

if __name__ == '__main__':
    main()
