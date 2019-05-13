import argparse
import glob
from os.path import join, realpath, dirname

from tqdm import tqdm
from multiprocessing import Pool
from lib.pysot.datasets import OTBDataset
from lib.pysot.evaluation import OPEBenchmark
from lib.pysot.visualization import draw_success_precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOT Evaluation')
    parser.add_argument('--dataset', type=str, default='OTB50',help='dataset name')
    parser.add_argument('--result_dir', type=str, default='test/OTB100',help='tracker result root')
    parser.add_argument('--tracker_prefix', type=str,default='test', help='tracker prefix')
    parser.add_argument('--show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=10)
    parser.add_argument('--vis',type=bool,default=True)
    args = parser.parse_args()

    root = join(realpath(dirname(__file__)), '../dataset/OTB100')
    tracker_dir = args.result_dir
    trackers = glob.glob(join(tracker_dir, args.tracker_prefix+'*'))
    trackers = [t.split('/')[-1] for t in trackers]
    trackers=['MCCTH-Staple','MKCFup-LP','MKCFup','CSRDCF-LP','DSST-LP','LDES','SAMF','Staple-CA','OPENCV-CSRDCF','DCF','MOSSE','KCF','CSK','Staple','DSST','CN','DAT','ECO-HC','ECO','BACF','CSRDCF']
    print(trackers)
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
