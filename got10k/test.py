from __future__ import absolute_import

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from got10k.experiments import *

from got10k.trackers.bacf import BACF


if __name__ == '__main__':

    #tracker = CSK()
    tracker = BACF()
    #tracker = KCF(features='hog', kernel='linear') #DCF-HOG

    # setup experiments
    experiments = [
        #ExperimentGOT10k('data/GOT-10k', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        ExperimentOTB('../dataset/OTB', version=2015), # 数据集路径 ln -s ./data /path/to/your/data/folder
        #ExperimentVOT('data/vot2018', version=2018),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('data/Temple-color-128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
