"""
StrangerZhang's pyECO for comparing performance of different tracker.
@link{https://github.com/StrangerZhang/pyECO}
"""

import numpy as np
from .base import BaseCF
from lib.eco.tracker import ECOTracker
import cv2

class ECO(BaseCF):
    def __init__(self,config):
        super(ECO).__init__()
        self.tracker=ECOTracker(is_color=True,config=config)


    def init(self,first_frame,bbox):

        first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = bbox[0],bbox[1], bbox[0]+bbox[2],bbox[1]+bbox[3]
        new_bbox = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
        self.tracker.init(first_frame,new_bbox)


    def update(self,current_frame,vis=False):
        current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2RGB)
        bbox=self.tracker.update(current_frame,train=True,vis=vis)
        if vis is True:
            self.score=self.tracker.score
            self.crop_size = tuple(self.tracker.crop_size.astype(np.int64))
        x1, y1, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = [int(x1), int(y1), int(w), int(h)]
        return pos



