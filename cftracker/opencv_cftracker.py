"""
warpper of cf trackers implemented in opencv
"""
from .base import BaseCF
import cv2


class OpenCVCFTracker(BaseCF):
    def __init__(self, name):
        super(OpenCVCFTracker).__init__()
        self.name = name

    def init(self, first_frame, bbox):
        if self.name == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        elif self.name == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        elif self.name == 'CSRDCF':
            self.tracker = cv2.TrackerCSRT_create()
        else:
            raise NotImplementedError
        self.tracker.init(first_frame, bbox)

    def update(self, current_frame, vis=False):
        _, bbox = self.tracker.update(current_frame)
        x1, y1, w, h = bbox
        pos = [x1, y1, w, h]
        return pos
