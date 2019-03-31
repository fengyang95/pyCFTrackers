class BaseCF:
    def __init__(self):
        raise NotImplementedError

    def init(self,first_frame,bbox):
        raise NotImplementedError

    def update(self,current_frame):
        raise NotImplementedError


