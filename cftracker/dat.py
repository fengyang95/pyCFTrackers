"""
Python re-implementation of "In Defense of Color-based Model-free Tracking"
@inproceedings{Possegger2015In,
  title={In Defense of Color-based Model-free Tracking},
  author={Possegger, Horst and Mauthner, Thomas and Bischof, Horst},
  booktitle={Computer Vision & Pattern Recognition},
  year={2015},
}
"""
import numpy as np
import cv2
from cftracker.base import BaseCF
from lib.utils import cos_window
import copy
from cftracker.config.dat_config import DATConfig

class DAT(BaseCF):
    def __init__(self):
        super(DAT).__init__()
        self.config=DATConfig()
        self.target_pos_history=[]
        self.target_sz_history=[]

    def init(self,first_frame,bbox):
        bbox=np.array(bbox).astype(np.int64)
        x,y,w,h=tuple(bbox)
        self._scale_factor=min(1,round(10*self.config.img_scale_target_diagonal/cv2.norm(np.array([w,h])))/10.)
        self._center=(self._scale_factor*(x+(w-1)/2),self._scale_factor*(y+(h-1)/2))
        self.w,self.h=int(w*self._scale_factor),int(h*self._scale_factor)
        self._target_sz=(self.w,self.h)

        img=cv2.resize(first_frame,None,fx=self._scale_factor,fy=self._scale_factor)
        if self.config.color_space=='lab':
            img=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
        elif self.config.color_space=='hsv':
            img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0] * 256 / 180)
            img = img.astype(np.uint8)
        else:
            pass

        surr_sz=(int(np.floor(self.config.surr_win_factor*self.w)),int(np.floor(self.config.surr_win_factor*self.h)))
        surr_rect=pos2rect(self._center,surr_sz,(img.shape[1],img.shape[0]))
        obj_rect_surr=pos2rect(self._center,self._target_sz,(img.shape[1],img.shape[0]))
        obj_rect_surr=(obj_rect_surr[0]-surr_rect[0],
                    obj_rect_surr[1]-surr_rect[1],
                    obj_rect_surr[2],obj_rect_surr[3])
        surr_win=get_sub_window(img,self._center,surr_sz)
        self.bin_mapping=get_bin_mapping(self.config.num_bins)
        self.prob_lut_,prob_map=get_foreground_background_probs(surr_win,obj_rect_surr,
                                self.config.num_bins,self.bin_mapping)
        self._prob_lut_distractor=copy.deepcopy(self.prob_lut_)
        self._prob_lut_masked=copy.deepcopy(self.prob_lut_)
        self.adaptive_threshold_=get_adaptive_threshold(prob_map,obj_rect_surr)
        self.target_pos_history.append((self._center[0]/self._scale_factor,self._center[1]/self._scale_factor))
        self.target_sz_history.append((self._target_sz[0]/self._scale_factor,self._target_sz[1]/self._scale_factor))

    def update(self,current_frame,vis=False):
        img_preprocessed=cv2.resize(current_frame,None,fx=self._scale_factor,fy=self._scale_factor)
        if self.config.color_space=='lab':
            img=cv2.cvtColor(img_preprocessed,cv2.COLOR_BGR2Lab)
        elif self.config.color_space=='hsv':
            img=cv2.cvtColor(img_preprocessed,cv2.COLOR_BGR2HSV)
            img[:,:,0]=(img[:,:,0]*256/180)
            img=img.astype(np.uint8)
        else:
            img=img_preprocessed
        prev_pos=self.target_pos_history[-1]
        prev_sz=self.target_sz_history[-1]
        if self.config.motion_estimation_history_size>0:
            prev_pos=prev_pos+get_motion_prediciton(self.target_pos_history,self.config.motion_estimation_history_size)
        target_pos=(prev_pos[0]*self._scale_factor,prev_pos[1]*self._scale_factor)
        target_sz=(prev_sz[0]*self._scale_factor,prev_sz[1]*self._scale_factor)
        search_sz_w=int(np.floor(target_sz[0]+self.config.search_win_padding*max(target_sz[0],target_sz[1])))
        search_sz_h=int(np.floor(target_sz[1]+self.config.search_win_padding*max(target_sz[0],target_sz[1])))
        search_sz=(search_sz_w,search_sz_h)
        search_rect=pos2rect(target_pos,search_sz)
        self.crop_size=(search_rect[2],search_rect[3])
        search_win,padded_search_win=get_subwindow_masked(img,target_pos,search_sz)
        #Apply probability LUT
        pm_search=get_foreground_prob(search_win,self.prob_lut_,self.bin_mapping)
        if self.config.distractor_aware is True:
            pm_search_dist=get_foreground_prob(search_win,self._prob_lut_distractor,self.bin_mapping)
            pm_search=(pm_search+pm_search_dist)/2
        pm_search=pm_search*padded_search_win
        window=cos_window(search_sz)
        hypotheses,vote_scores,dist_scores=get_nms_rects(pm_search,target_sz,self.config.nms_scale,
                                                         self.config.nms_overlap,self.config.nms_score_factor,
                                                         window,self.config.nms_include_center_vote)

        candidate_centers=[]
        candidate_scores=[]
        for i in range(len(hypotheses)):
            candidate_centers.append((hypotheses[i][0]+hypotheses[i][2]/2,
                                      hypotheses[i][1]+hypotheses[i][3]/2))
            candidate_scores.append(vote_scores[i]*dist_scores[i])
        best_candidate=np.argmax(np.array(candidate_scores))
        target_pos=candidate_centers[best_candidate]

        distractors=[]
        distractor_overlap=[]
        if len(hypotheses)>1:
            target_rect=pos2rect(target_pos,target_sz,(pm_search.shape[1],pm_search.shape[0]))
            for i in range(len(hypotheses)):
                if i!=best_candidate:
                    distractors.append(hypotheses[i])
                    distractor_overlap.append(cal_iou(target_rect,distractors[-1]))
        if vis:
            self.score=pm_search


        target_pos_img=(target_pos[0]+search_rect[0],target_pos[1]+search_rect[1])
        if self.config.prob_lut_update_rate>0:
            surr_sz=(int(self.config.surr_win_factor*target_sz[0]),int(self.config.surr_win_factor*target_sz[1]))
            surr_rect=pos2rect(target_pos_img,surr_sz,(img.shape[1],img.shape[0]))
            obj_rect_surr=pos2rect(target_pos_img,target_sz,(img.shape[1],img.shape[0]))
            obj_rect_surr=(obj_rect_surr[0]-surr_rect[0],obj_rect_surr[1]-surr_rect[1],obj_rect_surr[2],obj_rect_surr[3])
            surr_win=get_sub_window(img,target_pos_img,surr_sz)
            prob_lut_bg,_=get_foreground_background_probs(surr_win,obj_rect_surr,self.config.num_bins)

            if self.config.distractor_aware is True:
                if len(distractors)>1:
                    obj_rect=pos2rect(target_pos,target_sz,(search_win.shape[1],search_win.shape[0]))
                    prob_lut_dist=get_foreground_distractor_probs(search_win,obj_rect,distractors,self.config.num_bins)
                    self._prob_lut_distractor=(1-self.config.prob_lut_update_rate)*self._prob_lut_distractor+\
                        self.config.prob_lut_update_rate*prob_lut_dist
                else:
                    self._prob_lut_distractor=(1-self.config.prob_lut_update_rate)*self._prob_lut_distractor+\
                        self.config.prob_lut_update_rate*prob_lut_bg
                if len(distractors)==0 or np.max(distractor_overlap)<0.1:
                    self.prob_lut_=(1-self.config.prob_lut_update_rate)*self.prob_lut_+self.config.prob_lut_update_rate*prob_lut_bg
                prob_map=get_foreground_prob(surr_win,self.prob_lut_,self.bin_mapping)
                dist_map=get_foreground_prob(surr_win,self._prob_lut_distractor,self.bin_mapping)
                prob_map=0.5*prob_map+0.5*dist_map
            else:
                self.prob_lut_=(1-self.config.prob_lut_update_rate)*self.prob_lut_+self.config.prob_lut_update_rate*prob_lut_bg
                prob_map=get_foreground_prob(surr_win,self.prob_lut_,self.bin_mapping)
            self.adaptive_threshold_=get_adaptive_threshold(prob_map,obj_rect_surr)

        target_pos=(target_pos[0]+search_rect[0],target_pos[1]+search_rect[1])
        target_pos_original=(target_pos[0]/self._scale_factor,target_pos[1]/self._scale_factor)
        target_sz_original=(target_sz[0]/self._scale_factor,target_sz[1]/self._scale_factor)
        self.target_pos_history.append(target_pos_original)
        self.target_sz_history.append(target_sz_original)
        self._scale_factor=min(1,round(10*self.config.img_scale_target_diagonal/cv2.norm(target_sz_original))/10)
        return [target_pos_original[0]-target_sz_original[0]/2,target_pos_original[1]-target_sz_original[1]/2,
                target_sz_original[0],target_sz_original[1]]


def pos2rect(center,obj_sz,win_sz=None):
    obj_w,obj_h=obj_sz
    cx,cy=center
    rect=(int(round(cx-obj_w/2)),int(round(cy-obj_h/2)),obj_w,obj_h)
    if win_sz is not None:
        border=(0,0,win_sz[0]-1,win_sz[1]-1)
        rect=intersect_of_rects(border,rect)
    return rect

def get_foreground_background_probs(frame,obj_rect,num_bins,bin_mapping=None):
    frame=frame.astype(np.uint8)

    surr_hist = cv2.calcHist([frame],[0, 1, 2], None, [num_bins,num_bins,num_bins],
                             [0, 256, 0, 256, 0, 256])
    x,y,w,h=obj_rect
    if x+w>frame.shape[1]-1:
        w=(frame.shape[1]-1)-x
    if y+h>frame.shape[0]-1:
        h=(frame.shape[0]-1)-y
    x=int(max(x,0))
    y=int(max(y,0))
    obj_win=frame[y:y+h+1,x:x+w+1]

    obj_hist = cv2.calcHist([obj_win], [0, 1, 2], None, [num_bins, num_bins, num_bins], [0, 256, 0, 256, 0, 256])
    prob_lut = (obj_hist + 1) / (surr_hist + 2)
    prob_map = None
    if bin_mapping is not None:
        frame_bin = cv2.LUT(frame, bin_mapping).astype(np.int64)
        prob_map = prob_lut[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]

    return prob_lut,prob_map

def get_bin_mapping(num_bins):
    bin_mapping=np.zeros((256,))
    for i in range(bin_mapping.shape[0]):
        bin_mapping[i]=(np.floor(i/(256/num_bins)))
    return bin_mapping.astype(np.uint8)

def get_adaptive_threshold(prob_map, obj_rect, config=DATConfig()):
    x,y,w,h=obj_rect
    w+=1
    w=int(min(prob_map.shape[1]-x,w))
    h+=1
    h=int(min(prob_map.shape[0]-y,h))
    obj_prob_map=prob_map[y:y+h,x:x+w]
    bins=21
    H_obj=cv2.calcHist([obj_prob_map],[0],None,[bins],[-0.025,1.025],accumulate=False)
    H_obj=H_obj/np.sum(H_obj)
    cum_H_obj=copy.deepcopy(H_obj)
    for i in range(1,cum_H_obj.shape[0]):
        cum_H_obj[i,0]+=cum_H_obj[i-1,0]
    H_dist=cv2.calcHist([prob_map],[0],None,[bins],[-0.025,1.025],accumulate=False)
    H_dist=H_dist-H_obj
    H_dist=H_dist/np.sum(H_dist)
    cum_H_dist=copy.deepcopy(H_dist)
    for i in range(1,cum_H_dist.shape[0]):
        cum_H_dist[i,0]+=cum_H_dist[i-1,0]
    k=np.zeros_like(cum_H_obj)
    for i in range(k.shape[0]-1):
        k[i,0]=cum_H_obj[i+1,0]-cum_H_obj[i,0]
    # not sure

    x=np.abs(cum_H_obj-(1-cum_H_dist))+(cum_H_obj<(1-cum_H_dist))+(1-k)
    i=np.argmin(x)
    #print(i)
    threshold=max(0.4,min(0.7,config.adapt_thresh_prob_bins[i]))
    return threshold

def get_motion_prediciton(points,max_num_frames):
    predx,predy=0,0
    if len(points)>=3:
        max_num_frames=max_num_frames+2
        A1=0.8
        A2=-1
        V=copy.deepcopy(points[int(max(0,len(points)-max_num_frames)):len(points)])
        Px=[]
        Py=[]
        for i in range(2,len(V)):
            x=A1*(V[i][0]-V[i-2][0])+A2*(V[i-1][0]-V[i-2][0])
            y=A1*(V[i][1]-V[i-2][1]+A2*(V[i-1][1]-V[i-2][1]))
            Px.append(x)
            Py.append(y)
        predx=sum(Px)/len(Px)
        predy=sum(Py)/len(Py)
    return (predx,predy)

def get_subwindow_masked(img,pos,sz):
    tl=(int(np.floor(pos[0])+1-np.floor(sz[0]/2)),int(np.floor(pos[1])+1-np.floor(sz[1]/2)))
    out=get_sub_window(img,pos,sz)
    bbox=(tl[0],tl[1],sz[0],sz[1])
    bbox2=(0,0,img.shape[1]-1,img.shape[0]-1)
    bbox=intersect_of_rects(bbox,bbox2)
    bbox=(bbox[0]-tl[0],bbox[1]-tl[1],bbox[2],bbox[3])
    mask=np.zeros((sz[1],sz[0]),dtype=np.uint8)
    mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]=1
    return out,mask



def intersect_of_rects(rect1,rect2):
    tl = (max(rect1[0], rect2[0]), max(rect1[1], rect2[1]))
    br = (min(rect1[0] + rect1[2], rect2[0] + rect2[2]), min(rect1[1] + rect1[3], rect2[1] + rect2[3]))
    inter = (int(tl[0]), int(tl[1]), int(br[0] - tl[0]), int(br[1] - tl[1]))
    return inter

def cal_iou(rect1,rect2):
    inter=intersect_of_rects(rect1,rect2)
    iou=(inter[2]*inter[3])/(rect1[2]*rect1[3]+rect2[2]*rect2[3]-inter[2]*inter[3])
    return iou

def get_foreground_prob(frame,prob_lut,bin_mapping):
    frame_bin=cv2.LUT(frame,bin_mapping).astype(np.int64)
    prob_map = prob_lut[frame_bin[:, :, 0], frame_bin[:, :, 1], frame_bin[:, :, 2]]
    return prob_map

def get_nms_rects(prob_map,obj_sz,scale,overlap,score_frac,dist_map,include_inner):

    height,width=prob_map.shape[:2]
    rect_sz=(int(np.floor(obj_sz[0]*scale)),int(np.floor(obj_sz[1]*scale)))
    o_x,o_y=0,0
    if include_inner is True:
        o_x=int(np.round(max(1,rect_sz[0]*0.2)))
        o_y=int(np.round(max(1,rect_sz[1]*0.2)))
    stepx=int(max(1,int(np.round(rect_sz[0]*(1-overlap)))))
    stepy=int(max(1,int(np.round(rect_sz[1]*(1-overlap)))))
    posx,posy=[],[]
    for i in range(0,width-rect_sz[0],stepx):
        posx.append(i)
    for i in range(0,height-rect_sz[1],stepy):
        posy.append(i)
    posx,posy=np.arange(0,width-rect_sz[0],stepx),np.arange(0,height-rect_sz[1],stepy)
    x,y=np.meshgrid(posx,posy)
    r=x.flatten()+rect_sz[0]
    b=y.flatten()+rect_sz[1]
    r[r>width-1]=width-1
    b[b>height-1]=height-1
    boxes=[x.flatten(),y.flatten(),r-x.flatten(),b-y.flatten()]
    boxes=np.array(boxes).T
    if include_inner is True:
        boxes_inner=[x.flatten()+o_x,y.flatten()+o_y,(r-2*o_x)-x.flatten(),(b-2*o_y)-y.flatten()]
        boxes_inner=np.array(boxes_inner).T

    bl=np.array([b,x.flatten()]).T
    br=np.array([b,r]).T
    tl=np.array([y.flatten(),x.flatten()]).T
    tr=np.array([y.flatten(),r]).T

    if include_inner is True:
        rect_sz_inner=(rect_sz[0]-2*o_x,rect_sz[1]-2*o_y)
    bl_inner=np.array([b-o_y,x.flatten()+o_x]).T
    br_inner=np.array([b-o_y,r-o_x]).T
    tl_inner=np.array([y.flatten()+o_y,x.flatten()+o_x]).T
    tr_inner=np.array([y.flatten()+o_y,r-o_x]).T
    int_prob_map=cv2.integral(prob_map)
    int_dist_map=cv2.integral(dist_map)
    v_scores=int_prob_map[br[:,0],br[:,1]]-int_prob_map[bl[:,0],bl[:,1]]-int_prob_map[tr[:,0],tr[:,1]]+int_prob_map[tl[:,0],tl[:,1]]
    d_scores=int_dist_map[br[:,0],br[:,1]]-int_dist_map[bl[:,0],bl[:,1]]-int_dist_map[tr[:,0],tr[:,1]]+int_dist_map[tl[:,0],tl[:,1]]
    if include_inner is True:
        scores_inner = int_prob_map[br_inner[:,0],br_inner[:,1]] - int_prob_map[bl_inner[:,0],bl_inner[:,1]] -\
                       int_prob_map[tr_inner[:,0],tr_inner[:,1]] + int_prob_map[tl_inner[:,0],tl_inner[:,1]]
        v_scores=v_scores/(rect_sz[0]*rect_sz[1])+scores_inner/(rect_sz_inner[0]*rect_sz_inner[1])

    top_rects = []
    top_vote_scores = []
    top_dist_scores = []
    midx=np.argmax(v_scores)
    ms=v_scores[midx]
    best_score=ms

    while ms>score_frac*best_score:
        box_mid=tuple(boxes[midx])
        prob_map[box_mid[1]:box_mid[1]+box_mid[3],box_mid[0]:box_mid[0]+box_mid[2]]=0
        top_rects.append(tuple(boxes[midx]))
        top_vote_scores.append(v_scores[midx])
        top_dist_scores.append(d_scores[midx])
        boxes=np.delete(boxes,midx,axis=0)
        if include_inner is True:
            boxes_inner=np.delete(boxes_inner,midx,axis=0)
        bl=np.delete(bl,midx,axis=0)
        br=np.delete(br,midx,axis=0)
        tl=np.delete(tl,midx,axis=0)
        tr=np.delete(tr,midx,axis=0)

        if include_inner is True:
            bl_inner = np.delete(bl_inner, midx, axis=0)
            br_inner = np.delete(br_inner, midx, axis=0)
            tl_inner = np.delete(tl_inner, midx, axis=0)
            tr_inner = np.delete(tr_inner, midx, axis=0)

        int_prob_map=cv2.integral(prob_map)
        int_dist_map=cv2.integral(dist_map)
        v_scores = int_prob_map[br[:, 0], br[:, 1]] - int_prob_map[bl[:, 0], bl[:, 1]] - int_prob_map[
            tr[:, 0], tr[:, 1]] + int_prob_map[tl[:, 0], tl[:, 1]]
        d_scores = int_dist_map[br[:, 0], br[:, 1]] - int_dist_map[bl[:, 0], bl[:, 1]] - int_dist_map[
            tr[:, 0], tr[:, 1]] + int_dist_map[tl[:, 0], tl[:, 1]]
        if include_inner is True:
            scores_inner = int_prob_map[br_inner[:, 0], br_inner[:, 1]] - int_prob_map[bl_inner[:, 0], bl_inner[:, 1]] - \
                           int_prob_map[tr_inner[:, 0], tr_inner[:, 1]] + int_prob_map[tl_inner[:, 0], tl_inner[:, 1]]
            v_scores = v_scores / (rect_sz[0] * rect_sz[1]) + scores_inner / (rect_sz_inner[0] * rect_sz_inner[1])

        midx=np.argmax(v_scores)
        ms=v_scores[midx]

    return top_rects,top_vote_scores,top_dist_scores

def get_foreground_distractor_probs(frame,obj_rect,distractors,num_bins):
    Md=np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
    Mo=np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
    for i in range(len(distractors)):
        x,y,w,h=distractors[i]
        Md[y:y+h,x:x+w]=1
    Mo[obj_rect[1]:obj_rect[1]+obj_rect[3],obj_rect[0]:obj_rect[0]+obj_rect[2]]=1
    obj_hist=cv2.calcHist([frame],[0,1,2],Mo,[num_bins,num_bins,num_bins],[0,256,0,256,0,256])
    dist_hist=cv2.calcHist([frame],[0,1,2],Md,[num_bins,num_bins,num_bins],[0,256,0,256,0,256])
    prob_lut=(obj_hist*len(distractors)+1)/(dist_hist+obj_hist*len(distractors)+2)
    return prob_lut

def get_sub_window(frame,center,sz):
    h,w=frame.shape[:2]
    lt=(int(min(w-1,max(-sz[0]+1,center[0]-np.floor(sz[0]/2+1)))),
        int(min(h-1,max(-sz[1]+1,center[1]-np.floor(sz[1]/2+1)))))
    rb=(lt[0]+sz[0]-1,lt[1]+sz[1]-1)
    border=(-min(0,lt[0]),-min(lt[1],0),
            max(rb[0]-w+1,0),max(rb[1]-h+1,0))
    lt_limit=(max(lt[0],0),max(lt[1],0))
    rb_limit=(min(rb[0]+1,w),min(rb[1]+1,h))
    sub_window=frame[lt_limit[1]:rb_limit[1],lt_limit[0]:rb_limit[0]]
    if border!=(0,0,0,0):
        sub_window=cv2.copyMakeBorder(sub_window,border[1],border[3],border[0],border[2],cv2.BORDER_REPLICATE)
    return sub_window

