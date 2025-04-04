"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np

from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)

def cal_score_dif_batch(bboxes1, bboxes2):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bboxes2 = np.expand_dims(bboxes2, 0)
  bboxes1 = np.expand_dims(bboxes1, 1)

  score2 = bboxes2[..., 4]
  score1 = bboxes1[..., 4]

  return (abs(score2 - score1))


def convert_bbox_to_z(bbox, pre=False):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w / 2.
  y = bbox[1] + h / 2.
  if pre:
    return np.array([x, y, w, h]).reshape((4, 1))
  else:
    return np.array([x, y, w, h, 0, 0, 0, 0]).reshape((8, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  x1 = x[0] - x[2] / 2
  y1 = x[1] - x[3] / 2
  x2 = x[0] + x[2] / 2
  y2 = x[1] + x[3] / 2
  if (score == None):
    # if predict==True:
    return np.array([x1, y1, x2, y2]).reshape((1, 4))
    # else:
    #   return np.array([x1, y1, x2, y2, x[4], x[5], x[6], x[7]]).reshape((1,8))
  else:
    return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,det_thresh):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    # self.kf = KalmanFilter(dim_x=7, dim_z=4)
    # self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    # self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    #
    # self.kf_score = KalmanFilter(dim_x=2, dim_z=1)
    # self.kf_score.F = np.array([[1, 1],
    #                             [0, 1]])
    # self.kf_score.H = np.array([[1, 0]])
    #
    # self.kf.R[2:,2:] *= 10.
    # self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10.
    # self.kf.Q[-1,-1] *= 0.01
    # self.kf.Q[4:,4:] *= 0.01
    # self.kf.x[:4] = convert_bbox_to_z(bbox)
    #
    # self.kf_score.R[0:, 0:] *= 10.
    # self.kf_score.P[1:, 1:] *= 1000.  # give high uncertainty to the unobservable initial velocities 对不可观测的初始速度给予高度不确定性
    # self.kf_score.P *= 10.
    # self.kf_score.Q[-1, -1] *= 0.01
    # self.kf_score.Q[1:, 1:] *= 0.01
    # self.kf_score.x[:1] = bbox[-1]
    self.score = bbox[-1]
    self.score_pre = None
    self.x = convert_bbox_to_z(bbox, pre=False)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.det_thresh = det_thresh

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.x = convert_bbox_to_z(bbox, pre=True)
    self.score_pre = self.score
    if self.score < bbox[-1]:
      self.score = bbox[-1]


  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if((self.kf.x[6]+self.kf.x[2])<=0):
    #   self.kf.x[6] *= 0.0
    # self.kf.predict()
    # self.kf_score.predict()

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.x))
    #return self.history[-1], np.clip(self.score, self.det_thresh, 1.0)
    if not self.score_pre:
      return self.history[-1], np.clip(self.score, self.det_thresh, 1.0)
    else:
      return self.history[-1],  np.clip( self.score_pre*0.5 + self.score*0.5, self.det_thresh, 1.0)

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.x)


def associate_detections_to_trackers(detections, trackers, args, iou_threshold=0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      if args.TCM_first_step:
        cost_matrix = iou_matrix - cal_score_dif_batch(detections, trackers) * args.TCM_first_step_weight
        matched_indices = linear_assignment(-cost_matrix)
      else:
        matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort_star_score(object):
  def __init__(self, args, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.det_thresh = det_thresh
    self.args = args
    self.ret_st = []
    self.trackers_storage = []

  def update(self, output_results, img_info, img_size, net):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    self.net = net

    # post_process detections
    output_results = output_results.cpu().numpy()
    scores = output_results[:, 4] * output_results[:, 5]
    bboxes = output_results[:, :4]  # x1y1x2y2
    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale
    dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
    remain_inds = scores > self.det_thresh
    dets = dets[remain_inds]
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    ids_del = []
    ids_add = []
    temp = []
    trackers_st = []
    if self.frame_count > 1:
      predict, score_pre = net.forward(self.trackers_storage.copy())

    for t, trk in enumerate(trks):
      #pos, trk_score = self.trackers[t].predict()
      self.trackers[t].x = np.array(predict[t]).reshape((4, 1))
      if self.trackers[t].score > score_pre[t][0]:
        self.trackers[t].score = self.trackers[t].score*0.8 + score_pre[t][0]*0.2
      #self.trackers[t].score = score_pre[t][0]
      pos, trk_score = self.trackers[t].predict()

      trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], trk_score]
      #trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], trk_score]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.args, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], self.det_thresh)
        if self.frame_count > 1:
          ids_add.append(trk.id)
        self.trackers.append(trk)
    i = len(self.trackers)

    for trk in reversed(self.trackers):
      # d = trk.get_state()
      # d = d[0]
      d = trk.get_state()[0]
      # d = convert_x_to_bbox(trk)
      if (trk.time_since_update < 1) and (
              trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # +1 as MOT benchmark requires positive
        ret.append(np.concatenate((d,[trk.id])).reshape(1, -1))                     # add score
      if trk.id in ids_add:
        temp.append(np.concatenate((trk.x[:, 0],[trk.score], [trk.id])).reshape(1, -1))        # add score
      # else:
      #   ids_del.append(trk.id)
      # remove dead tracklet
      i -= 1
      if (trk.time_since_update > self.max_age):
        self.trackers.pop(i)
        ids_del.append(trk.id)
      # x = trk.x
      # c = trk.x[:, 0]
      if self.frame_count > 1:

        trackers_st.append(np.concatenate((trk.x[:4, 0],[trk.score], [trk.id])).reshape(1, -1))      # add score
      else:
        trackers_st.append(np.concatenate((trk.x[:, 0], [trk.score],[trk.id])).reshape(1, -1))          # add score
    if (len(ret) > 0):
      ret = np.concatenate(ret)

      if self.frame_count > 1:
        if len(temp) > 0:
          temp = np.concatenate(temp)
          temp = np.expand_dims(temp, axis=0)
          temp = np.repeat(temp, self.trackers_storage.shape[0], axis=0)
          self.trackers_storage = np.concatenate((self.trackers_storage, temp), axis=1)
        trackers_st = np.concatenate(trackers_st)
        trackers_st = trackers_st[::-1, :]
        # trackers_st = np.expand_dims(trackers_st, axis=0)
        track_offset = trackers_st[:, :4] - self.trackers_storage[-1, :, :4]
        track_score =  np.expand_dims(trackers_st[:, 4], axis=1)                                                  # add score
        track_index = np.expand_dims(trackers_st[:, 5], axis=1)
        trackers_st = np.concatenate((trackers_st[:, :4], track_offset), axis=1)
        trackers_st = np.concatenate((trackers_st, track_score), axis=1)                # add score
        trackers_st = np.concatenate((trackers_st, track_index), axis=1)
        trackers_st = np.expand_dims(trackers_st, axis=0)
        self.trackers_storage = np.concatenate((self.trackers_storage, trackers_st), axis=0)
        if self.trackers_storage.shape[0] > 10:
          self.trackers_storage = np.delete(self.trackers_storage, 0, axis=0)
        i = 0
        while i < self.trackers_storage.shape[1]:
          # if self.trackers_storage[:, i, 4] in ids_del:
          if np.isin(self.trackers_storage[:, i, 9], ids_del).any():                 # 8-> 9
            self.trackers_storage = np.delete(self.trackers_storage, i, axis=1)
          else:
            i = i + 1
      else:
        trackers_st = np.concatenate(trackers_st)
        trackers_st = trackers_st[::-1, :]
        self.trackers_storage = np.expand_dims(trackers_st, axis=0)

      # if self.trackers_storage.ndim == 2:
      #   self.trackers_storage = np.expand_dims(self.trackers_storage, axis=0)
      # trackers_list = self.trackers_storage.tolist()
      # for id in enumerate(ids):
      #     if trackers_list[:, :, 4] == id:

      return ret
    return np.empty((0, 5))