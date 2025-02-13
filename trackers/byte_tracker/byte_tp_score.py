import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

# from .kalman_filter import KalmanFilter
from trackers.byte_tracker import matching
from .basetrack import BaseTrack, TrackState

from .association import *
from collections import deque
from scipy.fftpack import fft, ifft

class LimitedQueue:
    def __init__(self, max_length=48):
        self.queue = deque(maxlen=max_length)

    def add(self, item):
        self.queue.append(item)
        
    def reset(self):
        self.queue.clear()
    
    def to_array(self):
        return np.array(self.queue)
    
    def length(self):
        return len(self.queue)
    
    def __str__(self):
        return str(self.queue)

class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, occluded_val=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        # self.kalman_filter = None
        # self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.pre_score = score
        self.pred_score = score
        self.tracklet_len = 0
        
        self.occluded = occluded_val
        
        ## parameter for TP
        self.obs_seq_len = 8
        self.pred_seq_len = 12
        self.frames_per_set = 6
        
        ## var
        self.tlwh_queue = LimitedQueue(max_length=10)
        self.overlap_len = 0
        self.pred_traj = [[]]
        self.last_tlbr = None
        
#     def predict(self):
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[7] = 0
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

#     @staticmethod
#     def multi_predict(stracks):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#             for i, st in enumerate(stracks):
#                 if st.state != TrackState.Tracked:
#                     multi_mean[i][7] = 0
#             multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov

    def activate(self, frame_id):
        """Start a new tracklet"""
        # self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        #self.pred_score = self.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        #initally defined the tlwh when call STrack
        #if not self.occluded:
        #self.tlwh_queue.add(self._tlwh)
        self.trj = np.array([self._tlwh[0], self._tlwh[1], self._tlwh[2], self._tlwh[3], self.pred_score]).reshape((1, 5))
        #a=self.trj[0]
        self.tlwh_queue.add(self.trj[0])
        
    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        #self.pred_score = new_track.score
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        
        #self.tlwh_queue.reset()
        #if not new_track.occluded:
        self.trj = np.array([self._tlwh[0], self._tlwh[1], self._tlwh[2], self._tlwh[3], self.pred_score]).reshape((1, 5))
        self.tlwh_queue.add(self.trj[0])

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        #self.score = new_track.score
        self.pre_score = self.score
        self.score = new_track.score
        self.pred_score = self.score*0.7 + self.pred_score*0.3
        #if not new_track.occluded:
        self.trj = np.array([self._tlwh[0], self._tlwh[1], self._tlwh[2], self._tlwh[3], self.pred_score]).reshape((1, 5))
        self.tlwh_queue.add(self.trj[0])

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class byte_tp_score(object):
    def __init__(self, net, args):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = args.track_buffer
        self.max_time_lost = self.buffer_size
        # self.kalman_filter = KalmanFilter()
        
        self.net = net
        
        ## parameter for TP
        self.obs_seq_len = 8
        self.pred_seq_len = 12
        self.frames_per_set = 6
        

    def process_tp(self, strack_pool):

        #KSTEPS = self.obs_seq_len + self.pred_seq_len
        track_num = len(strack_pool)
        obj_traj_temp = []
        #for i in pred_index:
        fram_tp = 0
        for i in range(track_num):
            array_representation = strack_pool[i].tlwh_queue.to_array()
            if array_representation.shape[0] > fram_tp:
                fram_tp = array_representation.shape[0]


        for i in range(track_num):

            array_representation = strack_pool[i].tlwh_queue.to_array()
            #strack_pool[i].last_tlbr = array_representation[-1]
            # if i == 0:
            #     fram_tp = array_representation.shape[0]
            if array_representation.shape[0] == fram_tp:

                if array_representation.shape[0] == 1:
                    offset = np.zeros((1,4),dtype=int)
                    tp = np.concatenate((array_representation, offset ), axis=1)
                else:
                    offset_first = np.zeros((1,4),dtype=int)
                    offset_last = array_representation[1:,:4] - array_representation[:-1,:4]
                    offset_all = np.concatenate((offset_first, offset_last ), axis=0)
                    tp = np.concatenate((array_representation, offset_all), axis=1)
            else:
                if array_representation.shape[0]== 1:
                    offset = np.zeros((1, 4), dtype=int)
                    tp = np.concatenate((array_representation, offset), axis=1)
                    tp = np.repeat(tp, ((fram_tp - array_representation.shape[0])+1), axis=0)

                elif array_representation.shape[0] == 2:
                    offset_first = np.zeros((1, 4), dtype=int)

                    tp_first = np.concatenate((np.expand_dims(array_representation[0], axis=0),offset_first), axis=1)

                    offset_last = array_representation[1:,:4] - array_representation[:-1,:4]
                    tp_last = np.concatenate((np.expand_dims(array_representation[1], axis=0), offset_last), axis=1)
                    tp_last = np.repeat(tp_last, (fram_tp - 1), axis=0)

                    tp = offset_all = np.concatenate((tp_first, tp_last ), axis=0)
                else:
                    offset_first = np.zeros((1, 4), dtype=int)
                    tp_first = np.concatenate((np.expand_dims(array_representation[0], axis=0), offset_first), axis=1)

                    offset_last = array_representation[1:,:4] - array_representation[:-1,:4]
                    # medium
                    i = offset_last[:-1]
                    j = array_representation[1:-1]
                    tp_med = np.concatenate((array_representation[1:-1], offset_last[:-1]), axis=1)

                    tp = offset_all = np.concatenate((tp_first, tp_med), axis=0)

                    tp_last = np.concatenate((np.expand_dims(array_representation[-1], axis=0), np.expand_dims(offset_last[-1], axis=0)), axis=1)
                    tp_last = np.repeat(tp_last, ((fram_tp - array_representation.shape[0]) + 1), axis=0)

                    tp = offset_all = np.concatenate((tp, tp_last), axis=0)
                # if array_representation.shape[0] == 1:
                #     offset = np.zeros((1,4),dtype=int)
                #     tp = np.concatenate((array_representation, offset ), axis=1)
                #     tp = np.repeat(tp, fram_tp, axis=0)
            tp = np.expand_dims(tp, axis=1)
            obj_traj_temp.append(tp)

            # x_coords = []
            # y_coords = []
            # for item in array_representation:
            #     x, y, w, h = item[0],item[1],item[2],item[3]
            #     x_coords.append(x+(w/2))
            #     y_coords.append(y+h)
            #
            # # FFT filter parameters
            # cutoff_freq_ratio = 0.3
            # averaged_coordinates = average_coordinates(np.array(x_coords), np.array(y_coords), cutoff_freq_ratio, self.frames_per_set)
            #obj_traj_temp.append(averaged_coordinates)
        obj_traj = np.concatenate(obj_traj_temp, axis=1)
        predict, conf_pre= self.net.forward(obj_traj.copy())
        for i in range(track_num):
            strack_pool[i]._tlwh = predict[i]
            strack_pool[i].pred_score = conf_pre[i][0]

        return strack_pool

    
    def update(self, output_results,  img_info, img_size):

        if output_results is None:
            return np.empty((0, 5))

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []


        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]


        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict the current location with KF
        # STrack.multi_predict(strack_pool)
        
        ################### T P ##################
        
        # check occlusion in strack_pool
        #detections_tlbr = [STrack.tlwh_to_tlbr(s._tlwh) for s in strack_pool]
        #occluded_val_list = check_occlusion(detections_tlbr)

        # Process occlusion
        if self.frame_id >1:
            strack_pool = self.process_tp(strack_pool) # process about strack_pool's _tlwh
        
        ##########################################
        
        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        dists = matching.add_score_kalman(dists, strack_pool, detections, self.args.TCM_first_step_weight,
                                          self.args.track_thresh)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # matched, unmatched_dets, unmatched_trks = associate(
        #     dets, trks, self.args.iou_thresh)


        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = matching.add_score_kalman(dists, r_tracked_stracks, detections_second,
                                                    self.args.TCM_byte_step_weight, self.args.track_thresh)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
        
            track.activate(self.frame_id)
            activated_starcks.append(track)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def fft_filter(coordinates, cutoff_freq_ratio):
    # Perform FFT
    fft_coords = fft(coordinates)
    # Remove high-frequency components
    num_freqs_to_keep = int(cutoff_freq_ratio * len(fft_coords))
    fft_coords[num_freqs_to_keep:] = 0

    filtered_coords = np.real(ifft(fft_coords))

    return filtered_coords

def average_coordinates(x_coords, y_coords, cutoff_freq_ratio, frames_per_set):
    num_sets = len(x_coords) // frames_per_set
    x_averaged_coordinates = []
    y_averaged_coordinates = []
    for i in range(num_sets):
        start = i * frames_per_set
        end = (i + 1) * frames_per_set

        # Apply FFT filter
        filtered_x = fft_filter(x_coords[start:end], cutoff_freq_ratio)
        filtered_y = fft_filter(y_coords[start:end], cutoff_freq_ratio)

        x_averaged_coordinates.append(np.median(filtered_x))
        y_averaged_coordinates.append(np.median(filtered_y))
    
    return [x_averaged_coordinates, y_averaged_coordinates]

def check_occlusion(detections_tlbr):
    # tlwh to tlbr
    if len(detections_tlbr) > 0:
        ious = matching.ious(detections_tlbr, detections_tlbr)

        occluded_val_list = []
        for iou_row in ious:
            if sum(iou_row)>1.33:
                occluded_val_list.append(True)
            else:
                occluded_val_list.append(False)
        return occluded_val_list
    else:
        return []

def calculate_v_obs(obs_traj):
    num_objects = obs_traj.shape[1]
    num_dimensions = obs_traj.shape[2]

    # Calculate the difference between consecutive position vectors
    diff = obs_traj[:, :, :, 1:] - obs_traj[:, :, :, :-1]

    # Pad the first time step with zeros to match the shape of the obs_traj tensor
    padding = torch.zeros((1, num_objects, num_dimensions, 1), dtype=torch.float32)

    # Concatenate the padding and the difference tensor
    v_obs = torch.cat((padding, diff), dim=-1)
    
    temp_total = []
    for traj in v_obs[0].tolist():
        temp = []
        for i in range(len(traj[0])):
            temp.append([traj[0][i], traj[1][i]])

        temp_total.append(temp)
    
    V_obs_final = []
    for j in range(len(temp_total[0])):
        temp = []
        for i in range(len(temp_total)):
            temp.append([temp_total[i][j][0], temp_total[i][j][1]])
        V_obs_final.append(temp)
    
    return torch.tensor(V_obs_final).unsqueeze(0)

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  #number of pedestrians in the graph
    seq_ = seq_[0]
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    

    return V

def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :],
                                       axis=0) + init_node[ped, :]

    return nodes_

