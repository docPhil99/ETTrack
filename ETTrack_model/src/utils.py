import os
import pickle
import random
import time
import numpy as np
import torch


# from cython_bbox import bbox_overlaps as bbox_ious


class Trajectory_Dataloader():
    def __init__(self, args):

        self.args = args
        self.data_dir = './datasets'
        self.TRAIN_DATA_PATH = args.train_data_path
        self.TEST_DATA_PATH = args.val_data_path

        if True:
            self.data_dirs = os.listdir(self.TRAIN_DATA_PATH)
            test_dirs = os.listdir(self.TEST_DATA_PATH)
            for seq in sorted(self.data_dirs):
                if '.DS_Store' in seq or '.ipy' in seq:
                    self.data_dirs.remove(seq)
            for seq in sorted(test_dirs):
                if '.DS_Store' in seq or '.ipy' in seq or '.txt' in seq:
                    test_dirs.remove(seq)

            train_set = [i for i in range(len(self.data_dirs))]
            self.test_set = [i for i in range(len(test_dirs))]

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [test_dirs[x] for x in self.test_set]

        else:
            raise NotImplementedError

        self.train_data_file = os.path.join(self.args.save_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.save_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")

        print("Creating pre-processed data from raw data.")

        self.traject_preprocess('train')
        self.traject_preprocess('test')
        print("Done.")

        # Load the processed data from the pickle file
        print("Preparing data batches.")
        if not (os.path.exists(self.train_batch_cache)):
            self.frameped_dict, self.pedtraject_dict = self.load_dict(self.train_data_file)
            #self.dataPreprocess('train')
            self.trainbatch, self.trainbatchnums = self.dataPreprocess('train')
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
            #self.dataPreprocess('test')
            self.testbatch, self.testbatchnums = self.dataPreprocess('test')
        self.trainbatch, self.trainbatchnums, _, _ = self.load_cache(self.train_batch_cache) #PMB
        self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache) #PMB
        print("Done.")

        print('Total number of training batches:', self.trainbatchnums)
        print('Total number of test batches:', self.testbatchnums)

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)

    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        TRAIN_DATA_PATH = self.args.train_data_path
        TEST_DATA_PATH = self.args.val_data_path
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []

        Pedlist_data = []
        frameped_dict = []  # peds id contained in a certain frame
        pedtrajec_dict = []  # trajectories of a certain ped
        # For each dataset
        for seti, directory in enumerate(data_dirs):
            if setname == 'train':
                file_path = os.path.join(TRAIN_DATA_PATH, directory, 'gt/gt.txt')  #if datasets is animaltrack so path is gt.txt, if not,  gt/gt.txt
            else:
                file_path = os.path.join(TEST_DATA_PATH, directory, 'gt/gt.txt')
            # Load the data from the csv file
            # data = np.genfromtxt(file_path, delimiter=',')
            # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
            data = np.loadtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset

            Pedlist = np.unique(data[:, 1]).tolist()
            numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            numFrame_data.append([])
            frameped_dict.append({})
            pedtrajec_dict.append({})

            for ind, pedi in enumerate(Pedlist):
                if ind % 100 == 0:
                    print(ind, len(Pedlist))
                # Extract trajectories of one person
                FrameContainPed = data[data[:, 1] == pedi, :]
                # Extract peds list
                FrameList = FrameContainPed[:, 0].tolist()
                if len(FrameList) < 2:
                    continue
                # Add number of frames of this trajectory
                numFrame_data[seti].append(len(FrameList))
                # Initialize the row of the numpy array
                Trajectories = []
                # For each ped in the current frame

                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    if fi == 0:
                        current_lx = FrameContainPed[FrameContainPed[:, 0] == frame, 2][0]
                        current_ly = FrameContainPed[FrameContainPed[:, 0] == frame, 3][0]
                        current_w = FrameContainPed[FrameContainPed[:, 0] == frame, 4][0]
                        current_h = FrameContainPed[FrameContainPed[:, 0] == frame, 5][0]
                        #conf = FrameContainPed[FrameContainPed[:, 0] == frame, 6][0]
                        offset_lx = 0
                        offset_ly = 0
                        offset_w = 0
                        offset_h = 0
                    else:
                        current_lx = FrameContainPed[FrameContainPed[:, 0] == frame, 2][0]
                        current_ly = FrameContainPed[FrameContainPed[:, 0] == frame, 3][0]
                        current_w = FrameContainPed[FrameContainPed[:, 0] == frame, 4][0]
                        current_h = FrameContainPed[FrameContainPed[:, 0] == frame, 5][0]
                        #conf = FrameContainPed[FrameContainPed[:, 0] == frame, 6][0]
                        if FrameContainPed[fi - 1, 0] == (frame - 1):
                            offset_lx = FrameContainPed[FrameContainPed[:, 0] == frame, 2][0] - \
                                        FrameContainPed[FrameContainPed[:, 0] == (frame - 1), 2][0]
                            offset_ly = FrameContainPed[FrameContainPed[:, 0] == frame, 3][0] - \
                                        FrameContainPed[FrameContainPed[:, 0] == (frame - 1), 3][0]
                            offset_w = FrameContainPed[FrameContainPed[:, 0] == frame, 4][0] - \
                                       FrameContainPed[FrameContainPed[:, 0] == (frame - 1), 4][0]
                            offset_h = FrameContainPed[FrameContainPed[:, 0] == frame, 5][0] - \
                                       FrameContainPed[FrameContainPed[:, 0] == (frame - 1), 5][0]
                        else:
                            offset_lx = 0
                            offset_ly = 0
                            offset_w = 0
                            offset_h = 0
                    #if self.args.dataset == 'dancetrack':
                    current_lx = current_lx + current_w / 2
                    current_ly = current_ly + current_h / 2
                    Trajectories.append(
                        [int(frame), current_lx, current_ly, current_w, current_h, offset_lx, offset_ly, offset_w,
                         offset_h])
                    # else:
                    #     current_rx = current_lx + current_w
                    #     current_ry = current_ly + current_h
                    #     # Add their pedID, x, y to the row of the numpy array
                    #     Trajectories.append(
                    #         [int(frame), current_lx, current_ly, current_rx, current_ry, offset_lx, offset_ly, offset_w,
                    #          offset_h])
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []
                    frameped_dict[seti][int(frame)].append(pedi)
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)

        f = open(data_file, "wb")
        pickle.dump((frameped_dict, pedtrajec_dict), f, protocol=2)
        f.close()

    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        '''
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        seq_width = []
        seq_height = []
        for seti, dict in enumerate(data_dict):
            frames = sorted(dict)
            maxframe = max(frames) - self.args.seq_length
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))
            ########################################################################
            # if setname == 'train':
            #     seq_info = open(os.path.join(self.TRAIN_DATA_PATH, self.train_dir[seti], 'seqinfo.ini')).read()
            # else:
            #     seq_info = open(os.path.join(self.TEST_DATA_PATH, self.train_dir[seti], 'seqinfo.ini')).read()
            # width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            # height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
            # seq_width.extend(list(seti for i in range(len(frames))))
            # seq_height.extend(list(seti for i in range(len(frames))))
            #######################################################################
        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]

        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            val_fraction = 0
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            cachefile = self.train_batch_cache

        else:
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            cachefile = self.test_batch_cache
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        data_index = self.get_data_index(frameped_dict, setname, ifshuffle=shuffle)
        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]
        trainbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, train_index, setname)
        trainbatchnums = len(trainbatch)

        valbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, val_index, setname)
        valbatchnums = len(valbatch)


        #PMB added save back
        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
        f.close()

        return trainbatch, trainbatchnums

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''

        # self.args.batch_around_ped = 256

        batch_data_mass = []
        batch_data = []
        Batch_id = []

        temp = self.args.batch_around_ped
        # if setname == 'train':
        #     skip = self.trainskip
        # else:
        #     skip = self.testskip

        ped_cnt = 0
        last_frame = 0
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + self.args.seq_length])
            except:
                continue
            present_pedi = framestart_pedi | frameend_pedi
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            for ped in present_pedi:
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame,
                                                                               self.args.seq_length)
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue

                # cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 4),)
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 8),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            traject_batch = np.concatenate(traject, 1)
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]

            cur_pednum = traject_batch.shape[1]
            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)

            if cur_pednum >= self.args.batch_around_ped * 2:
                # too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id = [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]], traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []

                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))

                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)

                    batch_data = self.massup_batch(batch_data)
                    batch_data_mass.append((batch_data, Batch_id,))

                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
            if len(batch_data_mass) >= 50000 and setname == 'train':  #2000
                break
            elif len(batch_data_mass) >= 20000 and setname != 'train':
                break

        if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            batch_data_mass.append((batch_data, Batch_id,))
        self.args.batch_around_ped = temp
        return batch_data_mass

    def find_trajectory_fragment(self, trajectory, startframe, seq_length):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
        '''
        return_trajec = np.zeros((seq_length, 9))
        endframe = startframe + (seq_length)
        # if trajectory[:, 0] == startframe:
        #     start_n = trajectory[startframe, 0]
        # else:
        #     start_n = 0
        # if trajectory[:, 0] == startframe:
        #     end_n = trajectory[startframe, 0]
        # else:
        #     end_n = 0
        start_n = np.where(trajectory[:, 0] == startframe)
        end_n = np.where(trajectory[:, 0] == endframe)

        iffull = False
        ifexsitobs = False
        ####################################################################
        if (end_n[0].shape[0] - start_n[0].shape[0]) == 20 and start_n[0].shape[0] != 0 and end_n[0].shape[0] != 0:
            end_n = end_n[0][0]
            start_n = start_n[0][0]
        elif (end_n[0].shape[0] - start_n[0].shape[0]) != 20 and start_n[0].shape[0] != 0 and end_n[0].shape[0] != 0:
            start_n = start_n[0][0]
            tempp = start_n
            p = trajectory[tempp, 0] - tempp
            for i in range(seq_length):
                if (trajectory[tempp, 0] - tempp) == p:
                    tempp = tempp + 1
                else:
                    end_n = tempp - 1
            end_n = tempp
        elif start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
            return return_trajec, iffull, ifexsitobs
        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
            # start_n = start_n[0][0]
            # tempp = start_n
            # p = trajectory[tempp, 0] - tempp
            # for i in range(seq_length - 1):
            #     if (trajectory[tempp, 0] - tempp) == p and tempp < trajectory.shape[0]:
            #         tempp = tempp + 1
            #     else:
            #         end_n = tempp - 1
            # end_n = tempp
            return return_trajec, iffull, ifexsitobs
        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
            return return_trajec, iffull, ifexsitobs
        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]
        ####################################################################
        # if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
        #     start_n = 0
        #     end_n = end_n[0][0]
        #     if end_n == 0:
        #         return return_trajec, iffull, ifexsitobs
        # elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
        #     start_n = start_n[0][0]
        #     end_n = trajectory.shape[0]
        # elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
        #     start_n = 0
        #     end_n = trajectory.shape[0]
        # else:
        #     end_n = end_n[0][0]
        #     start_n = start_n[0][0]
        #########################################################################################
        candidate_seq = trajectory[start_n:end_n]
        offset_start = int((candidate_seq[0, 0] - startframe))

        offset_end = self.args.seq_length + int((candidate_seq[-1, 0] - endframe))

        return_trajec[offset_start:offset_end + 1, :9] = candidate_seq

        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True

        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs

    def massup_batch(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]

        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 8))
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))
        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch)
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        #return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
        return (nodes_batch_b, seq_list_b, batch_pednum)

    def get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        # num_frame = inputnodes.shape[0]
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))
        select_dist = []
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                select_dist = []
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)
                #############################################################################
                # if self.args.dataset == 'MOT20':
                relative_cord = seqi[select, :2] - seqj[select, :2]
                # relative_cord:(20,4)
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred_mot20) | (
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred_mot20)
                # else:
                #     relative_cord = self.iou_distance(seqi[select, :4], seqj[select, :4])
                # # invalid data index   select_dist；（20，）
                #     for i in range(relative_cord.shape[0]):
                #         select_dist.append((relative_cord[i, i]) >= self.args.neighbor_thred)

                select_dist = np.array(select_dist, dtype=bool)
                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num

    def xyxy_xybr(self, batch_data):

        # batch, batch_shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum = batch_data
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data
        batch[:, :, 2] = batch[:, :, 2] - batch[:, :, 0]
        batch[:, :, 3] = batch[:, :, 3] - batch[:, :, 1]

        # batch_shift_value[:, :, 2] = batch_shift_value[:, :, 2] - batch_shift_value[:, :, 0]
        # batch_shift_value[:, :, 3] = batch_shift_value[:, :, 3] - batch_shift_value[:, :, 1]
        #
        # shift_value[:, :, 2] = shift_value[:, :, 2] - shift_value[:, :, 0]
        # shift_value[:, :, 3] = shift_value[:, :, 3] - shift_value[:, :, 1]
        batch_data = batch, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def xycbr_xybr(self, batch_data):

        # batch, batch_shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum = batch_data
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data
        batch[:, :, 0] = batch[:, :, 0] - batch[:, :, 2] / 2
        batch[:, :, 1] = batch[:, :, 1] - batch[:, :, 3] / 2

        # batch_shift_value[:, :, 2] = batch_shift_value[:, :, 2] - batch_shift_value[:, :, 0]
        # batch_shift_value[:, :, 3] = batch_shift_value[:, :, 3] - batch_shift_value[:, :, 1]
        #
        # shift_value[:, :, 2] = shift_value[:, :, 2] - shift_value[:, :, 0]
        # shift_value[:, :, 3] = shift_value[:, :, 3] - shift_value[:, :, 1]
        batch_data = batch, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def rotate_shift_batch(self, batch_data, ifrotate=False, ifdrop=False):
        '''
        Random ration and zero shifting.
        '''
        #batch, seq_list, nei_list, nei_num, batch_pednum = batch_data
        batch, seq_list, batch_pednum = batch_data
        # rotate batch
        # 如何设置Bbox的旋转

        if ifrotate:
            th = random.random() * np.pi / 9

            cur_ori = batch.copy()
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)

        if ifdrop:
            random_num = random.randint(0, 15)
            batch = np.delete(batch, random_num, axis=0)
            seq_list = np.delete(seq_list, random_num, axis=0)

        # shift_value = np.repeat(s.reshape((1, -1, 4)), self.args.seq_length, 0)
        batch_data = batch, seq_list, batch_pednum
        #batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate, ifdrop=self.args.randomDrop)
        # if self.args.dataset == 'MOT20':
        #     batch_data = self.xycbr_xybr(batch_data)
        # else:
        #     batch_data = self.xyxy_xybr(batch_data)
        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False, ifdrop=False)
        # if self.args.dataset == 'MOT20':
        #     batch_data = self.xycbr_xybr(batch_data)
        # else:
        #     batch_data = self.xyxy_xybr(batch_data)
        return batch_data, batch_id

    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0

    # def ious(self, atlbrs, btlbrs):
    #     """
    #     Compute cost based on IoU
    #     :type atlbrs: list[tlbr] | np.ndarray
    #     :type atlbrs: list[tlbr] | np.ndarray
    #
    #     :rtype ious np.ndarray
    #     """
    #     ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    #     if ious.size == 0:
    #         return ious
    #
    #     ious = bbox_ious(
    #         np.ascontiguousarray(atlbrs, dtype=np.float),
    #         np.ascontiguousarray(btlbrs, dtype=np.float)
    #     )
    #
    #     return ious


#     def iou_distance(self, atracks, btracks):
#         """
#         Compute cost based on IoU
#         :type atracks: list[STrack]
#         :type btracks: list[STrack]
#
#         :rtype cost_matrix np.ndarray
#         """
#
#         if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
#                 len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
#             atlbrs = atracks
#             btlbrs = btracks
#         else:
#             atlbrs = [track.tlbr for track in atracks]
#             btlbrs = [track.tlbr for track in btracks]
#         _ious = self.ious(atlbrs, btlbrs)
#         cost_matrix = 1 - _ious
#
#         return cost_matrix
# ########################################################################################################
def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    '''

    if outputs.dim() == 3:
        seq_length = outputs.shape[0]
    else:
        seq_length = outputs.shape[1]

    node_pre = node_first
    lossmask = torch.zeros(seq_length, seq_list.shape[1])

    if using_cuda:
        lossmask = lossmask.cuda()

    # For loss mask, only generate for those exist through the whole window
    for framenum in range(seq_length):
        if framenum == 0:
            lossmask[framenum] = seq_list[framenum] * node_pre
        else:
            lossmask[framenum] = seq_list[framenum] * lossmask[framenum - 1]

    return lossmask, sum(sum(lossmask) * 4)


def L2forTest(outputs, targets, obs_length, lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[obs_length - 1:, pedi_full]
    error = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()

    return error.item(), error_cnt, final_error.item(), final_error_cnt, error_full


def L2forTestS(outputs, targets, obs_length, lossMask, num_samples=20):
    '''
    Evaluation, stochastic version
    '''
    seq_length = outputs.shape[1]
    error = torch.norm(outputs - targets, p=2, dim=3)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[:, obs_length - 1:, pedi_full]

    error_full_sum = torch.sum(error_full, dim=1)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)

    error = torch.sum(error_full_sum_min)
    error_cnt = error_full.numel() / num_samples

    final_error = torch.sum(best_error[-1])
    final_error_cnt = error_full.shape[-1]

    return error.item(), error_cnt, final_error.item(), final_error_cnt


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return timed


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
