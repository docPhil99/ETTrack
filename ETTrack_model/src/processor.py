import torch
import torch.nn as nn
from torch.nn import functional as F
from .star import STAR

from .utils import *
import torch.nn.functional as F

from tqdm import tqdm
CUDA_LAUNCH_BLOCKING = 1
from transformer.Models import Transformer

class processor(object):
    def __init__(self, args):
        self.args = args
        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)
        # args.d_word_vec = args.d_model
        # self.net1 = STAR1( args,500,
        # 500,
        # d_k=args.d_k,
        # d_v=args.d_v,
        # d_model=args.d_model,
        # d_word_vec=args.d_word_vec,
        # d_inner=args.d_inner_hid,
        # n_layers=args.n_layers,
        # n_head=args.n_head,
        # dropout=args.dropout,)
        self.set_optimizer()

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')  #dump the model structure to file - don't think this is used again.
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = 0

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')


    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        #return self.net
        test_error, test_final_error = self.test_epoch()
        print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
                                                                                           self.args.load_model,
                                                                                    test_error, test_final_error))
        return self.net

    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):
            # self.net.eval()
            # test_error, test_final_error = self.test_epoch()
            self.net.train()

            train_loss, train1_loss = self.train_epoch(epoch)

            self.save_model(epoch)
            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error = self.test_epoch()
                self.best_epoch = epoch if test_error < self.best_ade else self.best_epoch
                self.best_ade = test_error if test_error < self.best_ade else self.best_ade
                self.best_fde = test_final_error if test_error < self.best_fde else self.best_fde
                self.save_model(epoch)

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(train1_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, train1_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, train_loss, train1_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}, train1_loss={:.5f}'
                      .format(epoch, train_loss, train1_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        loss1_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            for i in range(len(batch_id)):
                batch_id[i] = list(batch_id[i])
            batch_id = list(batch_id)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()

            loss1 = torch.zeros(1).cuda()


            #batch_abs, seq_list, nei_list, nei_num, batch_pednum = inputs
            batch_abs, seq_list, batch_pednum = inputs

            # batch_abs = batch_abs.transpose(0, 1)
            # seq_list = seq_list.transpose(0, 1)
            batch_abs_ = batch_abs.cpu().tolist()
            # batch_offset = batch_abs[1:] - batch_abs[0]
            batch_offset = batch_abs[1:] - batch_abs[:-1]

            batch_offset_ = batch_offset.cpu().tolist()
########################################################################################################
            node_index = self.net.get_node_index(seq_list)
            updated_batch_pednum = self.net.update_batch_pednum(batch_pednum, node_index)
            st_ed = self.net.get_st_ed(updated_batch_pednum)
            batch_offset_nom, nomlize_all = self.net.mean_normalize_abs_input(batch_offset[:, node_index], st_ed)
            #########################################################################
            # inputs_forward = batch_abs[:-1, :, :8], batch_offset[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                      :-1], batch_pednum
            inputs_forward = batch_abs[:15, :, :], batch_offset[:15], seq_list[:15],  batch_pednum
            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)
            ###############################################################################################################################################
            last_abs = batch_abs[:15, :, :4].cpu().detach().numpy()
            pred_abs = batch_abs[:15, :, :4].cpu().detach().numpy() + outputs.cpu().detach().numpy()
            true_abs = batch_abs[1:16, :, :4].cpu().detach().numpy()
            loss_dir = 0
            fram_all = self.args.seq_length - 5
            pedinum = true_abs.shape[1]
            for framenum in range(fram_all):
                last_abs_one = convert_bbox(last_abs[framenum, :, :])
                pred_abs_one = convert_bbox(pred_abs[framenum, :, :])
                true_abs_one = convert_bbox(true_abs[framenum, :, :])

                for pedi in range(pedinum):
                    pred_cen = speed_direction(last_abs_one[pedi], pred_abs_one[pedi])
                    pred_lt = speed_direction_lt(last_abs_one[pedi], pred_abs_one[pedi])
                    pred_rt = speed_direction_rt(last_abs_one[pedi], pred_abs_one[pedi])
                    pred_lb = speed_direction_lb(last_abs_one[pedi], pred_abs_one[pedi])
                    pred_rb = speed_direction_rb(last_abs_one[pedi], pred_abs_one[pedi])

                    true_cen = speed_direction(last_abs_one[pedi], true_abs_one[pedi])
                    true_lt = speed_direction_lt(last_abs_one[pedi], true_abs_one[pedi])
                    true_rt = speed_direction_rt(last_abs_one[pedi], true_abs_one[pedi])
                    true_lb = speed_direction_lb(last_abs_one[pedi], true_abs_one[pedi])
                    true_rb = speed_direction_rb(last_abs_one[pedi], true_abs_one[pedi])

                    cost_cen = cost_vel(pred_cen, true_cen, weight=0.2)
                    cost_lt = cost_vel(pred_lt, true_lt, weight=0.2)
                    cost_rt = cost_vel(pred_rt, true_rt, weight=0.2)
                    cost_lb = cost_vel(pred_lb, true_lb, weight=0.2)
                    cost_rb = cost_vel(pred_rb, true_rb, weight=0.2)

                    cost_all = cost_cen + cost_lt + cost_rt + cost_lb + cost_rb
                    loss_dir = loss_dir + cost_all
            loss_dir = loss_dir / (pedinum * fram_all)
            ###############################################################################################################################################
            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:16], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(F.smooth_l1_loss(outputs, batch_offset[:15, :, :4], reduction='none'), dim=2)
            loss += (torch.sum(loss_o * lossmask / num))

            loss = 0.99 * loss + 0.01 * loss_dir
            if torch.isnan(loss).any():
                print("NaN loss detected. Training halted.")
                print(batch)
                # 输出相关信息以进行诊断
                break

            loss_epoch += loss.item()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            outputs_ = self.net.mean_normalize_abs_input_vert(outputs, st_ed, nomlize_all)
            loss1 = F.l1_loss(outputs_, batch_offset[:15, :, :4], reduction="mean")
            loss1_epoch += loss1.item()
            loss1_epoch = 0
            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, train_loss = {:.5f},time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(), loss1.item(),
                                                                                               end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        train_loss1_epoch = loss1_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch, train_loss1_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        loss_epoch = 0
        loss1_epoch = 0
        for batch in tqdm(range(self.dataloader.testbatchnums - 1)):
            loss = torch.zeros(1).cuda()
            loss1 = torch.zeros(1).cuda()
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            #
            # inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                                   :-1], batch_pednum
            batch_abs,  seq_list, batch_pednum = inputs

            batch_offset = batch_abs[1:] - batch_abs[:-1]
            batch_offset_ = batch_offset.cpu().tolist()

            node_index = self.net.get_node_index(seq_list)
            updated_batch_pednum = self.net.update_batch_pednum(batch_pednum, node_index)
            st_ed = self.net.get_st_ed(updated_batch_pednum)
            name = 'test'
            #batch_abs_nom = self.normalize(batch_abs[:, node_index], st_ed, batch_id, name)
            batch_offset_nom, nomlize_all = self.net.mean_normalize_abs_input(batch_offset[:, node_index], st_ed)

            #batch_offset = batch_abs[1:] - batch_abs[:-1]

            inputs_forward = batch_abs[:15, :, :], batch_offset[:15], seq_list[:15], batch_pednum

            outputs_infer = self.net.forward(inputs_forward, iftest=True)

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:16], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(F.smooth_l1_loss(outputs_infer, batch_offset[:15, :, :4], reduction='none'), dim=2)
            loss += (torch.sum(loss_o * lossmask / num))
###########################################################################################################################################

###########################################################################################################################################
            loss_epoch += loss.item()

            #outputs_ = self.normalize_vert(outputs_infer, st_ed, batch_id, name)
            outputs_ = self.net.mean_normalize_abs_input_vert(outputs_infer, st_ed, nomlize_all)
            # loss_1 = torch.sum(self.criterion(outputs_, batch_abs[1:]), dim=2)
            loss1 = F.l1_loss(outputs_, batch_offset[:15, :, :4], reduction="mean")

            loss1_epoch += loss1.item()
            loss1_epoch = 0
        #     all_output = []
        #     for i in range(self.args.sample_num):
        #         outputs_infer = self.net.forward(inputs_forward, iftest=True)
        #         all_output.append(outputs_infer)
        #     self.net.zero_grad()
        #
        #     all_output = torch.stack(all_output)
        #
        #     lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
        #     error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_offset,
        #                                                                 self.args.obs_length, lossmask)
        #
        #     error_epoch += error
        #     error_cnt_epoch += error_cnt
        #     final_error_epoch += final_error
        #     final_error_cnt_epoch += final_error_cnt
        #
        # return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
        train_loss_epoch = loss_epoch / (self.dataloader.testbatchnums - 1)
        train_loss1_epoch = loss1_epoch / (self.dataloader.testbatchnums - 1)
        return train_loss_epoch, train_loss1_epoch


def convert_bbox(x):  # predict=None
    """
  Takes a bounding box in the centre form [x,y,w,h] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    x[:, 0] = x[:, 0] - x[:, 2] / 2
    x[:, 1] = x[:, 1] - x[:, 3] / 2
    x[:, 2] = x[:, 0] + x[:, 2] / 2
    x[:, 3] = x[:, 1] + x[:, 3] / 2

    return x
def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def cost_vel(pred, true, weight):
    # Y, X = speed_direction_batch(detections, previous_obs)
    pred_Y, pred_X = pred[0], pred[1]
    true_Y, true_X = true[0], true[1]

    angle_cos = pred_Y * true_Y + pred_X * true_X
    angle_cos = np.clip(angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(angle_cos)
    #diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    angle_diff_cost = diff_angle * weight

    return angle_diff_cost