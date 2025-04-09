import torch
import torch.nn as nn
from torch.nn import functional as F
from .star import STAR
#from .star_one import STAR1
#from .star_two import STAR2
#from .star_sort import STAR_sort
from .tcn_transformer import tcn_transformer
from .utils import *
import torch.nn.functional as F

from tqdm import tqdm
from loguru import logger
CUDA_LAUNCH_BLOCKING = 1
from STAR.transformer.Models import Transformer

class predict(object):
    def __init__(self, args):

        self.args = args

        #self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)
        #self.net_sort = STAR_sort(args)
        self.tcn_transformer = tcn_transformer(args)
        #self.net2 = STAR2(args)
        args.d_word_vec = args.d_model
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
            #self.net1 = self.net1.cuda()
            #self.net_sort = self.net_sort.cuda()
            self.tcn_transformer = self.tcn_transformer.cuda()

        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):
        if self.args.ettrack_model_path:
            self.args.model_save_path = self.args.ettrack_model_path
        else:
            logger.info(f'Loading model from {self.args.load_model}')
            if self.args.load_model is not None:
                self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
                logger.info(f'Generated model save path: {self.args.model_save_path}')
        if os.path.isfile(self.args.model_save_path):
            print('Loading checkpoint')
            checkpoint = torch.load(self.args.model_save_path)
            model_epoch = checkpoint['epoch']
            #self.net_sort.load_state_dict(checkpoint['state_dict'])
            self.tcn_transformer.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            logger.error(f'No model file at {self.args.load_model}')


    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')


    def test(self):

        #print('Testing begin')
        self.load_model()
        #self.net_sort.eval()
        self.tcn_transformer.eval()

        #return self.net_sort
        return self.tcn_transformer
        # test_error, test_final_error = self.test_epoch()
        # print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
        #                                                                                   self.args.load_model,
        #                                                                                test_error, test_final_error))
    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            # self.net.eval()
            # test_error, test_final_error = self.test_epoch()

            self.net.train()
            self.net1.train()
            self.net2.train()

            train_loss, train1_loss = self.train_epoch(epoch)

            self.save_model(epoch)
            if epoch >= self.args.start_test:
                self.net.eval()
                self.net1.eval()
                self.net2.eval()
                test_error, test_final_error = self.test_epoch()
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if test_final_error < self.best_fde else self.best_epoch
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
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
                print('----epoch {}, train_loss={:.5f},train1_loss={:.5f}'
                      .format(epoch, train_loss, train1_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        loss1_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            loss1 = torch.zeros(1).cuda()
            #loss = torch.zeros(1)
            # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            # inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                                   :-1], batch_pednum

            batch_abs, seq_list, nei_list, nei_num, batch_pednum = inputs
            # batch_abs = batch_abs.transpose(0, 1)
            # seq_list = seq_list.transpose(0, 1)
            # outputs_ = batch_abs.cpu().tolist()
            batch_offset = batch_abs[1:] - batch_abs[0]
            batch_offset_ = batch_offset.cpu().tolist()
            inputs_forward = batch_abs, batch_offset[:-1], seq_list[1:-1], nei_list[1:-1], nei_num[1:-1], batch_pednum

########################################################################################################

            node_index = self.net.get_node_index(seq_list)
            updated_batch_pednum = self.net.update_batch_pednum(batch_pednum, node_index)
            st_ed = self.net.get_st_ed(updated_batch_pednum)
            batch_abs_nom, nomlize_all = self.net.mean_normalize_abs_input(batch_abs[:, node_index], st_ed)
            # batch_abs_ = batch_abs.cpu().tolist()
            # batch_abs_nom_ = batch_abs_nom.cpu().tolist()
            # batch_abs_convert = self.net.mean_normalize_abs_input_vert(batch_abs, st_ed, nomlize_all)
            # batch_abs_convert = batch_abs_convert.cpu().tolist()
            #########################################################################
            # inputs_forward = batch_abs_nom[:-1], batch_abs[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                      :-1], batch_pednum
            # inputs_forward = batch_abs_nom, batch_abs, seq_list, nei_list, nei_num, batch_pednum

            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            #outputs__ = outputs_.cpu().tolist()
            # b = batch_abs[:-1].cpu().tolist()

            #outputs = outputs + batch_abs[:-1]
            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # batch_abs_dema = batch_abs[:-1, node_index].cpu().tolist()
            outputs_test = outputs.cpu().tolist()
            #batch_abs_test = batch_abs[1:].cpu().tolist()
            #batch_abs_nom_test = batch_abs_nom[1:, node_index].cpu().tolist()
            #outputs1 = outputs + batch_abs[:-1]
            #loss_o = torch.sum(self.criterion(outputs, batch_abs[1:]), dim=2)

            loss_o = F.l1_loss(outputs, batch_offset[1:9], reduction="mean")

            # loss_o = torch.sum(self.criterion(outputs_, batch_abs[1:, :, :4]), dim=2)
            #outputs_1 = outputs.cpu().tolist()
            #b = batch_abs[2:].cpu().tolist()
            #loss_o = torch.sum(self.criterion(outputs, batch_abs[:, 1:, :]), dim=2)
            #loss_o = F.l1_loss(outputs, batch_abs[2:], reduction="sum")
            #loss_1 = F.mse_loss(outputs, batch_abs[:, 1:, :])
            if torch.isnan(loss_o).any():
                print("NaN loss detected. Training halted.")
                print(batch)
                # 输出相关信息以进行诊断
                break

            #loss += (loss_o / lossmask.sum()) / 4
            #loss += (torch.sum(loss_o * lossmask / num))
            #loss += loss_o / lossmask.sum()
           # loss = loss.clone()
            loss_epoch += loss_o.item()

            torch.autograd.set_detect_anomaly(True)
            #retain_graph = True
            #inputs=list(self.net.parameters())
            loss_o.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)

            self.optimizer.step()

            end = time.time()

            outputs_ = self.net.mean_normalize_abs_input_vert(outputs, st_ed, nomlize_all)
            # loss_1 = torch.sum(self.criterion(outputs_, batch_abs[1:]), dim=2)
            # loss1 += (torch.sum(loss_1 * lossmask / num))
            loss1 = F.l1_loss(outputs_, batch_offset[1:9], reduction="mean")
            loss1_epoch += loss1.item()
            loss1_epoch = 0
            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, train1_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss_o.item(),loss1.item(),
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
            batch_abs, seq_list, nei_list, nei_num, batch_pednum = inputs

            node_index = self.net.get_node_index(seq_list)
            updated_batch_pednum = self.net.update_batch_pednum(batch_pednum, node_index)
            st_ed = self.net.get_st_ed(updated_batch_pednum)
            batch_abs_nom, nomlize_all = self.net.mean_normalize_abs_input(batch_abs[:, node_index], st_ed)

            batch_offset = batch_abs[1:] - batch_abs[0]

            # inputs_forward = batch_abs_nom[:-1], batch_abs[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                    :-1], batch_pednum
            inputs_forward = batch_abs, batch_offset[:-1], seq_list[1:-1], nei_list[1:-1], nei_num[1:-1], batch_pednum
            outputs_infer = self.net.forward(inputs_forward, iftest=False)

            loss = F.l1_loss(outputs_infer, batch_offset[1:9], reduction="mean")
            loss_epoch += loss.item()

            outputs_ = self.net.mean_normalize_abs_input_vert(outputs_infer, st_ed, nomlize_all)
            # outputs_ = self.normalize_vert(outputs_infer, st_ed, batch_id)
            # loss_1 = torch.sum(self.criterion(outputs_, batch_abs[1:]), dim=2)
            loss1 = F.l1_loss(outputs_, batch_abs[1:9], reduction="mean")
            loss1_epoch += loss1.item()
            loss1_epoch = 0
            # all_output = []
            # for i in range(self.args.sample_num):
            #     outputs_infer = self.net.forward(inputs_forward, iftest=True)
            #     outputs = self.net.mean_normalize_abs_input_vert(outputs_infer, st_ed, nomlize_all)
            #     all_output.append(outputs[:, :, :2])
            # self.net.zero_grad()
            #
            # all_output = torch.stack(all_output)
            #
            # lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_abs[1:, :, :2],
            #                                                             self.args.obs_length, lossmask)
            #
            # error_epoch += error
            # error_cnt_epoch += error_cnt
            # final_error_epoch += final_error
            # final_error_cnt_epoch += final_error_cnt
        train_loss_epoch = loss_epoch / (self.dataloader.testbatchnums - 1)
        train_loss1_epoch = loss1_epoch / (self.dataloader.testbatchnums - 1)
        #return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
        return train_loss_epoch, train_loss1_epoch