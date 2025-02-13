from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluatorDance as MOTEvaluator

from utils.args import make_parser
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import yaml
from STAR.src.predict import predict
from STAR.transformer.Translator import *
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu, net):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True
    rank = args.local_rank
    file_name = os.path.join(exp.output_dir, args.expn)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    result_dir = "{}_test".format(args.expn) if args.test else "{}_val".format(args.expn)
    results_folder = os.path.join(file_name, result_dir)
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))
#######################################################################################################
    # hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
    #                "--BENCHMARK MOT20 " \
    #                "--SPLIT_TO_EVAL train " \
    #                "--SEQMAP_FILE datasets/MOT20/train/train_seqmap.txt " \
    #                "--TRACKERS_TO_EVAL '' " \
    #                "--SKIP_SPLIT_FOL True " \
    #                "--METRICS HOTA CLEAR Identity VACE " \
    #                "--TIME_PROGRESS False " \
    #                "--USE_PARALLEL False " \
    #                "--NUM_PARALLEL_CORES 1  " \
    #                "--GT_FOLDER datasets/MOT20/train " \
    #                "--TRACKERS_FOLDER " + results_folder + " " \
    #                "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    # os.system(hota_command)
#######################################################################################################

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "ocsort_x_mot20.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start tracking
    if args.TCM_first_step:
        *_, summary = evaluator.evaluate_sort_score(
            args,model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )
    else:
        *_, summary = evaluator.evaluate_sort(
                args, model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder, net
        )
    
    if args.test:
        # we skip evaluation for inference on test set
        return 

    # if we evaluate on validation set, 
    logger.info("\n" + summary)

    # evaluate on the validation set
    ################################################################
    # # evaluate on the validation set
    # args.dataset = "MOT17"
    # if args.dataset == "animaltrack":
    #     hota_command = "python3 TrackEval/scripts/run_mot_challenge.py " \
    #                    "--SPLIT_TO_EVAL test  " \
    #                    "--METRICS HOTA CLEAR Identity " \
    #                    "--GT_FOLDER datasets/animaltrack/test " \
    #                    "--SEQMAP_FILE datasets/animaltrack/test/test_seqmap.txt " \
    #                    "--SKIP_SPLIT_FOL True " \
    #                    "--TRACKERS_TO_EVAL '' " \
    #                    "--TRACKER_SUB_FOLDER ''  " \
    #                    "--USE_PARALLEL True " \
    #                    "--NUM_PARALLEL_CORES 8 " \
    #                    "--PLOT_CURVES False " \
    #                    "--TRACKERS_FOLDER " + results_folder
    # elif args.dataset == "MOT17":
    #     hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
    #                    "--BENCHMARK MOT17 " \
    #                    "--SPLIT_TO_EVAL train " \
    #                    "--SEQMAP_FILE datasets/MOT17/train/train_seqmap.txt " \
    #                    "--TRACKERS_TO_EVAL '' " \
    #                    "--METRICS HOTA CLEAR Identity VACE " \
    #                    "--TIME_PROGRESS False " \
    #                    "--USE_PARALLEL False " \
    #                    "--NUM_PARALLEL_CORES 1  " \
    #                    "--GT_FOLDER datasets/MOT17/train " \
    #                    "--TRACKERS_FOLDER " + results_folder + " " \
    #                                                            "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    # elif args.dataset == "MOT20":
    #     hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
    #                    "--BENCHMARK MOT20 " \
    #                    "--SPLIT_TO_EVAL train " \
    #                    "--SEQMAP_FILE datasets/MOT20/train/train_seqmap.txt " \
    #                    "--TRACKERS_TO_EVAL '' " \
    #                    "--SKIP_SPLIT_FOL True " \
    #                    "--METRICS HOTA CLEAR Identity VACE " \
    #                    "--TIME_PROGRESS False " \
    #                    "--USE_PARALLEL False " \
    #                    "--NUM_PARALLEL_CORES 1  " \
    #                    "--GT_FOLDER datasets/MOT20/train " \
    #                    "--TRACKERS_FOLDER " + results_folder + " " \
    #                                                            "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    #     os.system(hota_command)
    # else:
    #     assert args.dataset in ["MOT20", "MOT17"]
    # os.system(hota_command)
    #################################################################
    hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                   "--BENCHMARK MOT20 " \
                   "--SPLIT_TO_EVAL train " \
                   "--SEQMAP_FILE datasets/MOT20/train/train_seqmap.txt " \
                   "--TRACKERS_TO_EVAL '' " \
                   "--SKIP_SPLIT_FOL True " \
                   "--METRICS HOTA CLEAR Identity VACE " \
                   "--TIME_PROGRESS False " \
                   "--USE_PARALLEL False " \
                   "--NUM_PARALLEL_CORES 1  " \
                   "--GT_FOLDER datasets/MOT20/train " \
                   "--TRACKERS_FOLDER " + results_folder + " " \
                                                           "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    os.system(hota_command)
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.expn:
        args.expn = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    args.save_dir = args.save_base_dir + str(args.test_set) + '/'
    args.model_dir = args.save_base_dir + str(args.test_set) + '/' + args.train_model + '/'

    trainer = predict(args)
    net = trainer.test()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu, net),
    )

# MOT17-13-DPM
# MOT17-13-FRCNN
# MOT17-13-SDP