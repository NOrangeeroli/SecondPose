import os
import sys
import argparse
import logging
import random

import torch
import gorilla
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from solver_old import test_func, get_logger
from dataset import TestDataset, TrainingDataset
from evaluation_utils import evaluate

def get_parser():
    parser = argparse.ArgumentParser(
        description="VI-Net")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=0,
                        help="test epoch")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dataset = args.dataset
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.log_dir = os.path.join('log', args.dataset)
    cfg.save_path = os.path.join(cfg.log_dir, 'results')
    if not os.path.isdir(cfg.save_path):
        os.makedirs(cfg.save_path)

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+"/test_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> loading model ...")
    from PN2 import Net
    ts_model = Net(cfg.n_cls)
    from VI_Net import Net
    r_model = Net(cfg.resolution, cfg.ds_rate)
    from SIM_Net import Net
    sim_model = Net(cfg.resolution, cfg.ds_rate)
    if len(cfg.gpus)>1:
        ts_model = torch.nn.DataParallel(ts_model, range(len(cfg.gpus.split(","))))
        r_model = torch.nn.DataParallel(r_model, range(len(cfg.gpus.split(","))))
        sim_model = torch.nn.DataParallel(sim_model, range(len(cfg.gpus.split(","))))
    ts_model = ts_model.cuda()
    r_model = r_model.cuda()
    sim_model= sim_model.cuda()

    checkpoint = os.path.join(cfg.log_dir, 'PN2', 'epoch_' + str(0) + '.pth')
    logger.info("=> loading PN2 checkpoint from path: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=ts_model, filename=checkpoint)

    checkpoint = os.path.join(cfg.log_dir, 'VI_Net', 'epoch_' + str(cfg.test_epoch) + '.pth')
    logger.info("=> loading VI-Net checkpoint from path: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=r_model, filename=checkpoint)

    checkpoint = os.path.join(cfg.log_dir, 'SIM_Net', 'epoch_' + str(10) + '.pth')
    logger.info("=> loading SIM-Net checkpoint from path: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=sim_model, filename=checkpoint)

    feature_file = os.path.join(BASE_DIR, cfg.feature.feature_path, cfg.feature.ref_feature_file)
    
    with open(feature_file, 'rb') as f:
        
        ref_feature= np.load(f, allow_pickle=True)
        
    ref_feature = torch.FloatTensor(ref_feature)
    
    # train_dataset  = TrainingDataset(
    #     cfg.train_dataset,
    #     cfg.dataset,
    #     'r',
    #     resolution = cfg.resolution,
    #     ds_rate = cfg.ds_rate,
    #     num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.train_dataloader.bs,
    #     )
    # # data loader
    # train_dataloder = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.train_dataloader.bs,
    #     num_workers=int(cfg.train_dataloader.num_workers),
    #     shuffle=cfg.train_dataloader.shuffle,
    #     sampler=None,
    #     drop_last=cfg.train_dataloader.drop_last,
    #     pin_memory=cfg.train_dataloader.pin_memory
    # )
    

    dataset = TestDataset(cfg.test, cfg.dataset)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False
        )
    save_path = cfg.save_path + '/VI_Net/epoch_' + str(cfg.test_epoch) 

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        test_func(ts_model, r_model, dataloder, save_path)
    evaluate(save_path, logger)
