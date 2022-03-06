import pprint
import time
import yaml
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
from torch.utils.data import DataLoader
from CrowdMultiPredictionModel import CrowdCounting
from CMP_Dataset import CMP_Dataset
from Utils import AverageMeter, is_valid_number, print_speed
from Utils import load_pretrain_net, create_logger, save_model
from LearningRateScheduler import build_lr_scheduler

from tensorboardX import SummaryWriter

eps = 1e-5

def train(train_loader, model, optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    counting_losses = AverageMeter()
    end = time.time()

    model.train()
    model = model.to(device)

    for iter, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        frames, count_gts, _ = data
        frames = frames.squeeze(dim=0).to(device)
        count_gts = count_gts.squeeze(dim=0).float().to(device)
        count_outs = model(frames)
        loss = model.loss(count_outs, count_gts)

        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()

        if is_valid_number(loss):
            optimizer.step()
        else:
            logger.info("loss: " + str(loss))
            logger.info("loss is not a valid number! Optimizer not stepped!")

        counting_losses.update(loss)

        batch_time.update(time.time() - end)
        
        if ((iter + 1) % cfg["LOG_PRINT_FREQ"] == 0):
            logger.info(
                'TRAIN - Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t \
                    Counting Loss:{counting_loss.avg:.5f}'.format( epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, 
                    data_time=data_time, counting_loss=counting_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg["END_EPOCH"] * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalars('Train_Losses', {'train_counting_loss' : counting_losses.avg, 
                            'learning_rate' : cur_lr},global_steps)       
        writer.add_scalars('Epoch_Iter', {'epoch' : epoch, 'iter' : iter+1},global_steps)      
        writer_dict['train_global_steps'] = global_steps + 1
    
        end = time.time()

def check_trainable(model, logger):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def build_opt_lr(cfg, model, logger, freeze_backbone=False):
    if freeze_backbone:
        logger.info("BACKBONE FREEZED")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True        

    for m in model.parameters():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,model.parameters()),
                          'lr': cfg["LEARNING_RATE"]["START_LR"]}]

    logger.info("check trainable params")
    logger.info(pprint.pformat('trainable_params:{}'.format(trainable_params)))

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg["MOMENTUM"],
                                weight_decay=cfg["WEIGHT_DECAY"])

    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    lr_scheduler.step()
    return optimizer, lr_scheduler


def main():
    yaml_name = "train.yaml"
    yaml_file = open(yaml_name, 'r')
    cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    logger, time_str = create_logger(cfg, 'counting_train_')
    logger.info(pprint.pformat(cfg))

    tensorboard_writer_path = os.path.join(cfg["TENSORBOARD_DIR"], "counting_train_" + time_str)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tensorboard_writer_path),
        'train_global_steps': 0,
        'validation_global_steps': 0,
    }
 
    model = CrowdCounting(pretrainedBackbone=cfg["LOAD_ONLY_PRETRAINED_BACKBONE"]).cuda()

    logger.info(pprint.pformat(model))

    if cfg["LOAD_PRETRAINED_MODEL"]:
        model = load_pretrain_net(model, cfg["PRETRAINED_MODEL_PATH"], logger=logger)
  
    optimizer, lr_scheduler = build_opt_lr(cfg, model, logger, freeze_backbone=cfg["FREEZE_BACKBONE"])

    # check trainable again
    check_trainable(model, logger)

    # parallel
    gpus = [int(i) for i in str(cfg["GPUS"]).split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

    #writer_dict['writer'].add_graph(model)
    logger.info(lr_scheduler)
    logger.info('model prepare done')

    train_set = CMP_Dataset(cfg, logger=logger, train=True)
 
    train_loader = DataLoader(train_set, batch_size=cfg["BATCH_SIZE"] * gpu_num, num_workers=cfg["DATALOADER_WORKERS"], 
                              pin_memory=True, sampler=None, drop_last=True)
    logger.info("Dataloader Created!")

    for epoch in range(cfg["END_EPOCH"]):
        if cfg["FREEZE_BACKBONE"] and (epoch == cfg["BACKBONE_UNFREEZE_EPOCH"]):
            logger.info('Time to unfreeze the backbone')
            optimizer, lr_scheduler = build_opt_lr(cfg, model, logger, freeze_backbone=False)
            check_trainable(model, logger)
            
        lr_scheduler.step()
        curLR = lr_scheduler.get_cur_lr()

        train(train_loader, model, optimizer, epoch + 1, curLR, cfg, writer_dict, logger, device)
        
        # save model
        save_model(model, epoch, optimizer, "CrowdCounting", cfg["CHECKPOINT_DIR"], isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
