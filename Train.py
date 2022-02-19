import os
import pprint
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utils import AverageMeter, is_valid_number, print_speed
from Utils import load_pretrain_net, create_logger, save_model
from LearningRateScheduler import build_lr_scheduler

from tensorboardX import SummaryWriter

eps = 1e-5

def train(train_loader, model, optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    counting_losses = AverageMeter()
    anomaly_losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        counting_loss, anomaly_loss = model.loss()

        counting_loss = torch.mean(counting_loss)
        anomaly_loss = torch.mean(anomaly_loss)
        loss = counting_loss + anomaly_loss
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)  

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, input.size(0))

        counting_loss = counting_loss.item()
        counting_losses.update(counting_loss, input.size(0))

        anomaly_loss = anomaly_loss.item()
        anomaly_losses.update(anomaly_loss, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if ((iter + 1) % cfg["PRINT_FREQ"] == 0):
            logger.info(
                'TRAIN - Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t \
                 Counting Loss:{counting_loss.avg:.5f} \t Anomaly Loss:{anomaly_loss.avg:.5f} \t Total Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, counting_loss=counting_losses, anomaly_loss=anomaly_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg["END_EPOCH"] * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalars('Train Losses', {'train_total_loss' : losses, 'train_counting_loss' : counting_losses.avg,
                            'train_anomaly_loss' : anomaly_losses.avg},global_steps)        
        writer_dict['train_global_steps'] = global_steps + 1



def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params




