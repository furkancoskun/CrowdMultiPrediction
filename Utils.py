import os
import logging
import time
import torch
import numpy as np
from pathlib import Path
import math
import pprint

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)

def to_torch(ndarray):
    return torch.from_numpy(ndarray)

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img        

def create_logger(cfg, phase='train'):
    output_dir = Path(cfg["LOG_DIR"])
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = 'CMP_{}_{}.log'.format(time_str, phase)
    final_log_file = output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger, time_str

def print_speed(i, i_time, n, logger):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, 
                 i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def save_model(model, epoch, optimizer, model_name, checkpoint_dir, isbest=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if epoch > 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_name,
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }, isbest, checkpoint_dir, 'checkpoint_e%d.pth' % (epoch + 1))
    else:
        print('epoch not save(<5)')

def check_keys(model, pretrained_state_dict, logger, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('pretrained_dict keys:{}'.format(ckpt_keys)))
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('missing keys:{}'.format(missing_keys)))
    if print_unuse:
        logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
        logger.info(pprint.pformat('unused checkpoint keys:{}'.format(unused_pretrained_keys)))
    logger.info("<<<<<<<<<<<<<<<<<< ------------------------- >>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(pprint.pformat('used keys:{}'.format(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def load_pretrain_net(model, pretrained_path, logger, print_unuse=True):
    logger.info('load pretrained whole network from {}'.format(pretrained_path))
    logger.info(pprint.pformat(model))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = pretrained_dict
    check_keys(model, pretrained_dict, logger, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model        