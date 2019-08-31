import sys
import torch
import argparse
import numpy as np
import os
import pdb
import logging
import math
import cv2
import scipy.misc as m

import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
from models import get_model
from utils.data_loader_S2self import DataLoader
from utils import util
import options.options as option
from torch.nn import DataParallel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/train_ESRCNN_S2self.json', help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    if opt['path']['resume_state']:
        resume_state = torch.load(opt['path']['resume_state'])
    else:
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                    and 'pretrain_model' not in key and 'resume' not in key))

    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)

    logger.info(option.dict2str(opt))

    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True

    # Setup TrainDataLoader
    trainloader = DataLoader(opt['datasets']['train']['dataroot'], split='train')
    train_size = int(math.ceil(len(trainloader) / opt['datasets']['train']['batch_size']))
    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(trainloader), train_size))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
    TrainDataLoader = data.DataLoader(trainloader, batch_size=opt['datasets']['train']['batch_size'], num_workers=12, shuffle=True)
    #Setup for validate
    valloader = DataLoader(opt['datasets']['train']['dataroot'], split='val')
    VALDataLoader = data.DataLoader(valloader,batch_size=opt['datasets']['train']['batch_size']//5, num_workers=1, shuffle=True)
    logger.info('Number of val images:{:d}'.format(len(valloader)))


    # Setup Model
    model = get_model('esrcnn_s2self',opt)

    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0


    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for i, train_data in enumerate(TrainDataLoader):

            current_step += 1
            if current_step > total_iters:
                break

            model.update_learning_rate()
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}>'.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k,v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v[0])
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v[0], current_step)
                logger.info(message)

            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                for i_val, val_data in enumerate(VALDataLoader):
                    idx += 1
                    img_name = val_data[3][0].split('.')[0]
                    model.feed_data(val_data)
                    model.val()

                    visuals = model.get_current_visuals()
                    pred_img = util.tensor2img(visuals['Pred'])
                    gt_img = util.tensor2img(visuals['label'])
                    avg_psnr += util.calculate_psnr(pred_img, gt_img)

                avg_psnr = avg_psnr / idx

                logger.info('# Validation #PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr:{:.4e}'.format(epoch, current_step, avg_psnr))

                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training')


if __name__ == '__main__':
    main()