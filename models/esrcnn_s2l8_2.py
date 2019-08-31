import os
import logging
from collections import OrderedDict
import pdb
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class esrcnn_s2l8_2(BaseModel):
    def __init__(self, opt):
        super(esrcnn_s2l8_2, self).__init__(opt)
        train_opt = opt['train']

        self.net = networks.define_ESRCNN_S2L8_2(opt)

        if self.is_train:
            self.net.train()
        self.load()

        if self.is_train:
            self.cri_pix = nn.MSELoss()
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=train_opt['lr'], \
                weight_decay=wd, betas=(train_opt['beta1'],0.999))

            if train_opt['lr_scheme'] == 'MultiStepLR':
                self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer, \
                    train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()

    def feed_data_cpu(self, data, need_LB=True):
        inputs = torch.cat((data[0],data[1]), 1)      

        self.inputs = Variable(inputs)
        self.input_1 = Variable(data[0])
        self.input_2 = Variable(data[1])
        if need_LB:
            self.label = Variable(data[2])       

    def feed_data(self, data, need_LB=True):
        #L8_cloud_up = torch.nn.functional.upsample(input=data[0], size=(data[1].size(2), data[1].size(3)), mode='bilinear')
        inputs = torch.cat((data[0],data[1]), 1)

        self.inputs = Variable(inputs.cuda(0))
        self.input_1 = Variable(data[0].cuda(0))
        self.input_2 = Variable(data[1].cuda(0))
        if need_LB:
            self.label = Variable(data[2].cuda(0))


    def optimize_parameters(self, step):        
        self.optimizer.zero_grad()
        self.results = self.net(self.inputs)

        l_pix = 0
        l_pix = self.cri_pix(self.results, self.label)

        l_pix.backward()
        self.optimizer.step()

        self.log_dict['l_pix'] = l_pix.data.cpu().numpy()

    def val(self):
        self.net.eval()
        self.results = self.net(self.inputs)
        self.net.train()

    def test(self):
        with torch.no_grad():
            self.net.eval()
            self.results = self.net(self.inputs)
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LB = True):
        out_dict = OrderedDict()
        out_dict['input_1'] = self.input_1.detach()[0].float().data.cpu()
        out_dict['input_2'] = self.input_2.detach()[0].float().data.cpu()
        out_dict['Pred'] = self.results.detach()[0].float().data.cpu()
        if need_LB:
            out_dict['label'] = self.label.detach()[0].float().data.cpu()
        return out_dict

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading pretrained model [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net)

    def save(self, iter_step):
        self.save_network(self.net, 'ESRCNN_S2L8_2', iter_step)
