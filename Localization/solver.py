# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:54
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @Software: PyCharm

import time
import torch
import numpy as np
from utils.utils import ensure_dir
from tensorboardX import SummaryWriter
from torch.nn.functional import interpolate
from model.metric import dc_score, MeanIoU
from model.loss import MSE_loss, DiceLoss, BCE_loss #引入不同loss
import os
import SimpleITK as sitk

class Solver():
    def __init__(self, gpus, model, criterion, metric, batch_size, epochs, lr,
                 trainset, valset, train_loader, val_loader, logs_dir,
                 patience, checkpoint_dir=None, scale=1.0, weight=None,
                 resume=False, resume_path=None, log=None, tolerate_shape=(128, 384, 384)):
        self.log = log
        self.device, device_ids = self._prepare_device(gpus)
        log.info("Available Devices: %s" % device_ids)
        self.model = model.to(self.device)
        self.cri_name = criterion
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model.to(self.device), device_ids=device_ids)
        self.weight = weight
        if criterion == "MSE_loss" or criterion == "DiceLoss":
            self.criterion = eval(criterion+"()")
        else:
            self.criterion = eval(criterion)
        self.metric = eval(metric)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.lr, betas=(0.9, 0.99))
        self.best_val_loss = np.inf
        self.best_mean_dice = 0

        self.len_train = len(trainset)
        self.len_val = len(valset)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(checkpoint_dir)
        self.scale = scale
        self.resume_path = resume_path
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  patience=patience, factor=0.5)
        self.tolerate_shape = tolerate_shape ##????

        if resume is True:
            self.start_epoch = self._resume_checkpoint(resume_path)
        else:
            self.start_epoch = 0
            log.info("Start a new training!")

        ensure_dir(logs_dir)
        self.writer = SummaryWriter(logs_dir)

    def train_epoch(self, epoch):
        self.model.train()
        loss_t = 0
        train_path = './training'
        train_path = os.path.join(train_path,str(epoch))
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        for batch_idx, (data, target) in enumerate(self.train_loader):

            if (data.shape[3] > self.tolerate_shape[1]) or (data.shape[2] > self.tolerate_shape[0]):
                data = interpolate(data, scale_factor=(self.tolerate_shape[0]/data.shape[2],
                                                       self.tolerate_shape[1]/data.shape[3],
                                                       self.tolerate_shape[1]/data.shape[3]),
                                                       mode="trilinear",align_corners=True)
                target = interpolate(target, scale_factor=(self.tolerate_shape[0]/target.shape[2],
                                                           self.tolerate_shape[1]/target.shape[3],
                                                           self.tolerate_shape[1]/target.shape[3]),
                                                       mode="nearest",align_corners=False)

            data = interpolate(data,scale_factor=self.scale,mode="trilinear",align_corners=True)
            target = interpolate(target, scale_factor=self.scale, mode="nearest")

            data, target = data.to(self.device), target.to(self.device).long()
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.cri_name == "BCE_loss": #如果是BCEloss 进行加权
                loss = self.criterion(output, target, weight_rate=self.weight,ignore=255).to(self.device)
            else:
                loss = self.criterion(output, target,ignore=255).to(self.device)
    
            loss.backward()
            loss_t = loss_t + loss.item() #tensor——>scalar
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.log.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx, self.len_train),
                loss.item()))

            if epoch%5==0:
                output = torch.argmax(output, dim=1)# two channels
                output = output[0].cpu().numpy().astype(np.uint8) * 255  # *255
                img = sitk.GetImageFromArray(output)
                sitk.WriteImage(img, os.path.join(train_path,str(batch_idx)+'_output.nii.gz'))

                data = data[0,0].cpu().numpy()
                data = sitk.GetImageFromArray(data)
                sitk.WriteImage(data, os.path.join(train_path,str(batch_idx)+'_raw.nii.gz'))
                # import pdb
                # pdb.set_trace()
                target = target[0,0].cpu().numpy().astype(np.uint8)
                target = sitk.GetImageFromArray(target)
                sitk.WriteImage(target, os.path.join(train_path,str(batch_idx)+'_target.nii.gz'))


        self.writer.add_scalar("Train_loss", loss_t/self.len_train, epoch)
        self.log.info("Train_loss: {:.6f},{}".format(loss_t/self.len_train, epoch))
    def validate_epoch(self, epoch):
        loss_v = 0
        dice_list,miou_list = [],[]
        validation_path = os.path.join('./validation',str(epoch))
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        for batch_idx, (data, target) in enumerate(self.val_loader):

            data, target = data.to(self.device), target.to(self.device)
            data = interpolate(data, scale_factor=self.scale, mode="trilinear", align_corners=True)
            target = interpolate(target, scale_factor=self.scale, mode="nearest")
            output = self.model(data).to(self.device)
            if self.cri_name == "BCE_loss":
                loss = self.criterion(output, target, weight_rate=self.weight,ignore=255).to(self.device)
            else:
                loss = self.criterion(output, target,ignore=255).to(self.device)
            loss_v = loss_v + loss.item()

            output = interpolate(output, scale_factor=(1/self.scale[0], 1/self.scale[1], 1/self.scale[2]), mode="nearest")
            target = interpolate(target, scale_factor=(1/self.scale[0], 1/self.scale[1], 1/self.scale[2]),
                                 mode="nearest")
            output = torch.argmax(output, dim=1)# two channels

            mean_iou = MeanIoU()
            mean_iou_value = mean_iou(output.unsqueeze(0),target)

            output = output.squeeze(0).cpu().numpy().astype(np.uint8) * 255  # *255

            #save results
            img = sitk.GetImageFromArray(output)
            sitk.WriteImage(img, os.path.join(validation_path,str(batch_idx)+'_output.nii.gz'))


            target = target.squeeze(0).squeeze(0).cpu().numpy()
            dice_score = dc_score(output, target)
            self.log.info("Validation: {} loss: {:.6f} DICE: {:.4f} MeanIoU: {:.4f}".format(batch_idx,loss.item(),dice_score,mean_iou_value.item()))
            if dice_score > 0.2:
                dice_list.append(dice_score)
            
        aver_loss = loss_v/(self.len_val)
        self.log.info('Validation Loss: {:.6f}'.format(
            aver_loss))

        self._save_checkpoint(epoch)

        self.log.info("MEAN DICE: {}".format(np.mean(dice_list)))
        self._best_model(aver_loss,np.mean(dice_list))

        self.writer.add_scalar("Validation_loss", aver_loss)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(aver_loss)
        self.log.info("Current learning rate is {}".format(self.optimizer.param_groups[0]["lr"]))
    # torch.cuda.empty_cache()
    def train(self):
        for epoch in range(self.start_epoch,self.epochs):
            start_time = time.time()
            self.train_epoch(epoch)
            self.model.eval()
            # with torch.no_grad():
            #     self.validate_epoch(epoch)
            self._save_checkpoint(epoch)
            end_time = time.time()
            self.log.info('Epoch: {} Spend Time: {:.3f}s'.format(epoch, end_time - start_time))

    def _progress(self, batch_idx,len_set):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = int(len_set/self.batch_size)
        return base.format(current, total, 100.0 * current / total)

    def _best_model(self,aver_loss,mean_dice):
        if aver_loss < self.best_val_loss:
            self.best_val_loss = aver_loss
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
        self.log.info("Best val loss: {:.6f} , Best mean dice: {:.4f} ".format(self.best_val_loss, self.best_mean_dice))


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.log.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.log.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _metric_score(self, output, target):
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        target = target.squeeze(0).squeeze(0).cpu().numpy()
        score = self.metric(output, target)
        return score

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.log.warning("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.log.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

        return self.start_epoch

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__ #model name
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = str(self.checkpoint_dir)+'/checkpoint-epoch{}.pth'.format(str(epoch))
        torch.save(state, filename)
        self.log.info("Saving checkpoint: {} ...".format(filename))