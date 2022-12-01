import sys
import time
import torch
import datetime
import warnings

from ..utils import *
from tqdm import tqdm
from copy import deepcopy


warnings.filterwarnings('ignore')


class BMTrainer(BMBaseTrainer):

    def __init__(self, configs, exp_dir, train_loader, val_loader, resume):
        super(BMTrainer, self).__init__(configs, exp_dir, train_loader, val_loader, resume)

    def forward(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(self.train_loader, epoch)

    def _train_epoch(self, loader, epoch):
        self.model.train()

        for step, batch_data in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                # lr scheduler on per iteration
                self.scheduler.step(step / len(loader) + (epoch - 1))

                feature, label = [d.to(self.device) for d in batch_data]
                pred = self.model(feature)
                # print(feature.size(), label.size(), pred.size())

                loss = self.loss(pred, label)
                print(loss.item())

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

    #     header = 'Train Epoch:[{}]'.format(epoch)
    #     logger = MetricLogger(header, self.print_freq)
    #     logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

    #     data_iter = logger.log_every(loader)
    #     for iter_step, data in enumerate(data_iter):

    #         # lr scheduler on per iteration
    #         if iter_step % self.accum_iter == 0:
    #             self._adjust_learning_rate(iter_step / len(loader) + (epoch - 1))
    #         logger.update(lr=self.optimizer.param_groups[0]['lr'])

    #         # forward
    #         with autocast():
    #             images, targets, classes = [d.to(self.device) for d in data]
    #             preds = self.model(images)
    #             if isinstance(preds, tuple):
    #                 seg_preds, cls_preds = preds
    #             else:
    #                 seg_preds = preds
    #                 cls_preds = None

    #             seg_loss = self.seg_loss_func(seg_preds, targets)
    #             if torch.isnan(seg_loss):
    #                 print('seg_loss is nan')
    #             logger.update(seg=seg_loss.item())
    #             loss = seg_loss

    #             if (cls_preds is not None) and (self.cls_loss_func is not None):
    #                 cls_loss = self.cls_loss_func(cls_preds, classes)
    #                 if torch.isnan(cls_loss):
    #                     print('cls_loss is nan')
    #                 logger.update(cls=cls_loss.item())
    #                 loss += cls_loss

    #             if torch.isnan(loss):
    #                 print('loss is nan')
    #                 sys.exit(1)

    #         loss4opt = loss / self.accum_iter
    #         self.scaler.scale(loss4opt).backward()
    #         if (iter_step + 1) % self.accum_iter == 0:
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #             self.optimizer.zero_grad()

    #     logger_info = {
    #         key: meter.global_avg
    #         for key, meter in logger.meters.items()
    #     }
    #     return logger_info

    # @torch.no_grad()
    # def _val_epoch(self, loader, epoch):
    #     self.model.eval()

    #     all_valid_preds, all_valid_labels = [], []
    #     for _, data in tqdm(enumerate(loader), total=len(loader), ncols=66):
    #         images, valid_labels, _ = [d for d in data]
    #         images = images.to(self.device)
    #         images_dflip = torch.flip(images, [2]).to(self.device)
    #         images_cflip = torch.flip(images, [4]).to(self.device)
    #         valid_images = [images, images_dflip, images_cflip]

    #         preds_list = []
    #         for image in valid_images:
    #             pred = self.model(image)
    #             if isinstance(pred, tuple):
    #                 seg_pred, _ = pred
    #             else:
    #                 seg_pred = pred
    #             seg_pred = torch.sigmoid(seg_pred)

    #             seg_pred = seg_pred.cpu().numpy()
    #             # seg_pred: (B, 1, D, H, W)
    #             preds_list.append(seg_pred)

    #         # revert tta image
    #         preds_list[1] = np.flip(preds_list[1], [2])
    #         preds_list[2] = np.flip(preds_list[2], [4])

    #         # ensemble predictions
    #         preds = np.concatenate(preds_list, axis=1)
    #         preds_ensemble = np.mean(preds, axis=1)
    #         # preds_ensemble: (B, D, H, W)

    #         for pred in preds_ensemble:
    #             pred_blur = gaussian_filter(pred, sigma=1.5)
    #             # pred_blur: (D, H, W)
    #             all_valid_preds.append(pred_blur)
            
    #         for label in valid_labels[:, 0, ...].numpy():
    #             # label: (D, H, W)
    #             all_valid_labels.append(label)

    #     # evaluate using dynamic threshold
    #     metrics_dynamic = evaluate(
    #         y_det=iter(all_valid_preds), y_true=iter(all_valid_labels),
    #         y_det_postprocess_func=lambda pred: self._dynamic_postprocess(pred)
    #     )
    #     Dap = metrics_dynamic.AP
    #     Dauroc = metrics_dynamic.auroc
    #     Dscore = metrics_dynamic.score
    #     Dmessage = ' Val  Epoch:[{}] Dynamic score:{:.4f} ap:{:.4f} auroc:{:.4f}'
    #     print(Dmessage.format(epoch, Dscore, Dap, Dauroc))

    #     # evaluate using static threshold
    #     metrics_static = evaluate(
    #         y_det=iter(all_valid_preds), y_true=iter(all_valid_labels),
    #         y_det_postprocess_func=lambda pred: self._static_postprocess(pred)
    #     )
    #     Sap = metrics_static.AP
    #     Sauroc = metrics_static.auroc
    #     Sscore = metrics_static.score
    #     Smessage = ' Val  Epoch:[{}] Static  score:{:.4f} ap:{:.4f} auroc:{:.4f}'
    #     print(Smessage.format(epoch, Sscore, Sap, Sauroc))

    #     logger_info = {
    #         'Dscore': Dscore, 'Dap': Dap, 'Dauroc': Dauroc,
    #         'Sscore': Sscore, 'Sap': Sap, 'Sauroc': Sauroc
    #     }
    #     return logger_info

    # def _dynamic_postprocess(self, pred):
    #     # process softmax prediction to detection map dynamic
    #     pred = extract_lesion_candidates(pred, threshold=self.val_dynamic)[0]

    #     if self.remove_ring:
    #         # remove (some) secondary concentric/ring detections
    #         pred[pred < (np.max(pred) / 5)] = 0

    #     return pred
    
    # def _static_postprocess(self, pred):
    #     # process softmax prediction to detection map static
    #     pred = extract_lesion_candidates(pred, threshold=self.val_static)[0]
    #     return pred
