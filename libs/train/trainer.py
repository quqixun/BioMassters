import time
import torch
import datetime
import warnings
import numpy as np

from ..utils import *
from ..process import *
from torch.cuda.amp import autocast, GradScaler 


warnings.filterwarnings('ignore')


class BMTrainer(BMBaseTrainer):

    def __init__(self, configs, exp_dir, resume):
        super(BMTrainer, self).__init__(configs, exp_dir, resume)
        self.scaler = GradScaler()

    def forward(self, train_loader, val_loader):
        self.norm_stats = val_loader.dataset.norm_stats['label']
        self.process_method = val_loader.dataset.process_method

        best_val_rmse = np.inf
        start_time = time.time()
        basic_msg = 'RMSE:{:.4f} Epoch:{}'

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self._train_epoch(epoch, train_loader)
            val_metrics   = self._val_epoch(epoch, val_loader)

            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                best_msg = basic_msg.format(best_val_rmse, epoch)
                print('>>> Best Val Epoch - Lowest RMSE - Save Model <<<')
                self._save_model()

            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch == self.epochs):
                self._save_checkpoint(epoch)

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)
            print()

        print(best_msg)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('- Training time {}'.format(total_time_str))

    def _train_epoch(self, epoch, loader):
        self.model.train()
        self.optimizer.zero_grad()

        header = 'Train Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)
        logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

        data_iter = logger.log_every(loader)
        for step, batch_data in enumerate(data_iter):
            # lr scheduler on per iteration
            current = step / len(loader) + (epoch - 1)
            self.scheduler.step(current)
            logger.update(lr=self.optimizer.param_groups[0]['lr'])

            with autocast():
                feature, label = [d.to(self.device) for d in batch_data]
                pred = self.model(feature)
                loss = self.loss(pred, label)
                logger.update(loss=loss.item())

            loss4opt = loss / self.accum_iter
            self.scaler.scale(loss4opt).backward()
            if (step + 1) % self.accum_iter == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics

    @torch.no_grad()
    def _val_epoch(self, epoch, loader):
        self.model.eval()

        header = ' Val  Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for step, batch_data in enumerate(data_iter):
            feature, label = [d.to(self.device) for d in batch_data]
            label = label.cpu().numpy()

            pred = self.model(feature).cpu().numpy()
            pred = recover_label(pred, self.norm_stats, self.process_method)

            rmse = np.sqrt(np.mean((pred - label) ** 2, axis=(1, 2, 3)))
            rmse = np.mean(rmse).astype(float)
            logger.update(rmse=rmse)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(pred[0, 0])
        # plt.subplot(122)
        # plt.imshow(label[0, 0])
        # plt.tight_layout()
        # plt.show()

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics
