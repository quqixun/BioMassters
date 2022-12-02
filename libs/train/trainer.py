import time
import torch
import datetime
import warnings
import numpy as np

from ..utils import *


warnings.filterwarnings('ignore')


class BMTrainer(BMBaseTrainer):

    def __init__(self, configs, exp_dir, train_loader, val_loader, resume):
        super(BMTrainer, self).__init__(configs, exp_dir, train_loader, val_loader, resume)

    def forward(self):

        best_val_rmse = np.inf
        start_time = time.time()
        basic_msg = 'RMSE:{:.4f} Epoch:{}'

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

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


    def _train_epoch(self, epoch):
        self.model.train()

        header = 'Train Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)
        logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

        accumu_loss = 0.0
        data_iter = logger.log_every(self.train_loader)
        for step, batch_data in enumerate(data_iter):
            feature, label = [d.to(self.device) for d in batch_data]

            with self.accelerator.accumulate(self.model):
                # lr scheduler on per iteration
                current = step / len(self.train_loader) + (epoch - 1)
                self.scheduler.step(current)
                logger.update(lr=self.optimizer.param_groups[0]['lr'])

                pred = self.model(feature)
                loss = self.loss(pred, label)
                accumu_loss += loss

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients and \
               self.accelerator.is_main_process:
                step_loss = accumu_loss / self.accum_iter
                step_loss = self.accelerator.gather(step_loss)
                logger.update(rec_loss=step_loss.item())
                accumu_loss = 0.0

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()

        header = ' Val  Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(self.val_loader)
        for step, batch_data in enumerate(data_iter):
            feature, label = [self._type_convert(d.to(self.device))
                              for d in batch_data]

            pred = self.model(feature)
            # if self.accelerator.is_main_process:
            preds, labels = self.accelerator.gather_for_metrics((pred, label))
            preds  = recover_data(preds.cpu().numpy())
            labels = recover_data(labels.cpu().numpy())
            rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=(1, 2, 3)))
            rmse = np.mean(rmse).astype(float)
            logger.update(rmse=rmse)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(preds[0, 0])
        # plt.subplot(122)
        # plt.imshow(labels[0, 0])
        # plt.tight_layout()
        # plt.show()

        metrics = {key: meter.global_avg
                   for key, meter in logger.meters.items()}
        return metrics
