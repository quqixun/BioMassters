import os
import torch

from ..models import define_model
from accelerate import Accelerator
from .scheduler import WarmupCosineAnnealingLR


class BMBaseTrainer(object):

    def __init__(self, configs, exp_dir, train_loader, val_loader, resume=False):

        # creates dirs for saving outputs
        self.logs_dir  = os.path.join(exp_dir, 'logs')
        self.ckpts_dir = os.path.join(exp_dir, 'ckpts')
        os.makedirs(self.logs_dir,  exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)

        # trainer parameters
        self.start_epoch     = 1
        self.epochs          = configs.trainer.epochs
        self.ckpt_freq       = configs.trainer.ckpt_freq
        self.accum_iter      = configs.trainer.accum_iter
        self.print_freq      = configs.trainer.print_freq
        self.mixed_precision = configs.trainer.mixed_precision

        # initializes accelerator with log tracking
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.accum_iter,
            log_with='all', logging_dir=self.logs_dir
        )
        self.device = self.accelerator.device

        # instantiates model
        model = define_model(configs.model)
        model = model.to(self.device)

        # instantiates loss
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.L1Loss()

        # instantiates optimizer
        lr           = configs.optimizer.lr
        betas        = configs.optimizer.betas
        amsgrad      = configs.optimizer.amsgrad
        weight_decay = configs.optimizer.weight_decay
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas,
            amsgrad=amsgrad, weight_decay=weight_decay
        )

        # instantiates scheduler
        min_lr = configs.scheduler.min_lr
        warmup = configs.scheduler.warmup
        scheduler = WarmupCosineAnnealingLR(
            optimizer, max_lr=lr, min_lr=min_lr,
            total=self.epochs, warmup=warmup
        )
        
        # prepare everything for accelerator
        self.train_loader, self.val_loader, self.model, \
        self.optimizer, self.scheduler = self.accelerator.prepare(
            train_loader, val_loader, model, optimizer, scheduler
        )

        # loads checkpoint
        self._load_checkpoint(resume)

    def _load_checkpoint(self, resume):
        if not resume:
            return

        # find checkpoint from ckpt_dir
        ckpt_list = os.listdir(self.ckpts_dir)
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt_target = ckpt_list[-1]
        else:
            self.accelerator.print('\n>>> No checkpoint was found, ignore\n')
            return

        try:
            self.accelerator.print(f'>>> Resumed from: {ckpt_target}\n')
            self.accelerator.load_state(ckpt_target)
            ckpt_name = os.path.basename(ckpt_target)
            self.start_epoch = int(ckpt_name.split('_')[-1]) + 1
        except Exception:
            self.accelerator.print(f'>>> Faild to resume ckpt from: {ckpt_target}')

    def _save_checkpoint(self, epoch):
        ckpt_name = f'epoch_{epoch:06d}'
        ckpt_dir  = os.path.join(self.ckpts_dir, ckpt_name)
        self.accelerator.save_state(ckpt_dir)
        self.accelerator.print(f'>>> Save ckpt to: {ckpt_dir}')

    # def _save_model(self, model, model_name):
    #     model_path = os.path.join(self.exp_dir, f'{model_name}.pth')
    #     torch.save(model.state_dict(), model_path)

    # def _save_logs(self, epoch, train_metrics, val_metrics=None):
    #     log_stats = {
    #         'epoch': epoch,
    #         **{f'train_{k}': v for k, v in train_metrics.items()},
    #         **{f'val_{k}': v for k, v in val_metrics.items()}
    #     }
    #     df = pd.DataFrame(log_stats, index=[0])

    #     if not os.path.isfile(self.log_path):
    #         df.to_csv(self.log_path, index=False)
    #     else:
    #         df.to_csv(self.log_path, index=False, mode='a', header=False)

    # def _save_metrics(self, val_metrics, metrics_type):
    #     assert metrics_type in ['dynamic', 'static']

    #     df = pd.DataFrame(val_metrics, index=[0])

    #     if metrics_type == 'dynamic':
    #         metrics_path = self.Dmetrics_path
    #     else:  # metrics_type == 'static'
    #         metrics_path = self.Smetrics_path
    #     df.to_csv(metrics_path, index=False)
