"""
Author: HuynhVanThong & Nguyen Hong Hai
Department of AI Convergence, Chonnam Natl. Univ.
"""
import os
import pathlib

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, StochasticWeightAveraging, \
    BasePredictionWriter, LearningRateMonitor
from core.callbacks import MultiStageABAW3

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from core import config, ABAW3Model, ABAW3DataModule
from core.config import cfg
from core.io import pathmgr

from datetime import datetime
from tqdm import tqdm
import wandb

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str = 'epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


if __name__ == '__main__':
    config.load_cfg_fom_args("ABAW 2022")
    config.assert_and_infer_cfg()
    cfg.freeze()

    pl.seed_everything(cfg.RNG_SEED)

    pathmgr.mkdirs(cfg.OUT_DIR)
    n_kfold = cfg.N_KFOLD
    if n_kfold < 0:
        # run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_version = cfg.MODEL_TYPE
    else:
        run_version = 'fold{}'.format(n_kfold)


    if cfg.LOGGER == 'wandb':
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, '{}_{}'.format(cfg.TASK, run_version))
    else:
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, cfg.TASK, run_version)
    pathmgr.mkdirs(cfg_file_dir)
    cfg_file = config.dump_cfg(cfg_file_dir)

    if cfg.LOGGER == 'TensorBoard':
        logger = TensorBoardLogger(cfg.OUT_DIR, name=cfg.TASK, version=run_version)
        output_dir = logger.log_dir
    elif cfg.LOGGER == 'wandb':
        logger = WandbLogger(project='Affwild2-ABAW3',
                             save_dir=cfg.OUT_DIR,
                             name='{}_{}'.format(cfg.TASK, run_version),
                             offline=False,
                             notes=f"{cfg.MODEL.BACKBONE} {'kfold{}'.format(str(n_kfold)) + cfg.NOTE}"
                             )
        output_dir = cfg_file_dir

    else:
        raise ValueError('Do not implement with {} logger yet.'.format(cfg.LOGGER))

    if cfg.TEST.WEIGHTS != '':
        result_dir = '/'.join(cfg.TEST.WEIGHTS.split('/')[:-1])
    else:
        result_dir = ''

    print('Working on Task: ', cfg.TASK)
    print(cfg.MODEL.BACKBONE, ' unfreeze: ', cfg.MODEL.BACKBONE_FREEZE)
    max_epochs = cfg.OPTIM.MAX_EPOCH if cfg.TEST.WEIGHTS == '' else 1

    if n_kfold > -1:
        all_abaw3_dataset = ABAW3DataModule()
        all_abaw3_dataset.setup()

        abaw3_dataset = ABAW3DataModule()
        abaw3_dataset.setup(kfold=n_kfold)
        abaw3_model = ABAW3Model()
    else:
        abaw3_dataset = ABAW3DataModule()
        abaw3_dataset.setup(kfold=n_kfold)
        abaw3_model = ABAW3Model()

    fast_dev_run = False
    richProgressBarTheme = RichProgressBarTheme(description="blue", progress_bar="green1",
                                                progress_bar_finished="green1")

    # backbone_finetunne = MultiStageABAW3(unfreeze_temporal_at_epoch=3, temporal_initial_ratio_lr=0.1,
    #                                         should_align=True, initial_denom_lr=10, train_bn=True)
    ckpt_cb = ModelCheckpoint(monitor='val_metric', mode="max", save_top_k=1, save_last=True)
    trainer_callbacks = [ckpt_cb,
                        PredictionWriter(output_dir=output_dir, write_interval='epoch'),
                        LearningRateMonitor(logging_interval=None)
                        ]
    if cfg.LOGGER in ['TensorBoard', ]:
        trainer_callbacks.append(RichProgressBar(refresh_rate_per_second=1, theme=richProgressBarTheme, leave=True))
        trainer_callbacks.append(RichModelSummary())

    if cfg.OPTIM.USE_SWA:
        swa_callbacks = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None)
        trainer_callbacks.append(swa_callbacks)

    trainer = Trainer(gpus=1, fast_dev_run=fast_dev_run, accumulate_grad_batches=cfg.TRAIN.ACCUM_GRAD_BATCHES,
                      max_epochs=max_epochs, deterministic=True, callbacks=trainer_callbacks, enable_model_summary=False,
                      num_sanity_val_steps=0, enable_progress_bar=True, logger=logger,
                      gradient_clip_val=0.,
                      limit_train_batches=cfg.TRAIN.LIMIT_TRAIN_BATCHES, limit_val_batches=1.0,
                      precision=32 // (cfg.TRAIN.MIXED_PRECISION + 1),
                      #auto_lr_find=True, #auto_scale_batch_size=None,
                      )

    # trainer.tune(abaw3_model, datamodule=abaw3_dataset, lr_find_kwargs={})
    if cfg.TEST_ONLY != 'none':
        print('Testing only. Loading checkpoint: ', cfg.TEST_ONLY)
        if not os.path.isfile(cfg.TEST_ONLY):
            raise ValueError('Could not find {}'.format(cfg.TEST_ONLY))
        # Load pretrained weights
        pretrained_state_dict = torch.load(cfg.TEST_ONLY)['state_dict']
        abaw3_model.load_state_dict(pretrained_state_dict, strict=True)

        # Prepare test set
        abaw3_dataset.setup(stage='test')
        # Generate prediction
        print('Generate prediction')
        trainer.test(dataloaders=all_abaw3_dataset.test_dataloader(),  ckpt_path=None, model=abaw3_model)
        trainer.test(dataloaders=all_abaw3_dataset.val_dataloader(), ckpt_path=None, model=abaw3_model)
        # trainer.predict(dataloaders=abaw3_dataset.test_dataloader(), ckpt_path=None, model=abaw3_model)
        print('Testing finished.')
    else:
        trainer.fit(abaw3_model, datamodule=abaw3_dataset)

        if cfg.LOGGER == 'wandb':
            wandb.run.log_code("./core/", include_fn=lambda path: path.endswith(".py") or path.endswith('.yaml'), )
            wandb.run.log_code("./scripts/", include_fn=lambda path: path.endswith('.sh'), )

        print('Pass with best val_metric: {}. Generating the prediction ...'.format(ckpt_cb.best_model_score))

        if cfg.OPTIM.USE_SWA:
            print('Evaluating with SWA')
            trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path=None, model=abaw3_model)
            trainer.save_checkpoint(ckpt_cb.last_model_path.replace('.ckpt', '_swa.ckpt'))
        if n_kfold < 0:
            trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path='best')
            trainer.test(dataloaders=abaw3_dataset.test_dataloader(), ckpt_path='best')
        else:
            trainer.test(dataloaders=abaw3_dataset.train_dataloader(), ckpt_path='best')
            trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path='best')
            # trainer.predict(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path='best')
            # # print("_____ Validation Set _____")
            trainer.test(dataloaders=all_abaw3_dataset.val_dataloader(), ckpt_path='best')
            trainer.test(dataloaders=all_abaw3_dataset.test_dataloader(), ckpt_path='best')
