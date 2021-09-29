"""
Code template for training neural networks with PyTorch Lightning.

**Features**
- PyTorch Lightning for code organization with datamodules provided by PyTorch Lightning Bolts.
- Experiment configuration handled by Hydra structured configs. This allows for runtime
  config validation and auto-complete support in IDEs.
- Weights & Biases (wandb.ai) logger for metric visualization and checkpoints saving
  as wandb artifacts.
- Console logging and printing with `rich` formatting.
- Typing hints for most of the source code.

**See**
- https://pytorch-lightning.readthedocs.io/en/latest/
- https://lightning-bolts.readthedocs.io/en/latest/
- https://hydra.cc/docs/next/tutorials/intro/
- https://docs.wandb.ai/
- https://github.com/willmcgugan/rich
"""

from __future__ import annotations

import os
from typing import Any, cast

import hydra
import pytorch_lightning as pl
import setproctitle  # type: ignore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from ray.tune import SyncConfig
from wandb.sdk.wandb_run import Run
from zzsn2021.systems.SpatioTemporalRegressor import SpatioTemporalRegressor
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from zzsn2021.configs import Config, get_tags, register_configs
from zzsn2021.utils.callbacks import CustomCheckpointer, get_resume_checkpoint
from zzsn2021.utils.logging import log
from zzsn2021.utils.rundir import setup_rundir
from zzsn2021.helpers.DataViewer import DataViewer

wandb_logger: WandbLogger
# os.environ['WANDB_MODE'] = 'dryrun'


def train_covid_for_tune(tune_config, cfg: Config, model, num_epochs=10, num_gpus=0):
    metrics = {"loss": "ptl/val_loss"}
    # when assigning the value directly there is an error " Value '(32,)' could not be converted to Integer"
    cfg.experiment.__setattr__('hidden_channels', tune_config['hidden_channels']),
    cfg.experiment.__setattr__('output_channels', tune_config['output_channels']),
    cfg.experiment.__setattr__('temporal_conv_output_channels', tune_config['temporal_conv_output_channels']),
    # cfg.optim.optimizer.__setattr__('lr', config['lr'])

    dm = instantiate(
        cfg.experiment.datamodule,
        val_split=cfg.experiment.val_split,
        node_features=cfg.experiment.node_features,
        shuffle=cfg.experiment.shuffle,
        batch_size=cfg.experiment.batch_size,
        full_dataset=cfg.experiment.full_dataset,
        horizon=cfg.experiment.horizon)
    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        max_epochs=num_epochs,
        gpus=num_gpus,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
        checkpoint_callback=False
    )
    trainer.fit(model, dm)

def run_tune(cfg: Config):
    config = {
        "hidden_channels": tune.grid_search([8, 16, 32]),
        "output_channels": tune.grid_search([32, 64, 128]),
        "temporal_conv_output_channels": tune.grid_search([8, 16, 32])
    }

    # Create main system (system = models + training regime)
    system = SpatioTemporalRegressor(cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)

    trainable = tune.with_parameters(
        train_covid_for_tune,
        cfg=cfg,
        model=system,
        num_epochs=cfg.lightning.max_epochs,
        num_gpus=cfg.lightning.gpus)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": cfg.lightning.gpus
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=2,
        sync_config=SyncConfig(sync_to_driver=False),
        name="tune_covid")

    print(analysis.best_config)


def run_training(cfg: Config):
    RUN_NAME = os.getenv('RUN_NAME')
    log.info(f'[bold yellow]\\[init] Run name --> {RUN_NAME}')

    run: Run = wandb_logger.experiment  # type: ignore

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    run.tags = tags
    run.notes = str(cfg.notes)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    resume_path = get_resume_checkpoint(cfg, wandb_logger)
    if resume_path is not None:
        log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

    callbacks: list[Any] = []

    checkpointer = CustomCheckpointer(
        period=1,  # checkpointing interval in epochs, but still will save only on validation epoch
        dirpath='checkpoints',
        filename='{epoch}',
    )
    if cfg.experiment.save_checkpoints:
        callbacks.append(checkpointer)

    log.info(f'[bold white]Overriding cfg.lightning settings with derived values:')
    log.info(f' >>> resume_from_checkpoint = {resume_path}')
    log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}\n')

    # Create main system (system = models + training regime)
    system = SpatioTemporalRegressor(cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)
    # Prepare data using datamodules
    datamodule: LightningDataModule = instantiate(
        cfg.experiment.datamodule,
        val_split=cfg.experiment.val_split,
        batch_size=cfg.experiment.batch_size,
        node_features=cfg.experiment.node_features,
        shuffle=cfg.experiment.shuffle,
        full_dataset=cfg.experiment.full_dataset,
        horizon=cfg.experiment.horizon
    )

    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        resume_from_checkpoint=resume_path,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
    )

    trainer.fit(system, datamodule=datamodule)
    trainer.test(system, datamodule=datamodule)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


def view_results(cfg: Config):
    VIEW_RUN_NAME = os.getenv('VIEW_RUN_NAME')

    log.info(f'[bold yellow]\\[init] View run name --> {VIEW_RUN_NAME}')

    resume_path = cfg.experiment.resume_checkpoint

    # Create main system (system = models + training regime)
    system = SpatioTemporalRegressor.load_from_checkpoint(resume_path, cfg=cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)
    # Prepare data using datamodules
    datamodule: LightningDataModule = instantiate(
        cfg.experiment.datamodule,
        val_split=cfg.experiment.val_split,
        batch_size=cfg.experiment.batch_size,
        node_features=cfg.experiment.node_features,
        shuffle=cfg.experiment.shuffle,
        full_dataset=cfg.experiment.full_dataset,
        horizon=cfg.experiment.horizon
    )

    trainer: pl.Trainer = instantiate(
        cfg.lightning
    )

    # system = SpatioTemporalRegressor.load_from_checkpoint()
    trainer.test(system, datamodule=datamodule)
    results = system.test_results
    mean = datamodule.dataset_test.mean
    std = datamodule.dataset_test.std

    data_viewer = DataViewer()
    data_viewer.import_data(datamodule.node_ids, results['output'], results['labels'], mean, std)
    data_viewer.view_plot()

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')


@hydra.main(config_path='configs', config_name='default')
def main(cfg: Config) -> None:
    """
    Main training dispatcher.

    Uses PyTorch Lightning with datamodules provided by PyTorch Lightning Bolts.
    Experiment configuration is handled by Hydra with StructuredConfigs, which allow for config
    validation and provide auto-complete support in IDEs.

    """
    RUN_MODE = os.getenv('RUN_MODE')

    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

    if RUN_MODE == 'tune':
        run_tune(cfg)
    elif RUN_MODE == 'view':
        view_results(cfg)
    else:
        run_training(cfg)


if __name__ == '__main__':
    setup_rundir()

    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT'),
        entity=os.getenv('WANDB_ENTITY'),
        name=os.getenv('RUN_NAME'),
        save_dir=os.getenv('RUN_DIR'),
    )

    # Init logger from source dir (code base) before switching to run dir (results)
    wandb_logger.experiment  # type: ignore

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()
