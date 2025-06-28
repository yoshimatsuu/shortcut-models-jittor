import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.torch_common import count_parameters, set_seed
from my_trainer import Trainer
from utils.logging import get_logger


@hydra.main(version_base=None, config_path='../configs/', config_name="default.yaml")
def main(cfg: DictConfig):
    set_seed(cfg.trainer.seed)

    batch_size = cfg.trainer.batch_size
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(cfg.trainer.dataset_dir, train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, )

    # Model
    model = hydra.utils.instantiate(cfg.model)

    # Optimizer
    opt = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.parameters())
    sche = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=opt)

    output_root = os.path.join(cfg.trainer.output_dir, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(output_root)
    logger = get_logger(os.path.join(output_root, 'log.log'))

    model.train()
    num_params = count_parameters(model) / 1e6
    logger.info("=== Parameters ===")
    logger.info(f"Model Params:\t{num_params:.2f} [million]")
    logger.info("=== Dataset ===")
    logger.info(f"Batch size: {cfg.trainer.batch_size}")
    logger.info("Train data:")
    logger.info(f"Files:\t{len(train_dataset)}")
    logger.info(f"Batches:\t{len(train_dataset) // cfg.trainer.batch_size}")

    # Start training
    trainer = Trainer(
        model, opt, sche, train_dataloader,  cfg, output_root, logger, ckpt_dir=cfg.trainer.ckpt_dir)

    trainer.start_training()


if __name__ == '__main__':
    main()
