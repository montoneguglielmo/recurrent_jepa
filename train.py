import os
import itertools
from functools import partial
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torch.optim import Adam
from omegaconf import OmegaConf, open_dict
import wandb

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, CNNNet, TimeWrapper, ConditionalDiffusionPredictor
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, setup_device
from tqdm import tqdm

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    device = setup_device(cfg.trainer.devices)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    col_stats = {}

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer, stats = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            col_stats[col] = stats

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # ##############################
    # ##       model / optim      ##
    # ##############################
    saved_optimizer_state = None
    if cfg.get('resume_from', None):
        print(f"Loading model from checkpoint: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            world_model = ckpt['model'].to(device)
            saved_optimizer_state = ckpt.get('optimizer', None)
        else:
            world_model = ckpt.to(device)
    else:
        vision_encoder = CNNNet(
            num_conv_layers=cfg.encoders.vision_encoder.num_conv_layers,
            input_size=cfg.img_size)
        vision_encoder = TimeWrapper(vision_encoder)

        proprio_encoder = MLP(input_dim=dataset.get_dim('proprio'),
                              hidden_dim=cfg.encoders.proprio_encoder.hidden_dim,
                              output_dim=cfg.encoders.proprio_encoder.embed_dim
                              )
        proprio_encoder = TimeWrapper(proprio_encoder)

        state_encoder = nn.GRU(
            input_size=vision_encoder.output_size + proprio_encoder.output_size,
            hidden_size=cfg.encoders.state_encoder.hidden_dim,
            batch_first=True
        )

        predictor = ConditionalDiffusionPredictor(
            embed_dim  = cfg.encoders.state_encoder.hidden_dim,
            hidden_dim = cfg.predictor.hidden_dim,
            num_steps  = cfg.predictor.num_steps,
            num_layers = cfg.predictor.num_layers,
        )

        action_decoder = MLP(input_dim=cfg.encoders.state_encoder.hidden_dim*2,
                             hidden_dim=cfg.decoder.action_decoder.hidden_dim,
                             output_dim=10
                             )
        action_decoder = TimeWrapper(action_decoder)

        world_model = JEPA(
            raw_encoders=[vision_encoder, proprio_encoder],
            encoder = state_encoder,
            decoder = action_decoder,
            predictor = predictor,
            sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
            proprio_stats=col_stats.get('proprio'),
            action_stats=col_stats.get('action'),
        )

        world_model = world_model.to(device)
    world_model.train()
    
    optimizer = Adam(
        [
            {"params": world_model.parameters(), "lr": cfg.optimizer.lr}
        ]
    )
    if saved_optimizer_state is not None:
        optimizer.load_state_dict(saved_optimizer_state)

    hydra_out = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = Path(swm.data.utils.get_cache_dir()) / "outputs" / hydra_out.parts[-2] / hydra_out.parts[-1] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {ckpt_dir}")
    best_val_loss = float("inf")

    if cfg.wandb.enabled:
        wandb_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.wandb.config, resolve=True).items()
                        if k != "log_model"}
        run_dir = ckpt_dir.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            **wandb_kwargs,
            dir=str(run_dir),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
    else:
        run = None

    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        pbar = tqdm(
            train,
            desc=f"Epoch {epoch}",
            #disable=cfg.logging.get("tqdm_silent", False),
        )

        limit = cfg.trainer.get("limit_batches", None)
        batch_iter = itertools.islice(pbar, limit) if limit else pbar
        for info in batch_iter:
            info = {k: v.to(device) for k, v in info.items()}
            pixels = info['pixels']
            action = info['action']
            proprio = info['proprio']
            info['raw_inputs'] = [pixels, proprio]
            info['target_actions'] = action
            
            optimizer.zero_grad()
            info = world_model(info)
            
            total_loss = world_model.cost(info, cfg.loss.sigreg.weight)
            
            total_loss["loss"].backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss['loss']:.4f}",
                    "pred_loss": f"{total_loss['pred_loss']:.4f}",
                    "action_loss": f"{total_loss['action_loss']:.4f}",
                    "reg_loss": f"{total_loss['sigreg_loss']:.4f}",
                    "emb_var": f"{total_loss['emb_var']:.4f}"
                }
            )
            if run is not None:
                run.log(
                    {f"train/{k}": v.item() for k, v in total_loss.items()},
                    step=global_step,
                )
            global_step += 1

        world_model.eval()
        val_totals = {}
        val_count = 0
        with torch.no_grad():
            for info in itertools.islice(val, limit) if limit else val:
                info = {k: v.to(device) for k, v in info.items()}
                info['raw_inputs'] = [info['pixels'], info['proprio']]
                info['target_actions'] = info['action']
                info = world_model(info)
                losses = world_model.cost(info, cfg.loss.sigreg.weight)
                for k, v in losses.items():
                    val_totals[k] = val_totals.get(k, 0) + v.item()
                val_count += 1
        world_model.train()
        val_avg = {k: v / val_count for k, v in val_totals.items()}
        print(
            f"[Val epoch {epoch}] "
            f"loss={val_avg['loss']:.4f}  "
            f"pred={val_avg['pred_loss']:.4f}  "
            f"reg={val_avg['sigreg_loss']:.4f}  "
            f"emb_var={val_avg['emb_var']:.4f}"
        )
        if run is not None:
            run.log(
                {f"val/{k}": v for k, v in val_avg.items()},
                step=global_step,
            )

        torch.save(
            {'model': world_model, 'optimizer': optimizer.state_dict()},
            ckpt_dir / f"ckpt_epoch_{epoch:04d}_object.ckpt",
        )
        if val_avg["loss"] < best_val_loss:
            best_val_loss = val_avg["loss"]
            torch.save(world_model, ckpt_dir / "best_object.ckpt")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    run()
