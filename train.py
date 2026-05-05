import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torch.optim import Adam
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

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
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

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

    
    # encoder = spt.backbone.utils.vit_hf(
    #     cfg.encoder_scale,
    #     patch_size=cfg.patch_size,
    #     image_size=cfg.img_size,
    #     pretrained=False,
    #     use_mask_token=False,
    # )


    # hidden_dim = encoder.config.hidden_size
    # embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    # predictor = ARPredictor(
    #     num_frames=cfg.wm.history_size,
    #     input_dim=embed_dim,
    #     hidden_dim=hidden_dim,
    #     output_dim=hidden_dim,
    #     **cfg.predictor,
    # )

    # action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    # projector = MLP(
    #     input_dim=hidden_dim,
    #     output_dim=embed_dim,
    #     hidden_dim=2048,
    #     norm_fn=torch.nn.BatchNorm1d,
    # )

    # predictor_proj = MLP(
    #     input_dim=hidden_dim,
    #     output_dim=embed_dim,
    #     hidden_dim=2048,
    #     norm_fn=torch.nn.BatchNorm1d,
    # )

    world_model = JEPA(
        raw_encoders=[vision_encoder, proprio_encoder],
        encoder = state_encoder,
        decoder = action_decoder,
        predictor = predictor,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs)
    )

    world_model = world_model.to(device)
    world_model.train()
    
    optimizer = Adam(
        [
            {"params": world_model.parameters(), "lr": cfg.optimizer.lr}
        ]
    )
    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        pbar = tqdm(
            train,
            desc=f"Epoch {epoch}",
            #disable=cfg.logging.get("tqdm_silent", False),
        )

        for info in pbar:
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
                    "reg_loss": f"{total_loss['sigreg_loss']:.4f}"
                }
            )

        global_step += 1
    # data_module = spt.data.DataModule(train=train, val=val)
    # world_model = spt.Module(
    #     model = world_model,
    #     sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
    #     forward=partial(lejepa_forward, cfg=cfg),
    #     optim=optimizers,
    # )

    # ##########################
    # ##       training       ##
    # ##########################

    # run_id = cfg.get("subdir") or ""
    # run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    # logger = None
    # if cfg.wandb.enabled:
    #     logger = WandbLogger(**cfg.wandb.config)
    #     logger.log_hyperparams(OmegaConf.to_container(cfg))

    # run_dir.mkdir(parents=True, exist_ok=True)
    # with open(run_dir / "config.yaml", "w") as f:
    #     OmegaConf.save(cfg, f)

    # object_dump_callback = ModelObjectCallBack(
    #     dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    # )

    # trainer = pl.Trainer(
    #     **cfg.trainer,
    #     callbacks=[object_dump_callback],
    #     num_sanity_val_steps=1,
    #     logger=logger,
    #     enable_checkpointing=True,
    # )

    # manager = spt.Manager(
    #     trainer=trainer,
    #     module=world_model,
    #     data=data_module,
    #     ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    # )

    # manager()
    return


if __name__ == "__main__":
    run()
