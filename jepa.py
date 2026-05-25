from collections import deque

from torch import nn
import torch
from stable_worldmodel.policy import BasePolicy

class JEPA(nn.Module, BasePolicy):

    def __init__(
        self, 
        raw_encoders, 
        encoder, 
        predictor=None, 
        decoder=None, 
        sigreg=None, 
        classifiers=None, 
        proprio_stats=None, 
        action_stats=None, 
        train_action_decoder=False):
        
        super().__init__()
        self.raw_encoders = nn.ModuleList(raw_encoders)
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.sigreg = sigreg
        self.classifiers = nn.ModuleList(classifiers) if classifiers else None
        self.train_action_decoder = train_action_decoder
        
        self._action_buffer = deque()

        self.register_buffer('proprio_mean', proprio_stats['mean'] if proprio_stats else None)
        self.register_buffer('proprio_std',  proprio_stats['std']  if proprio_stats else None)
        self.register_buffer('action_mean',  action_stats['mean']  if action_stats  else None)
        self.register_buffer('action_std',   action_stats['std']   if action_stats  else None)
    
    def forward(self, info):
        raw_inputs = info['raw_inputs']
        encoder_input = []
        for raw_input, raw_encoder in zip(raw_inputs, self.raw_encoders):
            encoder_input.append(raw_encoder(raw_input))
        encoder_input = torch.cat(encoder_input, axis=2)
        info['embeddings'] = self.encoder(encoder_input)
        info['pred_embeddings'] = self.predictor(info['embeddings'])
        
        # info['embeddings'] BxTxD
        if self.train_action_decoder:
            decoder_input = torch.cat([info['embeddings'][:, :-1], info['embeddings'][:, 1:]], dim=2)
            info['action'] = self.decoder(decoder_input)
        return info

    def cost(self, info, lambd, classifiers_targets=None):
        emb = info['pred_embeddings'][:,:-1]
        tgt_emb = info['embeddings'][:,1:]

        if self.train_action_decoder:
            act = info['action']
            tgt_act = info['target_actions'][:,:-1]

        # LeWM loss
        output = {}
        output["pred_loss"] = (emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["emb_var"] = info['embeddings'].var(dim=0).mean()

        if self.train_action_decoder:
            output["action_loss"] = (act - tgt_act).pow(2).mean()

        classifier_loss = torch.tensor(0.0, device=emb.device)
        if self.classifiers is not None and classifiers_targets is not None:
            emb_cls = info['embeddings'].detach()  # BxTxD — gradients stop here
            for classifier, target in zip(self.classifiers, classifiers_targets):
                classifier_loss = classifier_loss + (target - classifier(emb_cls)).pow(2).mean()
        output["classifier_loss"] = classifier_loss

        if self.train_action_decoder:
            output["loss"] = output["loss"] = output["pred_loss"] + output["action_loss"] + lambd * output["sigreg_loss"] + classifier_loss
        else:
            output["loss"] = output["pred_loss"]  + lambd * output["sigreg_loss"] + classifier_loss
        return output
    
    def _encode(self, raw_inputs):
        encoder_input = []
        for raw_input, raw_encoder in zip(raw_inputs, self.raw_encoders):
            encoder_input.append(raw_encoder(raw_input))
        encoder_input = torch.cat(encoder_input, axis=2)
        emb = self.encoder(encoder_input)
        return emb
        
    
    def set_normalization_stats(self, proprio_stats, action_stats):
        device = next(self.parameters()).device
        self.proprio_mean = proprio_stats['mean'].to(device)
        self.proprio_std  = proprio_stats['std'].to(device)
        self.action_mean  = action_stats['mean'].to(device)
        self.action_std   = action_stats['std'].to(device)

    def _preprocess_proprio(self, x):
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        x = x.float().to(device)
        if self.proprio_mean is not None:
            x = (x - self.proprio_mean) / self.proprio_std
        return x

    def _postprocess_action(self, action):
        if self.action_mean is not None:
            action = action * self.action_std + self.action_mean
        return action

    def _preprocess_pixels(self, px):
        """numpy (B, T, H, W, C) uint8 -> tensor (B, T, C, H, W) float, ImageNet-normalized."""
        device = next(self.parameters()).device
        if not isinstance(px, torch.Tensor):
            px = torch.from_numpy(px)
        px = px.float() / 255.0
        px = px.permute(0, 1, 4, 2, 3).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        return (px - mean) / std

    @torch.no_grad()
    def get_action(self, info, n_samples=10, n_steps=5):
        
        if not getattr(self, '_action_buffer', None):
            self._action_buffer = deque()
            pixels = self._preprocess_pixels(info['pixels'])
            goal_pixels = self._preprocess_pixels(info['goal'])
            proprio = self._preprocess_proprio(info['proprio'])
            goal_proprio = self._preprocess_proprio(info['goal_proprio'])

            current_status = [pixels, proprio]
            goal_status = [goal_pixels, goal_proprio]

            current_status_enc = self._encode(current_status)
            goal_enc = self._encode(goal_status)

            # Initial n_samples predictions: B x n_samples x D
            next_status = torch.cat(
                [self.predictor(current_status_enc) for _ in range(n_samples)], dim=1
            )
            first_step = next_status  # save for later selection

            # Roll forward and collect all steps including current state
            all_steps = [current_status_enc.expand_as(next_status)]  # B x n_samples x D
            all_steps.append(next_status)
            for _ in range(n_steps - 1):
                next_status = self.predictor(next_status)
                all_steps.append(next_status)

            # B x n_samples x (1 + n_steps) x D
            all_steps = torch.stack(all_steps, dim=2)

            # goal_enc: B x 1 x 1 x D for broadcasting
            goal_expanded = goal_enc.unsqueeze(2)  # adjust dims if goal_enc is B x D
            if goal_expanded.dim() == 3:
                goal_expanded = goal_expanded.unsqueeze(1)

            # MSE across all steps per sample, take the min step per sample
            mse = (all_steps - goal_expanded).pow(2).mean(dim=-1)  # B x n_samples x (1 + n_steps)
            min_mse_per_sample = mse.min(dim=2).values              # B x n_samples

            # Best sample per batch element
            index_min = min_mse_per_sample.argmin(dim=1)  # (B,)

            # Gather the first-step prediction for the best sample
            batch_index = torch.arange(index_min.shape[0])
            best_next_status = first_step[batch_index, index_min, :].unsqueeze(1)

            decoder_input = torch.cat([current_status_enc, best_next_status], dim=2)
            #decoder_input = torch.cat([current_status_enc, goal_enc], dim=2)
            actions = self.decoder(decoder_input)
            for t in range(5):
                self._action_buffer.append(actions[:, 0, 2*t:2*t+2])
        return self._postprocess_action(self._action_buffer.popleft())
    
    def reset(self):
        self._action_buffer = deque()
        
        

            
        