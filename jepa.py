from torch import nn
import torch
from stable_worldmodel.policy import BasePolicy

class JEPA(nn.Module, BasePolicy):

    def __init__(self, raw_encoders, encoder, predictor=None, decoder=None, sigreg=None, classifiers=None):
        super().__init__()
        self.raw_encoders = nn.ModuleList(raw_encoders)
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.sigreg = sigreg
    
    def forward(self, info):
        raw_inputs = info['raw_inputs']
        encoder_input = []
        for raw_input, raw_encoder in zip(raw_inputs, self.raw_encoders):
            encoder_input.append(raw_encoder(raw_input))
        encoder_input = torch.cat(encoder_input, axis=2)
        info['embeddings'], _ = self.encoder(encoder_input)
        info['pred_embeddings'] = self.predictor(info['embeddings'])
        
        # info['embeddings'] BxTxD
        decoder_input = torch.cat([info['embeddings'][:, :-1], info['embeddings'][:, 1:]], dim=2)
        info['action'] = self.decoder(decoder_input)
        return info

    def cost(self, info, lambd):    
        emb = info['pred_embeddings']
        tgt_emb = info['embeddings']
        
        act = info['action']
        tgt_act = info['target_actions'][:,:-1]
    
        # LeWM loss
        output = {}
        output["pred_loss"] = (emb - tgt_emb).pow(2).mean()
        output["action_loss"] = (act - tgt_act).pow(2).mean()
        output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
        output["emb_var"] = info['embeddings'].var(dim=0).mean()
        output["loss"] = output["pred_loss"] + output["action_loss"] + lambd * output["sigreg_loss"]
        #output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        return output
    
    def _encode(self, raw_inputs):
        encoder_input = []
        for raw_input, raw_encoder in zip(raw_inputs, self.raw_encoders):
            encoder_input.append(raw_encoder(raw_input))
        encoder_input = torch.cat(encoder_input, axis=2)
        emb, _ = self.encoder(encoder_input)
        return emb
        
    
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
    def get_action(self, info, n_samples=30, n_steps=30):
        device = next(self.parameters()).device
        pixels = self._preprocess_pixels(info['pixels'])
        goal_pixels = self._preprocess_pixels(info['goal'])

        proprio = info['proprio']
        goal_proprio = info['goal_proprio']
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.from_numpy(proprio)
        if not isinstance(goal_proprio, torch.Tensor):
            goal_proprio = torch.from_numpy(goal_proprio)
        proprio = proprio.float().to(device)
        goal_proprio = goal_proprio.float().to(device)

        current_status = [pixels, proprio]
        goal_status = [goal_pixels, goal_proprio]
        
        current_status_enc = self._encode(current_status)
        goal_enc = self._encode(goal_status)
        
        #print('current_status_enc:', current_status_enc.shape)
        #print('goal_enc:', goal_enc.shape)
        

        next_status = torch.cat([self.predictor(current_status_enc) for _ in range(n_samples)], dim=1)
        #print('next status samples:', next_status.shape)
        
        #Sample from predictor
        next_status_list = []
        next_status_list.append(next_status)
        for _ in range(n_steps):
            next_status = self.predictor(next_status)
            next_status_list.append(next_status)
            
        #print('next status after steps:', next_status.shape)
        #print('goal enc shape:', goal_enc.shape)
        
        # (50, 30, 400) vs (50, 1, 400) -> mse per condition per sample -> (50, 30)
        mse_per_sample = (next_status - goal_enc).pow(2).mean(dim=2)
        index_min = mse_per_sample.argmin(dim=1)  # (50,) — best sample per condition

        idx = index_min.view(-1, 1, 1).expand(-1, 1, next_status_list[0].shape[2])
        best_next_status = next_status_list[0].gather(1, idx)  # (50, 1, 400)
        #print('best_next_status shape', best_next_status.shape)
        decoder_input = torch.cat([current_status_enc, best_next_status], dim=2)  # (50, 1, 800)
        actions = self.decoder(decoder_input)
        return actions[:, 0, :2]
        
        

            
        