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
        #output["loss"] = output["pred_loss"] + output["action_loss"] + lambd * output["sigreg_loss"]
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        return output
    
    def _encode(self, info):
        raw_inputs = info['raw_inputs']
        encoder_input = []
        for raw_input, raw_encoder in zip(raw_inputs, self.raw_encoders):
            encoder_input.append(raw_encoder(raw_input))
        encoder_input = torch.cat(encoder_input, axis=2)
        emb, _ = self.encoder(encoder_input)
        return emb
        
    
    def get_action(self, info):
        print('Info inside the get_action')
        print(info.keys())
        pixels = info['pixels']
        proprio = info['proprio']
        goal_pixels = info['goal']['pixels']
        
        current_status = [pixels, proprio]
        goal = [goal_pixels, ]
        
        current_status_enc = self._encode(current_status)
        goal_enc = self._encode(goal)
        
        #Sample from predictor
        status_list = []
        status_list.append(current_status_enc)
        for step in cfg.n_steps:
            next_status = self.predictor(current_status_enc)
            status_list.append(next_status)
            current_status_enc = next_status
        
        index_min = F.mse_loss(next_status, goal_enc).argmin()
        best_next_status = status_list[1][index_min]
        decoder_input = torch.cat([current_status_enc, best_next_status], dim=2)
        return self.decoder(decoder_input)
        
        

            
        