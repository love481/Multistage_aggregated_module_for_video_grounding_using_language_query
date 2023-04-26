import copy
import torch
from torch import nn
import numpy as np
import math
from torch.functional import F
from .model_utils import mySequential,pos_embedding,BertLayerNorm
from .transformer import TransformerBlock
from models.frame_modules.frame_pool import FrameAvgPool
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=116): ## d_hid defines the size of embedding
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
#used for the initialization of weight parameters, good to manually initialize the network
class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented

class TAN(BaseModel):
    def __init__(self,config):
        super(TAN,self).__init__()
        self.config=config
        self.device = "cuda"
        self.frame_layer = FrameAvgPool()
        self.tblocks = []

        self.project_vid = nn.Linear(config.visual_embedding_size, config.hidden_size)
        self.project_text = nn.Linear(config.text_embedding_size,  config.hidden_size)
        self.mask_embedding = nn.Embedding(1, config.hidden_size)
        self.text_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.text_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.postion_encoding = pos_embedding(194,config.hidden_size) # for tacos
        self.postion_encoding = pos_embedding(116, config.hidden_size) # for activity net

        #iou mask map for activity net
        self.iou_mask_map = torch.zeros(33,33).float()
        for i in range(0,32,1):
            self.iou_mask_map[i,i+1:min(i+17,33)] = 1.
        for i in range(0,32-16,2):
            self.iou_mask_map[i,range(18+i,33,2)] = 1.

        # transformer layer
        self.tblock1 = TransformerBlock(config)
        for i in range(config.num_hidden_layers-1):
            self.tblocks.append(TransformerBlock(config))
        self.tblocks = mySequential(*self.tblocks)
        self.apply(self.init_weights)

        # m, s, e representation layers
        # self.mlp_s = torch.nn.Sequential(
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #         torch.nn.Linear(config.hidden_size, config.hidden_size),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #     )
        # self.mlp_m = torch.nn.Sequential(
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #         torch.nn.Linear(config.hidden_size,config.hidden_size),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #     )
        # self.mlp_e =torch.nn.Sequential(
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #         torch.nn.Linear(config.hidden_size,config.hidden_size),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
        #     )

        self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size,config.hidden_size*3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
            )

        ##define layers to give scores to each starting , middle and ending representations
        self.mlp_score_s =  torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size, 1)
            )
        self.mlp_score_m = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size, 1)
            )
        self.mlp_score_e = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size,config.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size, 1)
            )

        self.token_prediction_layer =  torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size,config.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size, config.vocab_size)
            )

        self.mlp =torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size*3,config.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.hidden_dropout_prob, inplace=False),
                torch.nn.Linear(config.hidden_size, 3)
            )
        self.init_weight()
    def init_weight(self):
        for m in self.token_prediction_layer.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        # for m in self.mlp_s.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.constant_(m.bias, 0)
        # for m in self.mlp_m.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.constant_(m.bias, 0)
        # for m in self.mlp_e.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
		
    def forward(self, inps, attention_masks, token_masks, feats, batch=None, output_attention_probs=False):
        batch_size = feats.shape[0]
        assert feats.shape == (batch_size, feats.shape[1],self.config.visual_embedding_size),str(feats.shape)
        assert inps.shape == (batch_size, inps.shape[1],self.config.text_embedding_size),str(inps.shape)

        #pass through frame layer, converts feats from 256 to 129 length
        feats = self.frame_layer(feats.transpose(1,2))
        feats = feats.transpose(1,2)
        # print(feats.shape)

        attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
        ##for unpadded space and 0 for padded space i.e (1-(11110000)) 00000000 for text and visual sequence
        attention_masks=torch.cat(((1-attention_masks).bool(),torch.zeros((batch_size,1, 1, feats.shape[1])).bool().to(self.device)),-1)
        #assert attention_masks.shape == (batch_size,1, 1, feats.shape[1]+inps.shape[1]),str(attention_masks.shape)
     

        #project both video and text features to same dimension, 512
        feats = self.project_vid(feats)
        inps = self.project_text(inps)
        # print(feats.shape)
        
        # pass masked tokens throuh mask_embedding
        if self.training:
            _zero_id = torch.zeros(inps.shape[:2], dtype=torch.long, device = inps.device)
            inps[token_masks>0] = self.mask_embedding(_zero_id)[token_masks>0]


        # add position embedding original(b, (129+12), 512).(b, 129, 512), (b, 12, 512)
        ## this positional embedding is really good
        embeddings = torch.cat([feats, inps], dim=1) 
        embeddings =self.postion_encoding(embeddings)  ## demension of position embedding is same as feature dimensions for both visual and text
        feats, inps = torch.split(embeddings, [feats.size(1),inps.size(1)], 1)

        ##Need to layer_normalized to normalize the distribution of intermediate layers for smooth gardients and faster trainings
        ##drop out for reducing overfittings
        feats=self.visual_embedding_LayerNorm(feats)
        feats=self.visual_embedding_dropout(feats)
        inps=self.text_embedding_LayerNorm(inps)
        inps=self.text_embedding_dropout(inps)
        
        feats, inps, attention_masks = self.tblock1(feats, inps, attention_masks)
        feats , inps, _ = self.tblocks((feats, inps, attention_masks)) ##shape : (batch_size * 129 * 512),(batch_size * 12 * 512)

        #calculate token_predictions for mlm
        token_predictions = self.token_prediction_layer(inps)


        # starting_emb = self.mlp_s(feats) ##(batch_size * 129 * aggregated_units)
        # middle_emb = self.mlp_m(feats) ##(batch_size * 129 * aggregated_units)
        # ending_emb = self.mlp_e(feats) ##(batch_size * 129 * aggregated_units)
        feats = self.final_mlp_2(feats)
        starting_emb, ending_emb, middle_emb = torch.split(feats,self.config.hidden_size, dim=-1) 
        # print("there0")
        T = starting_emb.size(1) #129
        s_idx = torch.arange(T, device=self.device)#129
        e_idx = torch.arange(T, device=self.device)#129
        c_point = middle_emb[:,(0.5*(s_idx[:,None] + e_idx[None,:])).long().flatten(),:].view(middle_emb.size(0),T,T,middle_emb.size(-1))##(batch_size *129*129*512)
        s_c_e_points = torch.cat((starting_emb[:,:,None,:].repeat(1,1,T,1), c_point, ending_emb[:,None,:,:].repeat(1,T,1,1)), -1)##(batch_size*129*129*1536)
        logits_iou = self.mlp(s_c_e_points).permute(0,3,1,2).contiguous()##(batch_size * 3 * 129 * 129)
        
        ##find the score of how much they represent the starting middle and ending stages
        starting_score =  self.mlp_score_s(starting_emb) ##(batch_size * 129 * 1)
        middle_score = self.mlp_score_m(middle_emb)  ##(batch_size * 129 * 1)
        ending_score = self.mlp_score_e(ending_emb)  ##(batch_size * 129 * 1)
        # print("there1")

        assert starting_score.size() == (batch_size,feats.shape[1],1) , "starting score doesnot match"
        assert middle_score.size() == (batch_size,feats.shape[1],1) , "model score doesnot match"
        assert ending_score.size() == (batch_size,feats.shape[1],1) , "ending score doesnot match"
        assert starting_emb.size() == (batch_size,feats.shape[1],self.config.hidden_size), "starting  embdding size does not match"
        assert middle_emb.size() == (batch_size,feats.shape[1],self.config.hidden_size),"middle embedding size does not match"
        assert ending_emb.size() == (batch_size,feats.shape[1],self.config.hidden_size),"ending embedding size doesnot match"
        
        
        starting_score = starting_score.transpose(1,2)
        ending_score = ending_score.transpose(1,2)
        middle_score = middle_score.transpose(1,2)
        logits_visual = torch.cat((starting_score, ending_score, middle_score), 1)
        # print("ending")
        #return (batch_size, 12, self.self.vocab_size), (batch_size, 129, 1), proposals(batch_size, 2548, 3), proposals_indices(2548, 2)
        #return mlm_requirements, bce_requirements, regression_loss & match_loss 
        return token_predictions, logits_visual, logits_iou,self.iou_mask_map.clone().detach()



    
