
from torch import nn
from torch.functional import F
import torch
from .model_utils import BertLayerNorm
import math

class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.emb, self.heads = int(config.hidden_size / config.num_attention_heads),config.num_attention_heads
        self.all_head_size = self.heads * self.emb
        
        #initialize num of heads, dimension of query and key(batch_size)
        self.tokeys_v_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.toqueries_v_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.tovalues_v_v = nn.Linear(config.hidden_size, self.all_head_size)

        self.tokeys_l_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.toqueries_l_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.tovalues_l_v = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.tokeys_v_l = nn.Linear(config.hidden_size, self.all_head_size)
        self.toqueries_v_l = nn.Linear(config.hidden_size, self.all_head_size)
        self.tovalues_v_l = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.tokeys_l_l = nn.Linear(config.hidden_size, self.all_head_size)
        self.toqueries_l_l = nn.Linear(config.hidden_size, self.all_head_size)
        self.tovalues_l_l = nn.Linear(config.hidden_size, self.all_head_size)
        
        #can be treated as SelfAttenton output
        self.unifyheads_v = nn.Linear(self.all_head_size, config.hidden_size)
        self.unifyheads_l = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout_v = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_l = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, feats, inps, att_mask=None):
        b, t1, all_head_size = feats.size()
        t2 = inps.shape[1]
        h = self.heads
        e = self.emb
        assert all_head_size == self.all_head_size, f'Input embedding dim must equal layer embedding dim'
        #calculate key, queries, and values (12 matrices)
        queries_v_v = self.toqueries_v_v(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))
        keys_v_v = self.tokeys_v_v(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))
        values_v_v = self.tovalues_v_v(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))

        queries_l_v = self.toqueries_l_v(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))
        keys_l_v = self.tokeys_l_v(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))
        values_l_v = self.tovalues_l_v(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))

        queries_v_l = self.toqueries_v_l(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))
        keys_v_l = self.tokeys_v_l(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))
        values_v_l = self.tovalues_v_l(feats).view(b, t1, h, e).transpose(1,2).contiguous().view(b*h, t1, e)/ (e ** (1/4))

        queries_l_l = self.toqueries_l_l(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))
        keys_l_l = self.tokeys_l_l(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))
        values_l_l = self.tovalues_l_l(inps).view(b, t2, h, e).transpose(1,2).contiguous().view(b*h, t2, e)/ (e ** (1/4))
        
        #compute scaled dot-product self-attention
        dot_v_v = torch.bmm(queries_v_v, keys_v_v.transpose(1,2))
        dot_l_v = torch.bmm(queries_l_v, keys_l_v.transpose(1,2))
        dot_v_l = torch.bmm(queries_v_l, keys_v_l.transpose(1,2))
        dot_l_l = torch.bmm(queries_l_l, keys_l_l.transpose(1,2))
        

        #assertions
        assert dot_v_v.size() == (b*h, t1, t1)
        assert dot_l_v.size() == (b*h, t1, t2)
        assert dot_v_l.size() == (b*h, t2, t1)
        assert dot_l_l.size() == (b*h, t2, t2)


        #form attention matrix from 4 smaller matrices
        att_v = torch.cat([dot_v_v, dot_l_v], 2).view(b, h, t1, t1+t2)
        att_l = torch.cat([dot_v_l, dot_l_l], 2).view(b, h, t2, t1+t2)
        # att_mask = att_mask.unsqueeze(1) ##(batch_size,1,1,141)
        att_v = att_v.masked_fill(att_mask, -1e9)  ## fill with -1e9 in place where att_mask is 1 to avoid contribution of paddings and other as in the data
        att_l = att_l.masked_fill(att_mask, -1e9)
        att_v = att_v.view(b*h, t1, t1+t2)
        att_l = att_l.view(b*h, t2, t1+t2)

        att_v = F.softmax(att_v, dim=2)
        att_l = F.softmax(att_l, dim=2)

        # softmax
        values_v = torch.cat([values_v_v, values_l_v], 1)
        values_l = torch.cat([values_v_l, values_l_l], 1)
        assert values_v.shape == (b*h, t1+t2, e)
        assert values_l.shape == (b*h,t1+t2,e)

        feats=torch.bmm(att_v, values_v)
        feats=feats.view(b, h, t1, e)
        feats=feats.transpose(1,2)
        feats=feats.contiguous()
        feats=feats.view(b, t1, h*e)
        feats = self.unifyheads_v(feats)
        feats=self.dropout_v(feats)

        inps=torch.bmm(att_l, values_l)
        inps=inps.view(b, h, t2, e)
        inps=inps.transpose(1,2)
        inps=inps.contiguous()
        inps=inps.view(b, t2, h*e)
        inps = self.unifyheads_l(inps)
        inps=self.dropout_l(inps)
        assert feats.shape == (b, t1, e*h)
        assert inps.shape == (b, t2, e*h)

        return feats, inps

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.norm1_v = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout_v = nn.Dropout(config.hidden_dropout_prob)
        self.norm2_v = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ff_v = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size,config.hidden_size)
        )

        self.norm1_l = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout_l = nn.Dropout(config.hidden_dropout_prob)
        self.norm2_l = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ff_l = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, feats, inps, attention_masks):
        attendedv, attendedl = self.attention(feats, inps, attention_masks)
    
        feats = self.norm1_v(attendedv +feats)
        fedforward = self.ff_v(feats)
        fedforward = self.dropout_v(fedforward)
        feats = self.norm2_v(fedforward+feats)
        
        inps = self.norm1_l(attendedl +inps)
        fedforward = self.ff_l(inps)
        fedforward = self.dropout_l(fedforward)
        inps = self.norm2_l(fedforward+inps)
        return (feats, inps, attention_masks)




