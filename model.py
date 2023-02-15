import torch
from torch import nn
import  torch.nn.functional as Fa
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import numpy as np
import copy
import math
from einops.layers.torch import Rearrange


class cross_code_lstm(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,192, padding_idx=padding_idx)
        #self.position_encoder = nn.Embedding(pad_size+1,64, padding_idx=padding_idx)    

        self.code_lstm  = torch.nn.LSTM(
                                input_size = 192*5,#dimension词向量维度
                                hidden_size = 192,#表示输出的特征维度，如果没有特殊变化，相当于out
                                num_layers = 2,# 表示网络的层数
                                bidirectional = True,#双向向RNN
                                batch_first = True,
                                dropout=0.2
                                )        

        #self.fc0 = nn.Sequential(
        #    nn.Linear(192*4, 128*4),
        #)
        
        #self.fc01 = nn.Sequential(
         #   nn.Linear(64*2, 32),
        #)     

    def cross_attention(self,lstm_output, h_t):
           # lstm_output [3, 10, 16]  h_t[10, 16]
           h_t = h_t.unsqueeze(0)
           # [10, 16, 1]
           h_t = h_t.permute(1, 2, 0)
           
           attn_weights = torch.bmm(lstm_output, h_t)
           attn_weights = attn_weights.permute(1, 0, 2).squeeze()
    
           # [3, 10]
           if len(attn_weights.shape)==2:
               attention = Fa.softmax(attn_weights, 1)
               attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(-1).transpose(1,0))

               
           else:
               attention = Fa.softmax(attn_weights, 0)
               attn_out = torch.matmul(lstm_output.transpose(1, 2), attention.unsqueeze(-1))
           # bmm: [10, 16, 3] [10, 3, 1]
    
           return attn_out.squeeze()

    def get_code_post_feat(self,position,sequece_token):
        
        lengths = sequece_token[:,:,0] !=self.padding_idx
        lengths = lengths.type(torch.IntTensor).sum(dim=1)
        
        index_code_feat = self.code_encoder(sequece_token)
        #index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        #index_code_feat = index_code_feat.sum(dim=-2)
        
        #index_code_feat = index_code_feat[:,:,1:,:]
        
        #index_code_feat = torch.cat([index_code_feat[:,:,0,:].unsqueeze(dim=2),index_code_feat[:,:,1:,:].sum(dim=2).unsqueeze(dim=2)],dim=2)
        
        shape = index_code_feat.shape        
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        
        index_sequece_feat = index_code_feat#+index_position_feat
        #mask = mask.repeat(1,1,4)
        index_sequece_feat = pack_padded_sequence(input=index_sequece_feat, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out,(index_sequece_feat2,_) = self.code_lstm(index_sequece_feat)
        
        
        index_sequece_feat2 = torch.cat([index_sequece_feat2[i] for i in range(4)],dim=1)
        
        #index_sequece_feat2 = self.fc0(index_sequece_feat2)
        #index_sequece_feat2 = torch.relu(index_sequece_feat2)
        
        packed_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        #packed_out = self.fc01(packed_out)        
        
        
        #自注意力
        #index_sequece_feat2 = self.cross_attention(packed_out,index_sequece_feat2) 
        
        
        return index_sequece_feat2,packed_out  


    def forward(self,position_token,index_sequece_token,postive_sequece_token,nagetive_sequece_token=None):
        
        index_sequece_feat2,index_out = self.get_code_post_feat(1,index_sequece_token)
        postive_sequece_feat2,postive_out = self.get_code_post_feat(1,postive_sequece_token) 
        
        
        #postive_sequece_feat2 = self.cross_attention(index_out,postive_sequece_feat2) 
        
        if nagetive_sequece_token!=None:
            nagetive_sequece_feat2,nagetive_out = self.get_code_post_feat(1,nagetive_sequece_token)        
            #nagetive_sequece_feat2 = self.cross_attention(index_out,nagetive_sequece_feat2) 
            
            
            return index_sequece_feat2,postive_sequece_feat2,nagetive_sequece_feat2
        
        return index_sequece_feat2,postive_sequece_feat2



class code_lstm_attion(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,64, padding_idx=padding_idx)
        #self.position_encoder = nn.Embedding(pad_size+1,64, padding_idx=padding_idx)    

        self.code_lstm  = torch.nn.LSTM(
                                input_size = 64*5,#dimension词向量维度
                                hidden_size = 64,#表示输出的特征维度，如果没有特殊变化，相当于out
                                num_layers = 2,# 表示网络的层数
                                bidirectional = True,#双向向RNN
                                batch_first = True,
                                dropout=0.2
                                )        

        self.fc0 = nn.Sequential(
            nn.Linear(64*4, 64),
        )
        
        self.fc01 = nn.Sequential(
            nn.Linear(64*2, 64),
        )     

    def cross_attention(self,lstm_output, h_t):
           # lstm_output [3, 10, 16]  h_t[10, 16]
           h_t = h_t.unsqueeze(0)
           # [10, 16, 1]
           h_t = h_t.permute(1, 2, 0)
           
    
           attn_weights = torch.bmm(lstm_output, h_t)
           attn_weights = attn_weights.permute(1, 0, 2).squeeze()
    
           # [3, 10]
           attention = Fa.softmax(attn_weights, 1)
           # bmm: [10, 16, 3] [10, 3, 1]
    
           attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(-1).transpose(1,0))
           return attn_out.squeeze()

    def get_code_post_feat(self,sequece_token):
        
        lengths = sequece_token[:,:,0] !=self.padding_idx
        lengths = lengths.type(torch.IntTensor).sum(dim=1)
        
        index_code_feat = self.code_encoder(sequece_token)
        shape = index_code_feat.shape
        shape = index_code_feat.shape        
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        index_sequece_feat = index_code_feat#+index_position_feat
        #mask = mask.repeat(1,1,4)
        index_sequece_feat = pack_padded_sequence(input=index_sequece_feat, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out,(index_sequece_feat2,_) = self.code_lstm(index_sequece_feat)
        index_sequece_feat2 = torch.cat([index_sequece_feat2[i] for i in range(4)],dim=1)
        
        index_sequece_feat2 = self.fc0(index_sequece_feat2)
        index_sequece_feat2 = torch.relu(index_sequece_feat2)
        
        packed_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        packed_out = self.fc01(packed_out)        
        

        return index_sequece_feat2,packed_out  


    def forward(self,position_token,index_sequece_token,nagetive_sequece_token):
        
        index_sequece_feat2,index_out = self.get_code_post_feat(index_sequece_token)
        
        
        nagetive_sequece_feat2,nagetive_out = self.get_code_post_feat(nagetive_sequece_token)        
        
        nagetive_sequece_feat2 = self.cross_attention(index_out,nagetive_sequece_feat2) 
        postive_sequece_feat2 = self.cross_attention(nagetive_out,index_sequece_feat2) 
        
        
        
        return postive_sequece_feat2,nagetive_sequece_feat2





class code_turn_transformer(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size+1,128, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,128, padding_idx=pad_size)    
        
        self.encoder_layer0 = nn.TransformerEncoderLayer(d_model=pad_size, nhead=2, batch_first=True)
        self.operation_transformer = nn.TransformerEncoder(self.encoder_layer0, num_layers=1)
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128*6, nhead=2, batch_first=True)
        self.instruct_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)        
        
        #self.fc = nn.Sequential(
         #   nn.Linear(128*6, 128),
          #  nn.ReLU(),
           # nn.Linear(128, 128),
        #)        

        #self.dropout = nn.Dropout(0.1)


    def get_code_post_feat(self,position_token,sequece_token):
        #sequece_token:b,s,5
        pad_mask = sequece_token[:,:,0] ==self.padding_idx
        index_code_feat = self.code_encoder(sequece_token)        
        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        position_feat = self.position_encoder(position_token)
        
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)
        #b,s,h
        
        index_code_feat = self.operation_transformer(index_code_feat.permute(0,2,1))        
        index_feat = self.instruct_transformer(index_code_feat.permute(0,2,1),src_key_padding_mask= pad_mask)
        
        #index_feat = index_feat.masked_fill(pad_mask.unsqueeze(dim=-1), 0)
        index_feat = index_feat.mean(dim=1)        
        #index_feat = index_feat[:,0,:]        
        #index_feat = self.fc(self.dropout(index_feat))
        #index_feat = self.fc(index_feat)
        
        return index_feat,'transformer'
    
    
    def forward(self,position_token,index_sequece_token,postive_sequece_token,nagetive_sequece_token=None):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_feat2,_ = self.get_code_post_feat(position_token,postive_sequece_token) 
        
        
        if nagetive_sequece_token!=None:
            nagetive_sequece_feat2,_ = self.get_code_post_feat(position_token,nagetive_sequece_token)                    
            
            return index_sequece_feat2,postive_sequece_feat2,nagetive_sequece_feat2
        
        return index_sequece_feat2,postive_sequece_feat2


class cross_code_transformer(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size+1,192, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,192, padding_idx=pad_size)    
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=192*6, nhead=4, batch_first=True)
        self.code_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        """
        self.fc = nn.Sequential(
            nn.Linear(32*6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )        
        """
        self.dropout = nn.Dropout(0.1)


    def get_code_post_feat(self,position_token,sequece_token):
        #sequece_token:b,s,5
        pad_mask = sequece_token[:,:,0] ==self.padding_idx
        index_code_feat = self.code_encoder(sequece_token)        
        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        position_feat = self.position_encoder(position_token)
        
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)
        #b,s,h
        index_feat = self.code_transformer(index_code_feat,src_key_padding_mask= pad_mask)
        
        #index_feat = index_feat.masked_fill(pad_mask.unsqueeze(dim=-1), 0)
        index_feat = index_feat.mean(dim=1)        
        #index_feat = index_feat[:,0,:]        
        #index_feat = self.fc(self.dropout(index_feat))
        #index_feat = self.fc(index_feat)
        return index_feat,'transformer'


    def forward(self,position_token,index_sequece_token,postive_sequece_token,nagetive_sequece_token=None):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_feat2,_ = self.get_code_post_feat(position_token,postive_sequece_token) 
        
        
        if nagetive_sequece_token!=None:
            nagetive_sequece_feat2,_ = self.get_code_post_feat(position_token,nagetive_sequece_token)                    
            
            return index_sequece_feat2,postive_sequece_feat2,nagetive_sequece_feat2
        
        return index_sequece_feat2,postive_sequece_feat2




class TextCNN(torch.nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv1=torch.nn.ModuleList([torch.nn.Conv1d(192*6,192,k,1) for k in [5,5,5,5,3,3]])
        self.fc=torch.nn.Linear(6*192,192*4)        
    def forward(self, x):
        out_conv1=[Fa.relu(con(x)) for con in self.conv1]
        out_maxp=[Fa.max_pool1d(c,c.size(-1)).squeeze(dim=-1) for c in out_conv1]   
        #out_maxp=[c.mean(dim=-1) for c in out_conv1]
        
        out_fc1 = torch.cat(out_maxp,dim=1)
        out=self.fc(out_fc1)
        return out


class code_text_cnn(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,192, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,192, padding_idx=pad_size)    

        self.text_nn = TextCNN()      




    def get_code_post_feat(self,position_token,sequece_token):
        
        index_code_feat = self.code_encoder(sequece_token)
        position_feat = self.position_encoder(position_token)

        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)


        index_sequece_feat2 = self.text_nn(index_code_feat.permute(0,2,1))

        return index_sequece_feat2,'cnn'
    
    


    def forward(self,position_token,index_sequece_token,postive_sequece_token):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_token,_ = self.get_code_post_feat(position_token,postive_sequece_token)        
        
        return index_sequece_feat2,postive_sequece_token





class code_resnet(nn.Module):
    
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,128, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,128, padding_idx=pad_size)    
        #(N−K+2P)/S+1
        self.conv1 = nn.Conv2d(6, 3, kernel_size=3, stride=2)
        self.resnet = torchvision.models.resnet18()#batch*1000
        self.fc0 = nn.Sequential(
            nn.Linear(1000, 256),
        )
    def get_code_post_feat(self,position_token,sequece_token):
        
        index_code_feat = self.code_encoder(sequece_token)
        position_feat = self.position_encoder(position_token)
        position_feat = position_feat.unsqueeze(dim=2)
        
        index_code_feat = torch.cat([index_code_feat,position_feat],dim=2).permute(0,2,1,3)
        
        #shape = index_code_feat.shape
        #index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3

        index_code_feat = self.conv1(index_code_feat)
        index_sequece_feat2 = self.fc0(self.resnet(index_code_feat))

        return index_sequece_feat2,'resnet'    

    def forward(self,position_token,index_sequece_token,postive_sequece_token):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_token,_ = self.get_code_post_feat(position_token,postive_sequece_token)        
        
        return index_sequece_feat2,postive_sequece_token    



class TextCNN2(torch.nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv1=torch.nn.ModuleList([torch.nn.Conv1d(128*6,128,k,1) for k in [5,5,5,5,3,3]])
        self.fc=torch.nn.Linear(6*128,128*4)        
    def forward(self, x):
        out_conv1=[Fa.relu(con(x)) for con in self.conv1]
        out_maxp=[Fa.max_pool1d(c,c.size(-1)).squeeze(dim=-1) for c in out_conv1]        
        out_fc1=torch.cat(out_maxp,dim=1)
        out=self.fc(out_fc1)
        return out
    
class code_transformer_textcnn(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,128, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,128, padding_idx=pad_size)    
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128*6, nhead=4, batch_first=True)
        self.code_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.text_nn = TextCNN2()  
        
        #self.fc = nn.Sequential(
        #    nn.Linear(32*6, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 32),
        #)        

        #self.dropout = nn.Dropout(0.1)


    def get_code_post_feat(self,position_token,sequece_token):
        #sequece_token:b,s,5
        pad_mask = sequece_token[:,:,0] ==self.padding_idx
        index_code_feat = self.code_encoder(sequece_token)        
        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        position_feat = self.position_encoder(position_token)
        
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)
        #b,s,h
        index_feat = self.code_transformer(index_code_feat,src_key_padding_mask= pad_mask)
        
        index_sequece_feat2 = self.text_nn(index_feat.permute(0,2,1))

        return index_sequece_feat2


    def forward(self,position_token,index_sequece_token,postive_sequece_token,nagetive_sequece_token=None):
        
        index_sequece_feat2 = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_feat2 = self.get_code_post_feat(position_token,postive_sequece_token) 
        
        
        if nagetive_sequece_token!=None:
            nagetive_sequece_feat2 = self.get_code_post_feat(position_token,nagetive_sequece_token)                    
            
            return index_sequece_feat2,postive_sequece_feat2,nagetive_sequece_feat2
        
        return index_sequece_feat2,postive_sequece_feat2











def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic/robust as possible whereas
        on the other hand I only want to support a limited use-case.

        Also this is arguable the most important architectural component in the Transformer model.

        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:

        https://jalammar.github.io/illustrated-transformer/

        Optimization notes:

        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 "feed forward nets" (without activation/identity hence the quotation marks).
        Conceptually both implementations are the same.

        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).

        This has an important optimization implication, they can reshape their matrix into (B*NH, S/T, HD)
        (where B - batch size, S/T - max src/trg sequence length, NH - number of heads, HD - head dimension)
        in a single step and I can only get to (B, NH, S/T, HD) in single step
        (I could call contiguous() followed by view but that's expensive as it would incur additional matrix copy)
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
            # 即将mask==0的替换为-1e9,其余不变

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        # 3*8*2*64
        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)
        # 3*2*512
        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations



class code_MultiHeadedAttention(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        self.code_encoder = nn.Embedding(vocab_size,192, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,192, padding_idx=pad_size)    

        self.atten  = MultiHeadedAttention(192*6,8,0.1,False)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )     


    def get_code_post_feat(self,position_token,sequece_token):
        
        index_code_feat = self.code_encoder(sequece_token)
        position_feat = self.position_encoder(position_token)

        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)

        mask = sequece_token[:,:,0] !=self.padding_idx
        mask = mask.unsqueeze(1).unsqueeze(1)

        index_sequece_feat2 = self.atten(index_code_feat,index_code_feat,index_code_feat,mask)
        index_sequece_feat2 = index_sequece_feat2.mean(dim = -1)
        index_sequece_feat2 = self.fc(index_sequece_feat2)

        return index_sequece_feat2,'att'
    
    


    def forward(self,position_token,index_sequece_token,postive_sequece_token):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_token,_ = self.get_code_post_feat(position_token,postive_sequece_token)        
        
        return index_sequece_feat2,postive_sequece_token







class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        
        x = self.net(x)
        return x

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.norms = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            
            #Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            #Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = self.norms(x)
        x = x.permute(0,2,1)
        x = x + self.token_mix(x)
        x = x.permute(0,2,1)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    
    def __init__(self, dim, num_classes, sq_length, depth, token_dim, channel_dim):
        super().__init__()
        
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        #self.num_patch =  (image_size// patch_size) ** 2
        #self.to_patch_embedding = nn.Sequential(
        #    nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #    Rearrange('b c h w -> b (h w) c'),
        #)
        self.num_patch =  sq_length
        
        
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        
        #x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        
        return self.mlp_head(x)



class code_MLPMixer(nn.Module):
    def __init__(self,vocab_size,padding_idx,pad_size):
        super().__init__()
        self.padding_idx = padding_idx
        
        
        self.code_encoder = nn.Embedding(vocab_size,192, padding_idx=padding_idx)
        self.position_encoder = nn.Embedding(pad_size+1,192, padding_idx=pad_size)    

        self.mlpmix  = MLPMixer(dim=192*6, num_classes=192*4, sq_length=pad_size,depth=2,token_dim=256,channel_dim=1024)




    def get_code_post_feat(self,position_token,sequece_token):
        
        index_code_feat = self.code_encoder(sequece_token)
        position_feat = self.position_encoder(position_token)

        shape = index_code_feat.shape
        index_code_feat = index_code_feat.resize(shape[0],shape[1],shape[2]*shape[3])
        index_code_feat = torch.cat((index_code_feat,position_feat),dim=-1)

        index_sequece_feat2 = self.mlpmix(index_code_feat)


        return index_sequece_feat2,'mlp'
    
    


    def forward(self,position_token,index_sequece_token,postive_sequece_token):
        
        index_sequece_feat2,_ = self.get_code_post_feat(position_token,index_sequece_token)
        postive_sequece_token,_ = self.get_code_post_feat(position_token,postive_sequece_token)        
        
        return index_sequece_feat2,postive_sequece_token







