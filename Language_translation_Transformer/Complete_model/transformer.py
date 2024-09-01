import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# class ParameteresConfig():
#     def __init__(self,**kwargs):
#         self.batch_size = 30 # batch_size
#         self.max_sequence_len = 200 # max_sequence_len
#         self.d_model = 512 # d_model
#         self.num_heads = 8 # number_heads
#         self.fnn_hidden = 2048 #fnn_hidden
#         self.drop_prob = 0.1 # drop_prob
#         self.num_layer = 5 #num_layer
#         self.START_TOKEN = '<START>'
#         self.PADDING_TOKEN = '<PADDING>'
#         self.END_TOKEN = '<END>'
#         for key,val in kwargs.items():
#             setattr(self,key,val)
#     def display(self):
#         print("parameters are:")
#         for key,val in vars(self).items():
#             print(f"{key} = {val}")

# config = ParameteresConfig()

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_sequence_length = config.max_sequence_len
        self.d_model = config.d_model
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, config, language_to_index):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = config.max_sequence_len
        self.embedding = nn.Embedding(self.vocab_size, config.d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(config)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = config.START_TOKEN
        self.END_TOKEN = config.END_TOKEN
        self.PADDING_TOKEN = config.PADDING_TOKEN
        self.device = config.device

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(self.device)

    def forward(self, x, start_token=True, end_token=True): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(self.device)
        x = self.dropout(x + pos)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.d_model % config.num_heads ==0
        self.head_dim = config.d_model/config.num_heads
        self.num_head = config.num_heads
        self.d_model = config.d_model
        self.in_linear = nn.Linear(self.d_model,3*self.d_model)
        self.out_linear = nn.Linear(self.d_model,self.d_model)
    def forward(self,x,mask):
        B,T,C = x.size() #--> batch_size,max_sequence_len,d_model i.e embeddings
        q,k,v = self.in_linear(x).split(self.d_model,dim=-1)
        q = q.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        k = k.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        v = v.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        attention = scaled_dot_product(q,k,v,mask)
        attention = attention.transpose(1,2).reshape(B,T,C)
        out = self.out_linear(attention)
        return out

class LayerNormalization(nn.Module):
    def __init__(self,config,epl=1e-5):
        super().__init__()
        self.epl = epl
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))
    def forward(self,x):
        out = F.layer_norm(x,self.gamma.shape,self.gamma,self.beta,self.epl)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.in_linear = nn.Linear(config.d_model,config.fnn_hidden)
        self.out_linear = nn.Linear(config.fnn_hidden,config.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.drop_prob)
    def forward(self,x):
        x =  self.in_linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_linear(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm1 = LayerNormalization(config)
        self.dropout1 = nn.Dropout(config.drop_prob)
        self.pos_fnn = PositionwiseFeedForward(config)
        self.layernorm2 = LayerNormalization(config)
        self.dropout2 = nn.Dropout(config.drop_prob)

    def forward(self,x,self_attention_mask):
        residule_x = x.clone()
        x = self.attention(x,self_attention_mask)
        x = self.dropout1(x)
        x = self.layernorm1(x+residule_x)
        residule_x = x.clone()
        x = self.pos_fnn(x)
        x = self.dropout2(x)
        x = self.layernorm2(x+residule_x)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self,config,language_to_index):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(config, language_to_index)
        self.layers = SequentialEncoder(*[EncoderLayer(config) for _ in range(config.num_layer)])
    def forward(self,x,self_attention_mask,start_token,end_token):
        x = self.sentence_embedding(x,start_token,end_token)
        x = self.layers(x,self_attention_mask)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vk_linear = nn.Linear(config.d_model,config.d_model*2)
        self.q_linear = nn.Linear(config.d_model,config.d_model)
        self.out_linear = nn.Linear(config.d_model,config.d_model)
        self.num_head = config.num_heads
        self.d_model = config.d_model

    def forward(self,x,y,mask): #--> y = query , x = value,key
        B,T,C = x.size()
        k,v = self.vk_linear(x).split(self.d_model,dim=-1)
        q = self.q_linear(y)
        q = q.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        k = k.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        v = v.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        attention = scaled_dot_product(q,k,v,mask)
        value = attention.transpose(1,2).reshape(B,T,C)
        out = self.out_linear(value)

        return out

class LayerDecoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNormalization(config)
        self.drop1 = nn.Dropout(config.drop_prob)

        self.cross_attention = MultiHeadCrossAttention(config)
        self.norm2 = LayerNormalization(config)
        self.drop2 = nn.Dropout(config.drop_prob)

        self.fnn = PositionwiseFeedForward(config)
        self.norm3 = LayerNormalization(config)
        self.drop3 = nn.Dropout(config.drop_prob)

    def forward(self,x,y,self_attention_mask,cross_attention_mask):
        y_residual = y.clone()
        y = self.attention(y,mask = self_attention_mask)
        y = self.drop1(y)
        y = self.norm1(y+y_residual)

        y_residual = y.clone()
        y = self.cross_attention(x,y,mask=cross_attention_mask)
        y = self.drop2(y)
        y = self.norm2(y+y_residual)

        y_residual = y.clone()
        y = self.fnn(y)
        y = self.drop3(y)
        y = self.norm3(y+y_residual)

        return y

class DecoderSequence(nn.Sequential):
    def forward(self,*inputs):
        x,y,self_attention_mask,cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x,y,self_attention_mask,cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self,config,language_to_index):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(config,language_to_index)
        self.decoder = DecoderSequence(*[LayerDecoder(config) for _ in range(config.num_layer)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y,start_token,end_token)
        y = self.decoder(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    def __init__(self,config,english_to_index,hindi_to_index):
        super().__init__()

        self.encoder = Encoder(config=config,language_to_index=english_to_index)
        self.decoder = Decoder(config=config,language_to_index=hindi_to_index)
        self.linear = nn.Linear(config.d_model,len(hindi_to_index))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                x,
                y,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out


