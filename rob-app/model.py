#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch model definitions for Risk of Bias (RoB) classification tasks.
Includes:
- ConvNet: A Convolutional Neural Network for text classification.
- AttnNet: An RNN-based network with attention for text classification.
- HAN: A Hierarchical Attention Network for document classification.
- DistilClsConv: A DistilBERT-based model with a convolutional head for classification.

Updated for Python 3.10, PyTorch 2.x, Transformers 4.x.
Layer names in AttnNet, WordAttn, SentAttn, and HAN's output layer
have been reverted to align with original checkpoint structures.
AttnNet now always creates both LSTM and GRU layers to match original checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure transformers library is available
try:
    from transformers import DistilBertModel, DistilBertPreTrainedModel
except ImportError:
    print("Warning: 'transformers' library not found. DistilClsConv model will not be available.")
    class DistilBertPreTrainedModel(nn.Module):
        def __init__(self, config=None): super().__init__()
        def init_weights(self): pass 
    class DistilBertModel(nn.Module):
        def __init__(self, config=None): super().__init__()


class ConvNet(nn.Module): # Content remains the same as model_py_reverted_layers
    """
    A Convolutional Neural Network (CNN) for text classification.
    Uses an embedding layer followed by multiple 1D convolutional layers with different filter sizes,
    max-pooling, and a fully connected output layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_sizes: list[int], output_dim: int, dropout: float, 
                 pad_idx: int, embed_trainable: bool = True, batch_norm: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters,
                      kernel_size=(fsize, embedding_dim)) 
            for fsize in filter_sizes
        ])                
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) 
        self.dropout = nn.Dropout(dropout)
        self.apply_bn = batch_norm 
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(text)  
        embed = embed.permute(1, 0, 2) 
        embed = embed.unsqueeze(1)      
        conved = [F.relu(conv(embed)) for conv in self.convs]
        conved = [conv.squeeze(3) for conv in conved]
        pooled = [F.max_pool1d(conv_item, conv_item.shape[2]) for conv_item in conved]
        pooled = [pool_item.squeeze(2) for pool_item in pooled]
        cat = torch.cat(pooled, dim=1)
        dp = self.dropout(cat)
        out = self.fc(dp) 
        if self.apply_bn: 
            if self.fc_bn is not None: 
                 out = self.fc_bn(out)
        out = F.softmax(out, dim=1)
        return out


class AttnNet(nn.Module):
    """
    An RNN-based network with attention for text classification.
    Now always creates both self.lstm and self.gru to match original checkpoint structure.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_hidden_dim: int, 
                 rnn_num_layers: int, output_dim: int, bidirection: bool, 
                 rnn_cell_type: str, dropout: float, pad_idx: int, 
                 embed_trainable: bool = True, batch_norm: bool = False, output_attn: bool = False):
        super(AttnNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
            
        rnn_input_size = embedding_dim
        self.rnn_cell_type = rnn_cell_type.lower() 

        # Create BOTH LSTM and GRU layers, as in the original model.py
        # This ensures all keys from original checkpoints are expected.
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                           num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                           batch_first=True, bidirectional=bidirection)
        self.gru = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                          num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                          batch_first=True, bidirectional=bidirection)
            
        self.apply_bn = batch_norm
        self.output_attn = output_attn 
        
        num_directions = 2 if bidirection else 1
        
        # Attention parameters (reverted to original naming)
        w_tensor = torch.empty(num_directions * rnn_hidden_dim, num_directions * rnn_hidden_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        b_tensor = torch.zeros(num_directions * rnn_hidden_dim) 
        self.b = nn.Parameter(b_tensor)      
        c_tensor = torch.empty(num_directions * rnn_hidden_dim, 1)
        nn.init.kaiming_uniform_(c_tensor, mode='fan_in', nonlinearity='relu')
        self.c = nn.Parameter(c_tensor)
        
        self.dropout_layer = nn.Dropout(dropout) 
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(num_directions * rnn_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) 
        
    def forward(self, text: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(text)  
        embed = embed.permute(1, 0, 2)  
        
        if self.rnn_cell_type == 'lstm':
            a, (h_n, c_n) = self.lstm(embed)
        elif self.rnn_cell_type == 'gru': # Ensure this matches the key in args_json
            a, h_n = self.gru(embed)
        else:
            # Fallback or error if rnn_cell_type is unexpected, though __init__ should catch this.
            # This path should ideally not be taken if __init__ validates rnn_cell_type.
            raise ValueError(f"AttnNet forward: rnn_cell_type '{self.rnn_cell_type}' is not 'lstm' or 'gru'. This should have been caught in __init__.")

        u = self.tanh(torch.matmul(a, self.w) + self.b)
        s_logits = torch.matmul(u, self.c)
        attention_scores_softmax = F.softmax(s_logits, dim=1) 
        context_vector = torch.matmul(a.permute(0, 2, 1), attention_scores_softmax) 
        context_vector = context_vector.squeeze(2)  
        z_dropped = self.dropout_layer(context_vector)
        z_out = self.linear(z_dropped)  
        if self.apply_bn: 
             if self.fc_bn is not None:
                z_out = self.fc_bn(z_out)
        probs = F.softmax(z_out, dim=1)  
        if self.output_attn:
            return probs, attention_scores_softmax.squeeze(2) 
        else:
            return probs


class WordAttn(nn.Module): # Content remains the same as model_py_reverted_layers
    def __init__(self, vocab_size: int, embedding_dim: int, word_hidden_dim: int, 
                 word_num_layers: int, pad_idx: int, embed_trainable: bool = True, dropout_rate: float = 0.0):
        super(WordAttn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=word_hidden_dim,
                          num_layers=word_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if word_num_layers > 1 else 0)
        self.tanh = nn.Tanh()        
        rnn_output_dim = 2 * word_hidden_dim 
        w_tensor = torch.empty(rnn_output_dim, rnn_output_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        b_tensor = torch.zeros(rnn_output_dim) 
        self.b = nn.Parameter(b_tensor)      
        c_w_tensor = torch.empty(rnn_output_dim, 1)
        nn.init.kaiming_uniform_(c_w_tensor, mode='fan_in', nonlinearity='relu')
        self.c_w = nn.Parameter(c_w_tensor)
        
    def forward(self, sent_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(sent_batch)
        h_i, _ = self.gru(embed) 
        u_it_word = self.tanh(torch.matmul(h_i, self.w) + self.b) 
        attn_logits_word = torch.matmul(u_it_word, self.c_w) 
        word_attn_scores_softmax = F.softmax(attn_logits_word, dim=1) 
        sentence_vectors = torch.matmul(h_i.permute(0, 2, 1), word_attn_scores_softmax) 
        return word_attn_scores_softmax.squeeze(2), sentence_vectors.squeeze(2)


class SentAttn(nn.Module): # Content remains the same as model_py_reverted_layers
    def __init__(self, word_hidden_dim: int, sent_hidden_dim: int, sent_num_layers: int, dropout_rate: float = 0.0):
        super(SentAttn, self).__init__()
        self.gru = nn.GRU(input_size=2 * word_hidden_dim, 
                          hidden_size=sent_hidden_dim,
                          num_layers=sent_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if sent_num_layers > 1 else 0)
        self.tanh = nn.Tanh()        
        rnn_output_dim = 2 * sent_hidden_dim 
        w_tensor = torch.empty(rnn_output_dim, rnn_output_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        b_tensor = torch.zeros(rnn_output_dim) 
        self.b = nn.Parameter(b_tensor) 
        c_s_tensor = torch.empty(rnn_output_dim, 1)
        nn.init.kaiming_uniform_(c_s_tensor, mode='fan_in', nonlinearity='relu')
        self.c_s = nn.Parameter(c_s_tensor)
        
    def forward(self, doc_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_s, _ = self.gru(doc_batch) 
        u_it_sent = self.tanh(torch.matmul(h_s, self.w) + self.b)  
        attn_logits_sent = torch.matmul(u_it_sent, self.c_s) 
        sent_attn_scores_softmax = F.softmax(attn_logits_sent, dim=1) 
        document_vectors = torch.matmul(h_s.permute(0, 2, 1), sent_attn_scores_softmax) 
        return sent_attn_scores_softmax.squeeze(2), document_vectors.squeeze(2)


class HAN(nn.Module): # Content remains the same as model_py_reverted_layers
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 word_hidden_dim: int, word_num_layers: int, pad_idx: int, 
                 embed_trainable: bool, batch_norm: bool,
                 sent_hidden_dim: int, sent_num_layers: int, output_dim: int,
                 output_attn: bool, dropout_rate: float = 0.5):
        super(HAN, self).__init__()
        self.word_attn = WordAttn(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                  word_hidden_dim=word_hidden_dim, word_num_layers=word_num_layers,
                                  pad_idx=pad_idx, embed_trainable=embed_trainable, dropout_rate=dropout_rate)
        self.sent_attn = SentAttn(word_hidden_dim=word_hidden_dim, sent_hidden_dim=sent_hidden_dim,
                                  sent_num_layers=sent_num_layers, dropout_rate=dropout_rate)
        self.linear = nn.Linear(2 * sent_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) 
        self.apply_bn = batch_norm
        self.output_attn = output_attn
        self.dropout = nn.Dropout(dropout_rate) 
        self.sentence_attention_scores_inspect = None 
    
    def forward(self, doc_tensor: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_doc_len, max_sent_len = doc_tensor.shape
        reshaped_sents = doc_tensor.view(batch_size * max_doc_len, max_sent_len)
        _, sentence_vectors = self.word_attn(reshaped_sents) 
        doc_repr_for_sent_attn = sentence_vectors.view(batch_size, max_doc_len, -1)
        sent_attn_scores, document_vectors = self.sent_attn(doc_repr_for_sent_attn)
        self.sentence_attention_scores_inspect = sent_attn_scores 
        d_v_dropped = self.dropout(document_vectors) 
        z = self.linear(d_v_dropped) 
        if self.apply_bn: 
            if self.fc_bn is not None:
                z = self.fc_bn(z)
        probs = F.softmax(z, dim=1)
        if self.output_attn:
            return probs, self.sentence_attention_scores_inspect
        else:
            return probs
    

class DistilClsConv(DistilBertPreTrainedModel): # Content remains the same as model_py_reverted_layers
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        n_conv_filters = getattr(config, 'n_conv_filters', 100)
        conv_filter_sizes = getattr(config, 'conv_filter_sizes', [3, 4, 5])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_conv_filters,
                      kernel_size=(fsize, config.dim)) 
            for fsize in conv_filter_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)      
        self.linear = nn.Linear(n_conv_filters * len(conv_filter_sizes), config.num_labels)     
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None, 
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None, 
        inputs_embeds: torch.Tensor | None = None, 
        labels: torch.Tensor | None = None, 
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor: 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        distil_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
            inputs_embeds=inputs_embeds, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        hidden_state = distil_output[0] if not return_dict else distil_output.last_hidden_state
        hidden_state_unsqueezed = hidden_state.unsqueeze(1)
        hidden_conved = [self.relu(conv(hidden_state_unsqueezed)) for conv in self.convs]
        hidden_conved_squeezed = [hc.squeeze(3) for hc in hidden_conved]
        hc_pooled = [F.max_pool1d(hc_sq, hc_sq.shape[2]) for hc_sq in hidden_conved_squeezed]
        hc_pooled_squeezed = [hc_p.squeeze(2) for hc_p in hc_pooled]
        cat_output = torch.cat(hc_pooled_squeezed, dim=1)
        cat_output = self.relu(cat_output) 
        cat_output = self.dropout(cat_output)  
        logits = self.linear(cat_output)
        probs = self.softmax(logits)
        if return_dict:
            return probs 
        return probs
