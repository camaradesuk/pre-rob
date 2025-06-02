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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure transformers library is available
try:
    from transformers import DistilBertModel, DistilBertPreTrainedModel
except ImportError:
    print("Warning: 'transformers' library not found. DistilClsConv model will not be available.")
    # Define dummy classes to allow script to load if transformers is missing
    # This is primarily for environments where only some models might be used.
    class DistilBertPreTrainedModel(nn.Module):
        def __init__(self, config=None): super().__init__()
        def init_weights(self): pass 
            
    class DistilBertModel(nn.Module):
        def __init__(self, config=None): super().__init__()


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) for text classification.
    Uses an embedding layer followed by multiple 1D convolutional layers with different filter sizes,
    max-pooling, and a fully connected output layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_sizes: list[int], output_dim: int, dropout: float, 
                 pad_idx: int, embed_trainable: bool = True, batch_norm: bool = False):
        """
        Initializes the ConvNet model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimensionality of the word embeddings.
            n_filters (int): The number of filters for each convolutional layer.
            filter_sizes (list[int]): A list of kernel sizes for the convolutional layers.
            output_dim (int): The dimensionality of the output (number of classes).
            dropout (float): The dropout probability.
            pad_idx (int): The index of the padding token in the vocabulary.
            embed_trainable (bool, optional): Whether the embedding layer is trainable. Defaults to True.
            batch_norm (bool, optional): Whether to apply batch normalization to the fully connected layer. Defaults to False.
        """
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
        self.fc_bn = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        
        self.apply_bn = batch_norm
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvNet.

        Args:
            text (torch.Tensor): Input tensor of shape [seq_len, batch_size] representing token indices.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim] representing class probabilities (after softmax).
        """         
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
        if self.apply_bn and self.fc_bn:
            out = self.fc_bn(out)
        out = F.softmax(out, dim=1)
        return out


class AttnNet(nn.Module):
    """
    An RNN-based network with attention for text classification.
    Uses an embedding layer, an RNN (LSTM or GRU), an attention mechanism
    over the RNN's hidden states, and a fully connected output layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_hidden_dim: int, 
                 rnn_num_layers: int, output_dim: int, bidirection: bool, 
                 rnn_cell_type: str, dropout: float, pad_idx: int, 
                 embed_trainable: bool = True, batch_norm: bool = False, output_attn: bool = False):
        """
        Initializes the AttnNet model.
        Args are documented in the previous version.
        """
        super(AttnNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
            
        rnn_input_size = embedding_dim
        num_directions = 2 if bidirection else 1
        
        if rnn_cell_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                               num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                               batch_first=True, bidirectional=bidirection)
        elif rnn_cell_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                              num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                              batch_first=True, bidirectional=bidirection)
        else:
            raise ValueError(f"Unsupported RNN cell type: {rnn_cell_type}. Choose 'lstm' or 'gru'.")
            
        self.rnn_cell_type = rnn_cell_type.lower()
        self.apply_bn = batch_norm
        self.output_attn = output_attn
        
        self.W_attn = nn.Linear(num_directions * rnn_hidden_dim, num_directions * rnn_hidden_dim, bias=True)
        self.u_attn = nn.Linear(num_directions * rnn_hidden_dim, 1, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout) 
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(num_directions * rnn_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) if batch_norm else None
        
    def forward(self, text: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AttnNet.
        Args and Returns are documented in the previous version.
        """
        embed = self.embedding(text) 
        embed = embed.permute(1, 0, 2) 
        if self.rnn_cell_type == 'lstm':
            rnn_outputs, (hidden, cell) = self.rnn(embed)
        else: 
            rnn_outputs, hidden = self.rnn(embed)
        u_it = self.tanh(self.W_attn(rnn_outputs)) 
        attn_logits = self.u_attn(u_it).squeeze(2) 
        attention_scores = F.softmax(attn_logits, dim=1) 
        context_vector = torch.bmm(attention_scores.unsqueeze(1), rnn_outputs).squeeze(1)
        z = self.dropout_layer(context_vector)
        z = self.fc_out(z) 
        if self.apply_bn and self.fc_bn:
            z = self.fc_bn(z)
        probs = F.softmax(z, dim=1)
        if self.output_attn:
            return probs, attention_scores
        else:
            return probs


class WordAttn(nn.Module):
    """
    Word-level attention mechanism for Hierarchical Attention Networks (HAN).
    """
    def __init__(self, vocab_size: int, embedding_dim: int, word_hidden_dim: int, 
                 word_num_layers: int, pad_idx: int, embed_trainable: bool = True, dropout_rate: float = 0.0):
        """
        Initializes the WordAttn module.
        Args are documented in the previous version.
        """
        super(WordAttn, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=word_hidden_dim,
                          num_layers=word_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if word_num_layers > 1 else 0)
        
        self.tanh = nn.Tanh()        
        self.W_word = nn.Linear(2 * word_hidden_dim, 2 * word_hidden_dim, bias=True) 
        self.u_word = nn.Linear(2 * word_hidden_dim, 1, bias=False)
        
    def forward(self, sent_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for word-level attention.
        Args and Returns are documented in the previous version.
        """
        embed = self.embedding(sent_batch) 
        h_i, _ = self.gru(embed)
        u_it_word = self.tanh(self.W_word(h_i)) 
        attn_logits_word = self.u_word(u_it_word).squeeze(2) 
        word_attn_scores = F.softmax(attn_logits_word, dim=1) 
        sentence_vectors = torch.bmm(word_attn_scores.unsqueeze(1), h_i).squeeze(1)
        return word_attn_scores, sentence_vectors


class SentAttn(nn.Module):
    """
    Sentence-level attention mechanism for Hierarchical Attention Networks (HAN).
    """
    def __init__(self, word_hidden_dim: int, sent_hidden_dim: int, sent_num_layers: int, dropout_rate: float = 0.0):
        """
        Initializes the SentAttn module.
        Args are documented in the previous version.
        """
        super(SentAttn, self).__init__()
        self.gru = nn.GRU(input_size=2 * word_hidden_dim, 
                          hidden_size=sent_hidden_dim,
                          num_layers=sent_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if sent_num_layers > 1 else 0)
        self.tanh = nn.Tanh()        
        self.W_sent = nn.Linear(2 * sent_hidden_dim, 2 * sent_hidden_dim, bias=True) 
        self.u_sent = nn.Linear(2 * sent_hidden_dim, 1, bias=False) 
        
    def forward(self, doc_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sentence-level attention.
        Args and Returns are documented in the previous version.
        """
        h_s, _ = self.gru(doc_batch)
        u_it_sent = self.tanh(self.W_sent(h_s)) 
        attn_logits_sent = self.u_sent(u_it_sent).squeeze(2) 
        sent_attn_scores = F.softmax(attn_logits_sent, dim=1) 
        document_vectors = torch.bmm(sent_attn_scores.unsqueeze(1), h_s).squeeze(1)
        return sent_attn_scores, document_vectors


class HAN(nn.Module):
    """
    Hierarchical Attention Network (HAN) for document classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 word_hidden_dim: int, word_num_layers: int, pad_idx: int, 
                 embed_trainable: bool, batch_norm: bool,
                 sent_hidden_dim: int, sent_num_layers: int, output_dim: int,
                 output_attn: bool, dropout_rate: float = 0.5):
        """
        Initializes the HAN model.
        Args are documented in the previous version.
        """
        super(HAN, self).__init__()
        
        self.word_attn = WordAttn(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                  word_hidden_dim=word_hidden_dim, word_num_layers=word_num_layers,
                                  pad_idx=pad_idx, embed_trainable=embed_trainable, dropout_rate=dropout_rate)
        
        self.sent_attn = SentAttn(word_hidden_dim=word_hidden_dim, sent_hidden_dim=sent_hidden_dim,
                                  sent_num_layers=sent_num_layers, dropout_rate=dropout_rate)
        
        self.fc_out = nn.Linear(2 * sent_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) 
        
        self.apply_bn = batch_norm
        self.output_attn = output_attn

        self.sentence_attention_scores_inspect = None # To store sentence attention for output_attn=True
    
    def forward(self, doc_tensor: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HAN model.
        Args and Returns are documented in the previous version.
        """
        batch_size, max_doc_len, max_sent_len = doc_tensor.shape
        reshaped_sents = doc_tensor.view(batch_size * max_doc_len, max_sent_len)
        _, sentence_vectors = self.word_attn(reshaped_sents) # We only need sentence_vectors here
        doc_repr_for_sent_attn = sentence_vectors.view(batch_size, max_doc_len, -1)
        sent_attn_scores, document_vectors = self.sent_attn(doc_repr_for_sent_attn)
        self.sentence_attention_scores_inspect = sent_attn_scores 
        d_v_dropped = self.dropout(document_vectors)
        z = self.fc_out(d_v_dropped) 
        if self.apply_bn and self.fc_bn:
            z = self.fc_bn(z)
        probs = F.softmax(z, dim=1)
        if self.output_attn:
            return probs, self.sentence_attention_scores_inspect
        else:
            return probs
    

class DistilClsConv(DistilBertPreTrainedModel):
    """
    A DistilBERT-based model with a convolutional classification head.
    """
    def __init__(self, config):
        """
        Initializes the DistilClsConv model.
        Args are documented in the previous version.
        """
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
        # self.init_weights() # Called by parent

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
        """
        Forward pass of the DistilClsConv model.
        Args and Returns are documented in the previous version.
        """
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
            # from transformers.modeling_outputs import SequenceClassifierOutput
            # For compatibility, if used in HF Trainer, you'd return SequenceClassifierOutput
            # return SequenceClassifierOutput(loss=None, logits=logits, ...) 
            # For this script's usage, returning probs directly.
            return probs
        return probs
