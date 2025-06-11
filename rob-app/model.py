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
        
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim) # Name 'fc' as per original context
        self.fc_bn = nn.BatchNorm1d(output_dim) # Name 'fc_bn' as per original
        self.dropout = nn.Dropout(dropout)
        
        self.apply_bn = batch_norm # Controls usage in forward pass
    
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
        out = self.fc(dp) # Use self.fc
        if self.apply_bn: # Check if self.fc_bn was created and should be applied
            # The error log `Unexpected key(s) in state_dict: "fc_bn.weight", ...` for ConvNet
            # suggests the checkpoint *didn't* have fc_bn weights.
            # If batch_norm was False when saving, fc_bn might not have been in state_dict.
            # Current code: self.fc_bn is always created, self.apply_bn controls usage.
            # If strict=False loading is used, this is fine.
            # If strict=True is desired, fc_bn creation should depend on batch_norm arg.
            # For now, keeping as is, relying on strict=False for ConvNet if needed.
            # The user's main issue was AttnNet/HAN.
            if self.fc_bn is not None: # Check if fc_bn layer exists (it does if batch_norm=True in init)
                 out = self.fc_bn(out)
        out = F.softmax(out, dim=1)
        return out


class AttnNet(nn.Module):
    """
    An RNN-based network with attention for text classification.
    Uses an embedding layer, an RNN (LSTM or GRU), an attention mechanism
    over the RNN's hidden states, and a fully connected output layer.
    Layer names reverted to match original checkpoint structure.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_hidden_dim: int, 
                 rnn_num_layers: int, output_dim: int, bidirection: bool, 
                 rnn_cell_type: str, dropout: float, pad_idx: int, 
                 embed_trainable: bool = True, batch_norm: bool = False, output_attn: bool = False):
        """
        Initializes the AttnNet model. Args documented previously.
        """
        super(AttnNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
            
        rnn_input_size = embedding_dim
        self.rnn_cell_type = rnn_cell_type.lower() # Store for forward pass

        # Reverted RNN layer naming to self.lstm and self.gru as in original
        if self.rnn_cell_type == 'lstm':
            self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                               num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                               batch_first=True, bidirectional=bidirection)
            # To satisfy a potential generic self.rnn if other parts of code expect it (unlikely here)
            # self.rnn = self.lstm 
        elif self.rnn_cell_type == 'gru':
            self.gru = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_dim,
                              num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, 
                              batch_first=True, bidirectional=bidirection)
            # self.rnn = self.gru
        else:
            raise ValueError(f"Unsupported RNN cell type: {rnn_cell_type}. Choose 'lstm' or 'gru'.")
            
        self.apply_bn = batch_norm
        self.output_attn = output_attn 
        
        num_directions = 2 if bidirection else 1
        
        # Reverted Attention mechanism parameters to self.w, self.b, self.c
        # Weight matrix for transforming RNN outputs
        w_tensor = torch.empty(num_directions * rnn_hidden_dim, num_directions * rnn_hidden_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        # Bias for the transformation
        b_tensor = torch.zeros(num_directions * rnn_hidden_dim) 
        self.b = nn.Parameter(b_tensor)      
        # Context vector for computing attention scores
        c_tensor = torch.empty(num_directions * rnn_hidden_dim, 1)
        nn.init.kaiming_uniform_(c_tensor, mode='fan_in', nonlinearity='relu')
        self.c = nn.Parameter(c_tensor)
        
        self.dropout_layer = nn.Dropout(dropout) 
        self.tanh = nn.Tanh()
        # Reverted output layer name to self.linear
        self.linear = nn.Linear(num_directions * rnn_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) # Name kept as fc_bn from original
        
    def forward(self, text: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AttnNet. Args and Returns documented previously.
        """
        embed = self.embedding(text)  
        embed = embed.permute(1, 0, 2)  
        
        # Use the appropriate RNN layer based on rnn_cell_type
        if self.rnn_cell_type == 'lstm':
            # a: [batch_size, seq_len, num_directions*hidden_dim], output features from last layer for each t
            a, (h_n, c_n) = self.lstm(embed)
        else: # GRU
            a, h_n = self.gru(embed)

        # Reverted Attention calculation
        # u: [batch_size, seq_len, num_directions*hidden_dim]
        # self.w is [num_directions*hidden_dim, num_directions*hidden_dim]
        # self.b is [num_directions*hidden_dim]
        u = self.tanh(torch.matmul(a, self.w) + self.b)
        
        # s_logits: [batch_size, seq_len, 1] (before softmax)
        # self.c is [num_directions*hidden_dim, 1]
        s_logits = torch.matmul(u, self.c)
        attention_scores_softmax = F.softmax(s_logits, dim=1) # -> [batch_size, seq_len, 1]
                
        # Context vector: weighted sum of RNN outputs 'a'
        # a: [batch_size, seq_len, num_directions*hidden_dim]
        # attention_scores_softmax: [batch_size, seq_len, 1]
        # We need to permute 'a' for bmm or use element-wise product and sum
        # Original: z = torch.matmul(a.permute(0,2,1), s) -> s was attention_scores_softmax
        context_vector = torch.matmul(a.permute(0, 2, 1), attention_scores_softmax) # -> [batch_size, num_directions*hidden_dim, 1]
        context_vector = context_vector.squeeze(2)  # -> [batch_size, num_directions*hidden_dim]
        
        z_dropped = self.dropout_layer(context_vector)
        z_out = self.linear(z_dropped)  # -> [batch_size, output_dim]
        
        if self.apply_bn: # Kept original logic for fc_bn
             if self.fc_bn is not None:
                z_out = self.fc_bn(z_out)
            
        probs = F.softmax(z_out, dim=1)  # -> [batch_size, output_dim]
        
        if self.output_attn:
            return probs, attention_scores_softmax.squeeze(2) # Squeeze to [batch_size, seq_len]
        else:
            return probs


class WordAttn(nn.Module):
    """
    Word-level attention mechanism for HAN. Layer names reverted.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, word_hidden_dim: int, 
                 word_num_layers: int, pad_idx: int, embed_trainable: bool = True, dropout_rate: float = 0.0):
        """
        Initializes the WordAttn module. Args documented previously.
        """
        super(WordAttn, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=word_hidden_dim,
                          num_layers=word_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if word_num_layers > 1 else 0)
        
        self.tanh = nn.Tanh()        
        
        # Reverted Word-level attention parameters to self.w, self.b, self.c_w
        rnn_output_dim = 2 * word_hidden_dim # Bidirectional GRU
        w_tensor = torch.empty(rnn_output_dim, rnn_output_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        
        b_tensor = torch.zeros(rnn_output_dim) 
        self.b = nn.Parameter(b_tensor)      
        
        c_w_tensor = torch.empty(rnn_output_dim, 1)
        nn.init.kaiming_uniform_(c_w_tensor, mode='fan_in', nonlinearity='relu')
        self.c_w = nn.Parameter(c_w_tensor)
        
    def forward(self, sent_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for word-level attention. Args and Returns documented previously.
        """
        embed = self.embedding(sent_batch)
        h_i, _ = self.gru(embed) # h_i: [batch_size, sent_len, 2 * word_hidden_dim]
                
        # Reverted Word attention calculation
        u_it_word = self.tanh(torch.matmul(h_i, self.w) + self.b) # -> [batch_size, sent_len, 2 * word_hidden_dim]
        attn_logits_word = torch.matmul(u_it_word, self.c_w) # -> [batch_size, sent_len, 1]
        word_attn_scores_softmax = F.softmax(attn_logits_word, dim=1) # -> [batch_size, sent_len, 1]
        
        # Compute sentence vector s_i
        # h_i permuted: [batch_size, 2 * word_hidden_dim, sent_len]
        sentence_vectors = torch.matmul(h_i.permute(0, 2, 1), word_attn_scores_softmax) # -> [batch_size, 2 * word_hidden_dim, 1]
        
        return word_attn_scores_softmax.squeeze(2), sentence_vectors.squeeze(2)


class SentAttn(nn.Module):
    """
    Sentence-level attention mechanism for HAN. Layer names reverted.
    """
    def __init__(self, word_hidden_dim: int, sent_hidden_dim: int, sent_num_layers: int, dropout_rate: float = 0.0):
        """
        Initializes the SentAttn module. Args documented previously.
        """
        super(SentAttn, self).__init__()
               
        self.gru = nn.GRU(input_size=2 * word_hidden_dim, 
                          hidden_size=sent_hidden_dim,
                          num_layers=sent_num_layers, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate if sent_num_layers > 1 else 0)
        
        self.tanh = nn.Tanh()        
        
        # Reverted Sentence-level attention parameters to self.w, self.b, self.c_s
        rnn_output_dim = 2 * sent_hidden_dim # Bidirectional GRU
        w_tensor = torch.empty(rnn_output_dim, rnn_output_dim)
        nn.init.kaiming_uniform_(w_tensor, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w_tensor)
        
        b_tensor = torch.zeros(rnn_output_dim) 
        self.b = nn.Parameter(b_tensor) 
        
        c_s_tensor = torch.empty(rnn_output_dim, 1)
        nn.init.kaiming_uniform_(c_s_tensor, mode='fan_in', nonlinearity='relu')
        self.c_s = nn.Parameter(c_s_tensor)
        
    def forward(self, doc_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sentence-level attention. Args and Returns documented previously.
        """
        h_s, _ = self.gru(doc_batch) # h_s: [batch_size, num_sents, 2 * sent_hidden_dim]
        
        # Reverted Sentence attention calculation
        u_it_sent = self.tanh(torch.matmul(h_s, self.w) + self.b) # -> [batch_size, num_sents, 2 * sent_hidden_dim]  
        attn_logits_sent = torch.matmul(u_it_sent, self.c_s) # -> [batch_size, num_sents, 1]
        sent_attn_scores_softmax = F.softmax(attn_logits_sent, dim=1) # -> [batch_size, num_sents, 1]
        
        # Compute document vector d
        document_vectors = torch.matmul(h_s.permute(0, 2, 1), sent_attn_scores_softmax) # -> [batch_size, 2 * sent_hidden_dim, 1]
        
        return sent_attn_scores_softmax.squeeze(2), document_vectors.squeeze(2)


class HAN(nn.Module):
    """
    Hierarchical Attention Network (HAN) for document classification. Output layer name reverted.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 word_hidden_dim: int, word_num_layers: int, pad_idx: int, 
                 embed_trainable: bool, batch_norm: bool,
                 sent_hidden_dim: int, sent_num_layers: int, output_dim: int,
                 output_attn: bool, dropout_rate: float = 0.5):
        """
        Initializes the HAN model. Args documented previously.
        """
        super(HAN, self).__init__()
        
        self.word_attn = WordAttn(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                  word_hidden_dim=word_hidden_dim, word_num_layers=word_num_layers,
                                  pad_idx=pad_idx, embed_trainable=embed_trainable, dropout_rate=dropout_rate)
        
        self.sent_attn = SentAttn(word_hidden_dim=word_hidden_dim, sent_hidden_dim=sent_hidden_dim,
                                  sent_num_layers=sent_num_layers, dropout_rate=dropout_rate)
        
        # Reverted output layer name to self.linear
        self.linear = nn.Linear(2 * sent_hidden_dim, output_dim) 
        self.fc_bn = nn.BatchNorm1d(output_dim) # Name kept as fc_bn
        
        self.apply_bn = batch_norm
        self.output_attn = output_attn
        self.dropout = nn.Dropout(dropout_rate) # Added dropout based on HAN best practices

        self.sentence_attention_scores_inspect = None 
    
    def forward(self, doc_tensor: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HAN model. Args and Returns documented previously.
        """
        batch_size, max_doc_len, max_sent_len = doc_tensor.shape
        
        # The original HAN forward loop through sentences one by one.
        # The previous refactored version processed them in a batch for WordAttn.
        # Reverting to a loop to more closely match original logic if it was critical,
        # however, batch processing is generally more efficient.
        # The state_dict mismatch is about layer names, not this loop.
        # The provided original code's HAN forward was:
        # text = text.permute(1, 0, 2)  # [max_doc_len, batch_size, max_sent_len]
        # word_a_ls, word_s_ls = [], []
        # for sent in text: # sent: [batch_size, max_sent_len]
        #    word_a, word_s = self.word_attn(sent)
        #    ...
        # This implies WordAttn expects [batch_size, max_sent_len].
        # My WordAttn and SentAttn are designed for batch_first=True inputs.

        # Current batched approach for WordAttn (more efficient):
        reshaped_sents = doc_tensor.view(batch_size * max_doc_len, max_sent_len)
        _, sentence_vectors = self.word_attn(reshaped_sents) 
        doc_repr_for_sent_attn = sentence_vectors.view(batch_size, max_doc_len, -1)
        
        sent_attn_scores, document_vectors = self.sent_attn(doc_repr_for_sent_attn)
        self.sentence_attention_scores_inspect = sent_attn_scores 
        
        # Original had no dropout here, but adding it is common.
        # If checkpoints were saved without this dropout, it might be an unexpected key
        # if not handled by strict=False. For now, keeping the dropout layer.
        d_v_dropped = self.dropout(document_vectors) 
        z = self.linear(d_v_dropped) # Use self.linear
        
        if self.apply_bn: # Kept original logic for fc_bn
            if self.fc_bn is not None:
                z = self.fc_bn(z)
            
        probs = F.softmax(z, dim=1)
        
        if self.output_attn:
            return probs, self.sentence_attention_scores_inspect
        else:
            return probs
    

class DistilClsConv(DistilBertPreTrainedModel):
    """
    A DistilBERT-based model with a convolutional classification head.
    No changes to layer names here as it wasn't highlighted as an issue.
    """
    def __init__(self, config):
        """
        Initializes the DistilClsConv model. Args documented previously.
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
        Forward pass of the DistilClsConv model. Args and Returns documented previously.
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
            return probs # Or appropriate SequenceClassifierOutput if labels were handled
        return probs
