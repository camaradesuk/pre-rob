#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definitions.
Updated for Python 3.9, PyTorch 2.6, Transformers 4.x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure transformers library is updated (e.g., >= 4.30.0)
# pip install transformers>=4.30.0
from transformers import DistilBertModel, DistilBertPreTrainedModel

#%% --- Standard PyTorch Models ---
# These models use core PyTorch components.
# They are less likely to need changes for the PyTorch version upgrade itself,
# but rely on correct input shapes and vocab/padding from the data processing side (rob_fn.py).

class ConvNet(nn.Module):
    """CNN model for text classification."""
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx, embed_trainable, batch_norm):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable: # Simplified condition
            self.embedding.weight.requires_grad = False  # Freeze embedding

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fsize, embedding_dim))
            for fsize in filter_sizes
        ])
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        # Consider using LazyBatchNorm if input size might vary unexpectedly, but standard BatchNorm1d is fine here.
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply_bn = batch_norm

    def forward(self, text):
        """
        Forward pass for CNN model.
        Args:
            text (torch.Tensor): Input tensor [seq_len, batch_size]
        Returns:
            torch.Tensor: Output probabilities [batch_size, output_dim]
        """
        # text: [seq_len, batch_size]
        embed = self.embedding(text)  # [seq_len, batch_size, embedding_dim]
        # Permute to [batch_size, seq_len, embedding_dim] for Conv2d
        embed = embed.permute(1, 0, 2)
        # Add channel dimension: [batch_size, 1, seq_len, embedding_dim]
        embed = embed.unsqueeze(1)

        # Apply convolutions and ReLU activation
        # conved[n]: [batch_size, n_filters, (seq_len - filter_sizes[n] + 1), 1]
        conved = [F.relu(conv(embed)) for conv in self.convs]
        # Squeeze the last dimension: [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]
        conved = [conv.squeeze(3) for conv in conved]

        # Apply max pooling over time dimension
        # pooled[n]: [batch_size, n_filters, 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        # Squeeze the last dimension: [batch_size, n_filters]
        pooled = [pool.squeeze(2) for pool in pooled]

        # Concatenate pooled features from different filter sizes
        # cat: [batch_size, n_filters * len(filter_sizes)]
        cat = torch.cat(pooled, dim=1)
        dp = self.dropout(cat)
        out = self.fc(dp)  # [batch_size, output_dim]

        if self.apply_bn:
            # BatchNorm1d expects [batch_size, num_features] or [num_features]
            out = self.fc_bn(out)

        # Apply softmax to get probabilities
        out = F.softmax(out, dim=1)  # [batch_size, output_dim]

        return out


class AttnNet(nn.Module):
    """Attention-based RNN model for text classification."""
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, rnn_num_layers, output_dim,
                 bidirection, rnn_cell_type, dropout, pad_idx,
                 embed_trainable, batch_norm, output_attn=False): # Added default for output_attn
        super().__init__()

        self.output_attn = output_attn # Store whether to output attention scores

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False  # Freeze embedding

        rnn_module = nn.LSTM if rnn_cell_type.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_module(input_size=embedding_dim, hidden_size=rnn_hidden_dim,
                              num_layers=rnn_num_layers, dropout=dropout if rnn_num_layers > 1 else 0, # Apply dropout between RNN layers only
                              batch_first=True, # Input: [batch, seq, feature]
                              bidirectional=bidirection)

        num_directions = 2 if bidirection else 1
        self.rnn_cell_type = rnn_cell_type
        self.apply_bn = batch_norm

        # Attention mechanism parameters
        attn_input_dim = num_directions * rnn_hidden_dim
        self.W_attn = nn.Parameter(torch.Tensor(attn_input_dim, attn_input_dim))
        self.b_attn = nn.Parameter(torch.Tensor(attn_input_dim))
        self.u_attn = nn.Parameter(torch.Tensor(attn_input_dim, 1))

        # Initialize attention weights (example using Xavier initialization)
        nn.init.xavier_uniform_(self.W_attn)
        nn.init.zeros_(self.b_attn)
        nn.init.xavier_uniform_(self.u_attn)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(attn_input_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)


    def forward(self, text):
        """
        Forward pass for Attention RNN model.
        Args:
            text (torch.Tensor): Input tensor [seq_len, batch_size]
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                Output probabilities [batch_size, output_dim]
                (Optional) Attention scores [batch_size, seq_len, 1] if output_attn is True
        """
        # text: [seq_len, batch_size]
        embed = self.embedding(text)  # [seq_len, batch_size, embedding_dim]
        # Permute to [batch_size, seq_len, embedding_dim] for batch_first RNN
        embed = embed.permute(1, 0, 2)

        # RNN forward pass
        # rnn_output: [batch_size, seq_len, num_directions * hidden_dim]
        # h_n: [num_layers * num_directions, batch_size, hidden_dim] (final hidden state)
        # c_n: [num_layers * num_directions, batch_size, hidden_dim] (final cell state for LSTM)
        if self.rnn_cell_type.lower() == 'lstm':
            rnn_output, (h_n, c_n) = self.rnn(embed)
        else: # GRU
            rnn_output, h_n = self.rnn(embed)

        # Attention calculation
        # u_it = tanh(W * h_it + b)
        # u: [batch_size, seq_len, num_directions * hidden_dim]
        u = self.tanh(torch.matmul(rnn_output, self.W_attn) + self.b_attn)
        # scores = u_it * u_w
        # scores (alpha_it): [batch_size, seq_len, 1]
        scores = torch.matmul(u, self.u_attn)
        # Apply softmax over the sequence length dimension
        attn_weights = F.softmax(scores, dim=1) # [batch_size, seq_len, 1]

        # Compute context vector (weighted sum of RNN outputs)
        # context = sum(alpha_it * h_it)
        # rnn_output permuted: [batch_size, num_directions*hidden_dim, seq_len]
        # attn_weights permuted: [batch_size, 1, seq_len] (or keep as is and permute rnn_output)
        # context: [batch_size, num_directions * hidden_dim]
        context = torch.sum(attn_weights * rnn_output, dim=1) # Weighted sum along seq_len dim

        # Final prediction layer
        context_dp = self.dropout(context) # Apply dropout
        out = self.linear(context_dp)  # [batch_size, output_dim]

        if self.apply_bn:
            out = self.fc_bn(out)

        out_probs = F.softmax(out, dim=1)  # [batch_size, output_dim]

        if self.output_attn:
            return out_probs, attn_weights
        else:
            return out_probs


class WordAttn(nn.Module):
    """Word-level Attention module for HAN."""
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx, embed_trainable):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if not embed_trainable:
            self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=word_hidden_dim,
                          num_layers=word_num_layers, batch_first=True, bidirectional=True)

        num_directions = 2
        attn_input_dim = num_directions * word_hidden_dim

        # Word attention parameters
        self.W_word = nn.Parameter(torch.Tensor(attn_input_dim, attn_input_dim))
        self.b_word = nn.Parameter(torch.Tensor(attn_input_dim))
        self.u_word = nn.Parameter(torch.Tensor(attn_input_dim, 1))

        nn.init.xavier_uniform_(self.W_word)
        nn.init.zeros_(self.b_word)
        nn.init.xavier_uniform_(self.u_word)

        self.tanh = nn.Tanh()

    def forward(self, sent):
        """
        Forward pass for word attention.
        Args:
            sent (torch.Tensor): Input sentence tensor [batch_size, sent_len]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Word attention scores [batch_size, sent_len]
                Sentence vector [batch_size, 2 * word_hidden_dim]
        """
        # sent: [batch_size, sent_len]
        embed = self.embedding(sent)  # [batch_size, sent_len, embedding_dim]

        # Word-level GRU
        # h_i: [batch_size, sent_len, 2 * word_hidden_dim] (outputs for each word)
        h_i, _ = self.rnn(embed)

        # Word attention calculation
        # u_i = tanh(W * h_i + b)
        # u_i: [batch_size, sent_len, 2 * word_hidden_dim]
        u_i = self.tanh(torch.matmul(h_i, self.W_word) + self.b_word)
        # word_scores = u_i * u_w
        # word_scores (a_i): [batch_size, sent_len, 1]
        word_scores = torch.matmul(u_i, self.u_word)
        # Apply softmax over the sentence length dimension
        word_attn_weights = F.softmax(word_scores, dim=1)  # [batch_size, sent_len, 1]

        # Compute sentence vector (weighted sum of word annotations)
        # s_i = sum(a_i * h_i)
        # s_i: [batch_size, 2 * word_hidden_dim]
        s_i = torch.sum(word_attn_weights * h_i, dim=1) # Weighted sum along sent_len dim

        # Return attention weights (squeezed) and sentence vector
        return word_attn_weights.squeeze(2), s_i # [batch_size, sent_len], [batch_size, 2*word_hidden_dim]


class SentAttn(nn.Module):
    """Sentence-level Attention module for HAN."""
    def __init__(self, word_hidden_dim, sent_hidden_dim, sent_num_layers):
        super().__init__()

        num_directions_word = 2 # From bidirectional word GRU
        self.rnn = nn.GRU(input_size=num_directions_word * word_hidden_dim,
                          hidden_size=sent_hidden_dim,
                          num_layers=sent_num_layers, batch_first=True, bidirectional=True)

        num_directions_sent = 2
        attn_input_dim = num_directions_sent * sent_hidden_dim

        # Sentence attention parameters
        self.W_sent = nn.Parameter(torch.Tensor(attn_input_dim, attn_input_dim))
        self.b_sent = nn.Parameter(torch.Tensor(attn_input_dim))
        self.u_sent = nn.Parameter(torch.Tensor(attn_input_dim, 1))

        nn.init.xavier_uniform_(self.W_sent)
        nn.init.zeros_(self.b_sent)
        nn.init.xavier_uniform_(self.u_sent)

        self.tanh = nn.Tanh()

    def forward(self, doc):
        """
        Forward pass for sentence attention.
        Args:
            doc (torch.Tensor): Document tensor [batch_size, num_sents, 2 * word_hidden_dim]
                                (output vectors from WordAttn)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Sentence attention scores [batch_size, num_sents]
                Document vector [batch_size, 2 * sent_hidden_dim]
        """
        # doc: [batch_size, num_sents, 2 * word_hidden_dim]
        # Sentence-level GRU
        # h: [batch_size, num_sents, 2 * sent_hidden_dim] (outputs for each sentence)
        h, _ = self.rnn(doc)

        # Sentence attention calculation
        # u = tanh(W * h + b)
        # u: [batch_size, num_sents, 2 * sent_hidden_dim]
        u = self.tanh(torch.matmul(h, self.W_sent) + self.b_sent)
        # sent_scores = u * u_s
        # sent_scores (a): [batch_size, num_sents, 1]
        sent_scores = torch.matmul(u, self.u_sent)
        # Apply softmax over the num_sents dimension
        sent_attn_weights = F.softmax(sent_scores, dim=1)  # [batch_size, num_sents, 1]

        # Compute document vector (weighted sum of sentence annotations)
        # d = sum(a * h)
        # d: [batch_size, 2 * sent_hidden_dim]
        d = torch.sum(sent_attn_weights * h, dim=1) # Weighted sum along num_sents dim

        # Return attention weights (squeezed) and document vector
        return sent_attn_weights.squeeze(2), d # [batch_size, num_sents], [batch_size, 2*sent_hidden_dim]


class HAN(nn.Module):
    """Hierarchical Attention Network (HAN) model."""
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx,
                 embed_trainable, batch_norm,
                 sent_hidden_dim, sent_num_layers, output_dim,
                 output_attn):
        super().__init__()

        self.word_attn = WordAttn(vocab_size=vocab_size,
                                  embedding_dim=embedding_dim,
                                  word_hidden_dim=word_hidden_dim,
                                  word_num_layers=word_num_layers,
                                  pad_idx=pad_idx,
                                  embed_trainable=embed_trainable)

        self.sent_attn = SentAttn(word_hidden_dim=word_hidden_dim,
                                  sent_hidden_dim=sent_hidden_dim,
                                  sent_num_layers=sent_num_layers)

        self.linear = nn.Linear(2 * sent_hidden_dim, output_dim) # Input from bidirectional sentence GRU
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.apply_bn = batch_norm
        self.output_attn = output_attn # Store whether to output sentence attention

    def forward(self, text):
        """
        Forward pass for HAN model.
        Args:
            text (torch.Tensor): Input tensor [batch_size, max_doc_len, max_sent_len]
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                Output probabilities [batch_size, output_dim]
                (Optional) Sentence attention scores [batch_size, max_doc_len] if output_attn is True
        """
        # text: [batch_size, max_doc_len, max_sent_len]
        batch_size, max_doc_len, max_sent_len = text.size()

        # Reshape for WordAttn: [batch_size * max_doc_len, max_sent_len]
        text_reshaped = text.view(batch_size * max_doc_len, max_sent_len)

        # Apply WordAttn to get sentence vectors
        # word_a: [batch_size * max_doc_len, max_sent_len] (not needed here)
        # word_s: [batch_size * max_doc_len, 2 * word_hidden_dim]
        _, word_s = self.word_attn(text_reshaped)

        # Reshape sentence vectors for SentAttn: [batch_size, max_doc_len, 2 * word_hidden_dim]
        sent_vectors = word_s.view(batch_size, max_doc_len, -1)

        # Apply SentAttn to get document vector
        # doc_a: [batch_size, max_doc_len] (sentence attention scores)
        # doc_s: [batch_size, 2 * sent_hidden_dim] (document vector)
        doc_a, doc_s = self.sent_attn(sent_vectors)

        # Final classification layer
        out = self.linear(doc_s)  # [batch_size, output_dim]
        if self.apply_bn:
            out = self.fc_bn(out)

        out_probs = F.softmax(out, dim=1) # [batch_size, output_dim]

        if self.output_attn:
            return out_probs, doc_a # Return sentence attention scores
        else:
            return out_probs


#%% --- Transformer-based Model ---

class DistilClsConv(DistilBertPreTrainedModel):
    """DistilBERT model with CNN layers on top for classification."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Store num_labels from config

        # Load the base DistilBERT model
        self.distilbert = DistilBertModel(config)

        # Define CNN layers
        # Assuming config.dim holds the hidden size of DistilBERT (e.g., 768)
        cnn_out_channels = 100 # Number of filters for each kernel size
        filter_sizes = [3, 4, 5] # Kernel sizes for CNN

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, # Treat hidden states as 1 channel input
                      out_channels=cnn_out_channels,
                      kernel_size=(fsize, config.dim)) # Kernel height = fsize, width = hidden_dim
            for fsize in filter_sizes
        ])

        self.relu = nn.ReLU()
        # Use dropout probability from config if available, otherwise default
        dropout_prob = getattr(config, 'seq_classif_dropout', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer: input size = num_filters * num_kernel_sizes
        self.linear = nn.Linear(cnn_out_channels * len(filter_sizes), config.num_labels)
        self.softmax = nn.Softmax(dim=1) # Softmax for output probabilities

        # Initialize weights - DistilBertPreTrainedModel handles base model init
        # You might want custom initialization for the CNN/Linear layers if needed
        # self.init_weights() # This method is part of PreTrainedModel

    def forward(
        self,
        input_ids=None, attention_mask=None,
        head_mask=None, inputs_embeds=None, labels=None, # labels are typically used for loss calculation during training
        output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for DistilBERT + CNN classifier.
        """
        # Determine whether to return a dictionary based on config or argument
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through DistilBERT base model
        distil_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # --- Accessing hidden states (Modern Transformers API) ---
        # If return_dict is True, output is an object (e.g., BaseModelOutput)
        # If return_dict is False (or older versions), output is a tuple
        if return_dict:
            hidden_state = distil_output.last_hidden_state # [batch_size, seq_len, hidden_dim]
        else:
            hidden_state = distil_output[0] # [batch_size, seq_len, hidden_dim]
        # ---------------------------------------------------------

        # Prepare for Conv2d: [batch_size, 1, seq_len, hidden_dim]
        hidden_state_cnn = hidden_state.unsqueeze(1)

        # Apply CNN layers
        # conved[n]: [batch_size, cnn_out_channels, (seq_len - filter_sizes[n] + 1), 1]
        conved = [self.relu(conv(hidden_state_cnn)) for conv in self.convs]
        # Squeeze the last dimension: [batch_size, cnn_out_channels, (seq_len - filter_sizes[n] + 1)]
        conved = [hc.squeeze(3) for hc in conved]

        # Apply max pooling over the sequence dimension
        # hc_pooled[n]: [batch_size, cnn_out_channels, 1]
        hc_pooled = [F.max_pool1d(hc, hc.shape[2]) for hc in conved]
        # Squeeze the last dimension: [batch_size, cnn_out_channels]
        hc_pooled = [hc.squeeze(2) for hc in hc_pooled]

        # Concatenate pooled features
        # cat_output: [batch_size, cnn_out_channels * len(filter_sizes)]
        cat_output = torch.cat(hc_pooled, dim=1)

        # Apply dropout and final linear layer
        cat_output = self.dropout(cat_output) # Apply dropout *before* the final linear layer
        logits = self.linear(cat_output)  # [batch_size, num_labels]

        # Apply softmax to get probabilities for inference output
        # (Loss functions usually expect logits during training)
        probs = self.softmax(logits)   # [batch_size, num_labels]

        # --- Constructing Output ---
        # For consistency with Hugging Face models, return a suitable output object
        # if return_dict is True, otherwise return the probability tensor.
        # This example just returns probabilities for simplicity in the prediction script.
        # If used for training within HF Trainer, you'd return an object including loss and logits.

        # For this script's purpose (inference), returning probabilities is sufficient.
        return probs

