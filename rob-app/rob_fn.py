#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated for Python 3.9, PyTorch 2.6, Transformers 4.x, SentenceTransformers 2.x
"""

import json
import dill  # Or consider standard pickle if only simple dicts are saved
import torch
import re
import pandas as pd
import spacy
import warnings
from pathlib import Path

# --- Dependency Updates ---
# Ensure these libraries are installed with versions compatible with Python 3.9 and PyTorch 2.6+
# pip install torch torchvision torchaudio
# pip install transformers>=4.30.0 sentence-transformers>=2.2.0 spacy>=3.0.0 pandas dill
# python -m spacy download en_core_web_sm

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Import from updated libraries
from transformers import AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel # Using AutoTokenizer is more flexible
from sentence_transformers import SentenceTransformer, util

# Import custom models (assuming model.py is in the same directory or Python path)
# Ensure model.py itself doesn't have incompatible code (less likely for definitions)
from model import ConvNet, AttnNet, HAN, DistilClsConv

# Set device: GPU if available, otherwise CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Constants ---
# Default RoB item descriptions (if not provided in args)
ROB_ITEM_DESCRIPTIONS = {
    'RandomizationTreatmentControl': 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment',
    'BlindedOutcomeAssessment': 'Assessment of an outcome in a blinded fashion. Investigators measuring the outcome do not know which treatment group the animals belongs to and what treatment they had received',
    'SampleSizeCalculation': 'The manuscript reports the performance of a sample size calculation and describes how this number was derived statistically',
    'AnimalExclusions': 'All animals, all data and all outcomes measured are accounted for and presented in the final analysis. Reasons are given for animal exclusions',
    'AllocationConcealment': 'Investigators performing the experiment do not know which treatment an animal is being given',
    'AnimalWelfareRegulations': 'Research investigators complied with animal welfare regulations',
    'ConflictsOfInterest': 'Potential conflict of interest, like funding or affiliation to a pharmaceutical company'
}

# --- Model Loading ---

def load_vocab_info(fld_path):
    """
    Loads vocabulary info (stoi, pad_idx) from a file.
    *** This replaces loading the old torchtext.Field object. ***
    Assumes fld_path points to a file saved with dill/pickle
    containing a dictionary like: {'stoi': {...}, 'pad_token': '<pad>', 'unk_token': '<unk>'}
    or directly the Field object from which we extract the info.
    You MUST adapt this function or the saved files.
    """
    try:
        with open(fld_path, "rb") as fin:
            # Attempt to load assuming it's a dictionary or object with needed attributes
            loaded_obj = dill.load(fin)

        if isinstance(loaded_obj, dict) and 'stoi' in loaded_obj and 'pad_token' in loaded_obj and 'unk_token' in loaded_obj:
            vocab_stoi = loaded_obj['stoi']
            pad_token = loaded_obj['pad_token']
            unk_token = loaded_obj['unk_token']
            pad_idx = vocab_stoi.get(pad_token, 1) # Default pad_idx to 1 if not found directly
            unk_idx = vocab_stoi.get(unk_token, 0) # Default unk_idx to 0 if not found directly
            print(f"Loaded vocab info from dict: {Path(fld_path).name}")
            return vocab_stoi, pad_idx, unk_idx
        # Elif try to load as legacy Field object (requires torchtext legacy to be installed)
        # Note: Installing torchtext legacy alongside modern PyTorch might cause issues.
        # This part is experimental and might need removal.
        elif hasattr(loaded_obj, 'vocab') and hasattr(loaded_obj, 'pad_token') and hasattr(loaded_obj, 'unk_token'):
             warnings.warn(f"Loaded legacy torchtext.Field object from {Path(fld_path).name}. "
                           "Consider saving vocab info directly as a dictionary.", DeprecationWarning)
             vocab_stoi = loaded_obj.vocab.stoi
             pad_token = loaded_obj.pad_token
             unk_token = loaded_obj.unk_token
             pad_idx = vocab_stoi.get(pad_token, 1)
             unk_idx = vocab_stoi.get(unk_token, 0)
             print(f"Loaded vocab info from legacy Field: {Path(fld_path).name}")
             return vocab_stoi, pad_idx, unk_idx
        else:
             raise ValueError(f"Cannot extract vocab info from {fld_path}. "
                              "Expected a dict with 'stoi', 'pad_token', 'unk_token' or a legacy Field object.")

    except Exception as e:
        print(f"Error loading or processing vocab file {fld_path}: {e}")
        raise

def load_model_legacy(arg_path, pth_path, fld_path):
    """Loads legacy models (ConvNet, AttnNet, HAN)"""
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']

    # --- Load Vocabulary Info (Replaces TorchText Field Loading) ---
    # This is the critical change. Assumes fld_path contains pickled vocab info.
    try:
        vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
        vocab_size = len(vocab_stoi)
    except Exception as e:
        print(f"Failed to load vocabulary from {fld_path}. Cannot proceed.")
        raise e
    # --------------------------------------------------------------

    # Load model based on net_type from args
    net_type = args.get('net_type', 'unknown') # Use .get for safety
    model = None

    if net_type == 'cnn':
        sizes = [int(s) for s in args['filter_sizes'].split(',')]
        model = ConvNet(vocab_size=vocab_size, # Use actual vocab size
                        embedding_dim=args['embed_dim'],
                        n_filters=args['num_filters'],
                        filter_sizes=sizes,
                        output_dim=2, # Assuming binary classification
                        dropout=args['dropout'],
                        pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'],
                        batch_norm=args['batch_norm'])
    elif net_type == 'attn':
        model = AttnNet(vocab_size=vocab_size, # Use actual vocab size
                        embedding_dim=args['embed_dim'],
                        rnn_hidden_dim=args['rnn_hidden_dim'],
                        rnn_num_layers=args['rnn_num_layers'],
                        output_dim=2, # Assuming binary classification
                        bidirection=args['bidirection'],
                        rnn_cell_type=args['rnn_cell_type'],
                        dropout=args['dropout'],
                        pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'],
                        batch_norm=args['batch_norm'],
                        output_attn=False) # HAN model handles attention output if needed
    elif net_type == 'han':
         model = HAN(vocab_size=vocab_size, # Use actual vocab size
                    embedding_dim=args['embed_dim'],
                    word_hidden_dim=args['word_hidden_dim'],
                    word_num_layers=args['word_num_layers'],
                    pad_idx=pad_idx,
                    embed_trainable=args['embed_trainable'],
                    batch_norm=args['batch_norm'],
                    sent_hidden_dim=args['sent_hidden_dim'],
                    sent_num_layers=args['sent_num_layers'],
                    output_dim=2, # Assuming binary classification
                    output_attn=True) # HAN needs attention scores for sentence extraction
    else:
        raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path}")


    # Load checkpoint and update model weights
    # *** CRITICAL: Added weights_only=False due to PyTorch 2.6 default change ***
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    except RuntimeError as e:
         # Fallback for older PyTorch versions that don't support weights_only
         if "weights_only" in str(e):
             warnings.warn(f"torch.load failed with weights_only=False for {pth_path}. Trying without it (might be insecure if file is untrusted). Error: {e}", UserWarning)
             checkpoint = torch.load(pth_path, map_location=device)
         else:
             raise e # Re-raise other runtime errors
    except FileNotFoundError:
         print(f"Error: Model checkpoint file not found at {pth_path}")
         raise
    except Exception as e:
         print(f"Error loading checkpoint {pth_path}: {e}")
         raise


    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint itself is the state_dict (common practice)
        state_dict = checkpoint
        warnings.warn(f"Checkpoint {pth_path} does not contain a 'state_dict' key. Assuming the checkpoint *is* the state_dict.", UserWarning)

    # Load state dict - set strict=False if models might have slightly different architectures
    # (e.g., if vocab size changed, the embedding layer size will differ)
    try:
         model.load_state_dict(state_dict, strict=False)
         print(f"Loaded model state_dict from: {Path(pth_path).name}")
    except RuntimeError as e:
         print(f"Error loading state_dict into {net_type} model from {pth_path}: {e}")
         print("Possible issues: Mismatched layer names or sizes. Check model definition and checkpoint.")
         raise e


    # Explicitly clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model.to(device)
    model.eval() # Set model to evaluation mode

    # Return vocab_stoi and unk_idx as well, needed for prediction
    return model, args, vocab_stoi, pad_idx, unk_idx

def load_model_bert(arg_path, pth_path):
    """Loads BERT-based models (DistilClsConv)"""
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']

    # Determine RoB sentence description
    rob_item = args.get('rob_item')
    rob_sent = args.get('rob_sent')
    if rob_sent is None and rob_item in ROB_ITEM_DESCRIPTIONS:
        rob_sent = ROB_ITEM_DESCRIPTIONS[rob_item]
    elif rob_sent is None:
        raise ValueError(f"RoB item description not found for '{rob_item}' in args file {arg_path} and not provided via 'rob_sent'.")

    ### Load the BERT-based model ###
    # Assuming DistilClsConv is defined in model.py and inherits correctly
    # Use AutoModel for flexibility if needed, but DistilClsConv seems custom
    try:
        # Ensure the model definition matches the saved weights structure
        # num_labels=2 for binary classification
        model = DistilClsConv.from_pretrained('distilbert-base-uncased', num_labels=2, return_dict=True)
        print(f"Initialized DistilClsConv model from pretrained 'distilbert-base-uncased'")
    except Exception as e:
        print(f"Error initializing DistilClsConv from pretrained: {e}")
        raise

    ### Load checkpoint and update model weights ###
    # *** CRITICAL: Added weights_only=False due to PyTorch 2.6 default change ***
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    except RuntimeError as e:
         if "weights_only" in str(e):
             warnings.warn(f"torch.load failed with weights_only=False for {pth_path}. Trying without it (might be insecure if file is untrusted). Error: {e}", UserWarning)
             checkpoint = torch.load(pth_path, map_location=device)
         else:
             raise e
    except FileNotFoundError:
         print(f"Error: Model checkpoint file not found at {pth_path}")
         raise
    except Exception as e:
         print(f"Error loading checkpoint {pth_path}: {e}")
         raise


    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Assume checkpoint is the state_dict
        warnings.warn(f"Checkpoint {pth_path} does not contain a 'state_dict' key. Assuming the checkpoint *is* the state_dict.", UserWarning)

    # Load state dict - strict=False might be needed if custom head differs slightly
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model state_dict from: {Path(pth_path).name}")
    except RuntimeError as e:
        print(f"Error loading state_dict into DistilClsConv model from {pth_path}: {e}")
        print("Possible issues: Mismatched layer names or sizes (especially the classification head).")
        raise e


    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Initialize tokenizer (using AutoTokenizer is generally recommended)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize Sentence Transformer model (use a current recommended model)
    # 'distilbert-base-nli-stsb-mean-tokens' is older, consider 'all-MiniLM-L6-v2' or others
    try:
        sent_model_name = 'all-MiniLM-L6-v2' # Example: Replace with desired model
        sent_model = SentenceTransformer(sent_model_name, device=device)
        print(f"Loaded SentenceTransformer model: {sent_model_name}")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        # Fallback or specific model name from args if needed
        sent_model_name_legacy = 'distilbert-base-nli-stsb-mean-tokens'
        warnings.warn(f"Failed to load {sent_model_name}, falling back to legacy {sent_model_name_legacy}", UserWarning)
        try:
             sent_model = SentenceTransformer(sent_model_name_legacy, device=device)
             print(f"Loaded SentenceTransformer model: {sent_model_name_legacy}")
        except Exception as e2:
             print(f"Error loading fallback SentenceTransformer model {sent_model_name_legacy}: {e2}")
             raise


    # Explicitly clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return model, tokenizer, sent_model, rob_sent

# --- Prediction Functions ---

def pred_legacy(doc, model, args, vocab_stoi, pad_idx, unk_idx):
    """Prediction with non-BERT models (ConvNet, AttnNet)"""
    model.eval() # Ensure model is in eval mode

    # Tokenization using Spacy
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]
    # Convert tokens to indices using the loaded vocab_stoi
    idx = [vocab_stoi.get(t, unk_idx) for t in tokens] # Use unk_idx for OOV tokens

    # Padding / Truncation
    max_len = args.get('max_token_len', 512) # Get max length froargs, default 512
    if len(idx) < max_len:
        idx += [pad_idx] * (max_len - len(idx))
    elif len(idx) > max_len:
        idx = idx[:max_len]

    # Convert to tensor
    # Use torch.tensor instead of torch.LongTensor
    doc_tensor = torch.tensor(idx, dtype=torch.long).to(device)

    # Reshape for model input (depends on model type)
    # ConvNet/AttnNet expect [seq_len, batch_size=1]
    doc_tensor = doc_tensor.unsqueeze(1) # Add batch dimension

    # Prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        probs = model(doc_tensor) # Should output [batch_size=1, output_dim=2]

    # Extract probability of the positive class (assuming index 1)
    prob_positive = probs.squeeze().cpu().numpy()[1] # Squeeze batch dim, move to CPU, get numpy, take index 1

    return float(prob_positive) # Return as standard float

def pred_bert(text, model, tokenizer, sent_model, rob_sent, max_n_sent=30):
    """Prediction with BERT-based model (DistilClsConv) using sentence similarity"""
    model.eval() # Ensure model is in eval mode

    # Sentence Splitting using Spacy
    doc = nlp(text)
    sents = [str(s).strip() for s in doc.sents if len(str(s).split()) > 3] # Basic filtering

    if not sents:
        warnings.warn("No sentences found after filtering in pred_bert.", UserWarning)
        return 0.0 # Return neutral probability if no sentences

    # Compute sentence embeddings
    try:
        # Encode sentences in batches for efficiency if many sentences
        sent_embeds = sent_model.encode(sents, convert_to_tensor=True, device=device, show_progress_bar=False)
        rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True, device=device, show_progress_bar=False)
    except Exception as e:
        print(f"Error during sentence encoding: {e}")
        return 0.0 # Return neutral probability on error

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed) # Shape: [n_sents, 1]

    # Find top N most similar sentences
    # Use torch.topk for efficiency
    top_k = min(max_n_sent, len(sents))
    top_results = torch.topk(cos_scores.squeeze(), k=top_k) # Squeeze to 1D tensor

    # Get indices and scores of top sentences
    top_indices = top_results.indices.cpu().numpy()
    # top_scores = top_results.values.cpu().numpy() # Scores not directly needed for prediction text

    sim_sents = [sents[i] for i in top_indices]
    sim_text = " ".join(sim_sents) # Join with space, BERT tokenizer handles separation

    # Tokenize combined text for the classification model
    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512) # Use model's max length

    # Ensure inputs are on the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs) # Model should return probabilities or logits

    # Assuming the model output is probabilities (softmax applied internally)
    # If it outputs logits, apply softmax: probs = torch.softmax(outputs.logits, dim=-1)
    probs = outputs # If model returns probs directly
    if hasattr(outputs, 'logits'): # Handle HuggingFace standard output format
         probs = torch.softmax(outputs.logits, dim=-1)

    # Extract probability of the positive class (assuming index 1)
    prob_positive = probs.squeeze().cpu().numpy()[1] # Squeeze batch dim, move to CPU, get numpy, take index 1

    return float(prob_positive) # Return as standard float


def extract_sent_han(doc_text, model, args, vocab_stoi, pad_idx, unk_idx, num_sents):
    """Extract sentences based on attention scores from HAN model"""
    model.eval() # Ensure model is in eval mode

    # Split document into sentences using Spacy
    doc = nlp(doc_text)
    sents_text = [sent.text.strip() for sent in doc.sents]

    # Tokenize sentences and convert to indices
    sents_tokens = []
    sents_indices = []
    max_sent_len = args.get('max_sent_len', 100) # Get max sentence length from args

    for sent in sents_text:
        tokens = [tok.text.lower() for tok in nlp.tokenizer(sent)]
        if len(tokens) < 5: continue # Skip very short sentences (adjust threshold if needed)

        sents_tokens.append(tokens) # Keep original tokens for output
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]

        # Pad/truncate sentence
        if len(idx) < max_sent_len:
            idx += [pad_idx] * (max_sent_len - len(idx))
        else:
            idx = idx[:max_sent_len]
        sents_indices.append(idx)

    if not sents_indices:
         warnings.warn("No valid sentences found for HAN attention extraction.", UserWarning)
         return []

    # Pad/truncate document (number of sentences)
    max_doc_len = args.get('max_doc_len', 50) # Get max doc length from args
    if len(sents_indices) < max_doc_len:
        # Create padding tensor correctly
        pad_tensor = torch.full((max_sent_len,), pad_idx, dtype=torch.long)
        sents_indices.extend([pad_tensor.tolist()] * (max_doc_len - len(sents_indices)))
        # Also pad the tokens list with None or empty lists if needed for alignment, though not strictly necessary here
        sents_tokens.extend([[]] * (max_doc_len - len(sents_tokens))) # Pad tokens list for consistency
    elif len(sents_indices) > max_doc_len:
        sents_indices = sents_indices[:max_doc_len]
        sents_tokens = sents_tokens[:max_doc_len] # Truncate tokens list to match indices

    # Convert to tensor
    # Use torch.tensor instead of torch.LongTensor
    doc_tensor = torch.tensor(sents_indices, dtype=torch.long).to(device)
    doc_tensor = doc_tensor.unsqueeze(0)  # Add batch dimension [1, max_doc_len, max_sent_len]

    # Prediction and Attention Score Extraction
    with torch.no_grad():
        # Assuming the HAN model returns (probs, sentence_attention_scores)
        # Ensure model.output_attn was set to True during loading
        outputs = model(doc_tensor)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            probs, attn_score = outputs
        else:
            # Handle cases where model might not return attention or format changed
            warnings.warn("HAN model did not return expected (probs, attn_score) tuple. Cannot extract sentences by attention.", RuntimeWarning)
            # Optionally, try to predict without attention if needed elsewhere
            # probs = outputs if isinstance(outputs, torch.Tensor) else None
            return [] # Cannot extract sentences

    # Process attention scores (assuming attn_score shape [batch=1, num_sents])
    attn_scores_np = attn_score.squeeze().cpu().numpy() # Squeeze batch dim, move to CPU, get numpy

    # Align scores with original sentences (before padding/truncation of doc)
    num_original_sents = len([st for st in sents_tokens if st]) # Count non-empty token lists before padding
    aligned_scores = attn_scores_np[:num_original_sents]
    aligned_sents_tokens = sents_tokens[:num_original_sents]


    if len(aligned_scores) != len(aligned_sents_tokens):
         warnings.warn(f"Attention score length ({len(aligned_scores)}) mismatch with sentence token length ({len(aligned_sents_tokens)}). Cannot reliably extract sentences.", RuntimeWarning)
         return []

    # Create DataFrame and sort by attention
    df = pd.DataFrame({'sent_tokens': aligned_sents_tokens, 'attn': aligned_scores})
    df = df.sort_values(by=['attn'], ascending=False)

    # Get top N sentences
    top_sent_tokens = list(df['sent_tokens'][:num_sents])

    # Reconstruct sentences from tokens
    out_sents = []
    for tokens in top_sent_tokens:
        if not tokens: continue # Skip if padding resulted in empty list
        sent = " ".join(tokens)
        # Basic cleaning (similar to original code)
        sent = re.sub(r" \.", ".", sent)
        sent = re.sub(r" \,", ",", sent)
        sent = re.sub(r"\( ", "(", sent)
        sent = re.sub(r" \)", ")", sent)
        sent = re.sub(r"\[ ", "[", sent)
        sent = re.sub(r" \]", "]", sent)
        sent = re.sub(r" \- ", "-", sent)
        out_sents.append(sent)

    return out_sents

