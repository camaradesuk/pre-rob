#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated for Python 3.10, PyTorch 2.x, Transformers 4.x, SentenceTransformers 2.x
Handles legacy torchtext Field and NestedField objects for model loading.
"""

import json
import dill
import torch
import re
import pandas as pd
import spacy
import warnings
from pathlib import Path
import os
import types
import sys

# --- Shim for legacy torchtext.data.Field and torchtext.data.NestedField ---
class _DummyVocab:
    """
    A dummy class to stand in for torchtext.vocab.Vocab.
    Primarily holds 'stoi' (string-to-index) and 'itos' (index-to-string) mappings.
    """
    def __init__(self, stoi=None, itos=None, pad_token="<pad>", unk_token="<unk>", specials=None):
        self.stoi = stoi if stoi is not None else {"<unk>": 0, "<pad>": 1}
        if itos is None:
            self.itos = {i: s for s, i in self.stoi.items()}
        else:
            self.itos = itos
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.specials = specials if specials is not None else [pad_token, unk_token]

class _DummyFieldBase:
    """
    Base for dummy Field-like objects, holding common attributes.
    """
    def __init__(self, **kwargs):
        self.sequential = kwargs.get('sequential', True)
        self.use_vocab = kwargs.get('use_vocab', True)
        self.init_token = kwargs.get('init_token', None)
        self.eos_token = kwargs.get('eos_token', None)
        self.unk_token = kwargs.get('unk_token', "<unk>")
        self.pad_token = kwargs.get('pad_token', "<pad>")
        self.fix_length = kwargs.get('fix_length', None)
        self.dtype = kwargs.get('dtype', torch.long)
        self.preprocessing = kwargs.get('preprocessing', None)
        self.postprocessing = kwargs.get('postprocessing', None)
        self.lower = kwargs.get('lower', False)
        self.tokenizer_args = kwargs.get('tokenizer_args', {}) # Store tokenizer_args
        # self.tokenize = kwargs.get('tokenize', 'spacy') # Default from original torchtext
        self.tokenize = kwargs.get('tokenize', lambda x: x.split()) # Simplified default
        self.include_lengths = kwargs.get('include_lengths', False)
        self.batch_first = kwargs.get('batch_first', False)
        self.pad_first = kwargs.get('pad_first', False)
        self.truncate_first = kwargs.get('truncate_first', False)
        self.stop_words = kwargs.get('stop_words', None)
        self.is_target = kwargs.get('is_target', False)
        self.vocab_cls = _DummyVocab
        self.vocab = self.vocab_cls(
            stoi=kwargs.get('vocab_stoi', {"<unk>": 0, "<pad>": 1, "dummy_for_nested": 2}), # Ensure some content
            pad_token=self.pad_token,
            unk_token=self.unk_token
        )
        # Capture any other attributes dill might try to set
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

class _DummyField(_DummyFieldBase):
    """
    A dummy class to stand in for torchtext.data.Field.
    It needs to have attributes that the unpickled object expects,
    especially 'vocab' with an 'stoi' map.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This is a simple Field, vocab is directly on it.

class _DummyNestedField(_DummyFieldBase):
    """
    A dummy class to stand in for torchtext.data.NestedField.
    A NestedField typically wraps another Field (nesting_field)
    and tokenizes text into a nested list structure.
    The vocabulary is often associated with the nesting_field.
    """
    def __init__(self, nesting_field_arg, **kwargs): # Renamed to avoid clash if 'nesting_field' is in kwargs
        super().__init__(**kwargs)
        self.nesting_field = nesting_field_arg # This is the first arg to NestedField constructor

        # If the nesting_field itself has vocab, that's the one we usually care about for HAN.
        if hasattr(self.nesting_field, 'vocab') and hasattr(self.nesting_field.vocab, 'stoi'):
            self.vocab = self.nesting_field.vocab
            self.pad_token = getattr(self.nesting_field, 'pad_token', self.pad_token)
            self.unk_token = getattr(self.nesting_field, 'unk_token', self.unk_token)
        # Else, the vocab initialized in _DummyFieldBase is used.

        self.fixup_attrs = True # A flag sometimes used in torchtext legacy

def _split_tokenizer(x): # As in original user code
    return x.split()

# Create the fake module structure for dill to find these classes
torchtext_mod = types.ModuleType("torchtext")
data_mod = types.ModuleType("torchtext.data")
field_mod = types.ModuleType("torchtext.data.field") # dill looks for torchtext.data.field.ClassName
vocab_mod = types.ModuleType("torchtext.vocab")
utils_mod = types.ModuleType("torchtext.data.utils")

# Assign the dummy classes to the shimmed field_mod
field_mod.Field = _DummyField
field_mod.NestedField = _DummyNestedField # Make NestedField available here

# Assign other components
vocab_mod.Vocab = _DummyVocab
utils_mod._split_tokenizer = _split_tokenizer # If pickled objects refer to this

# Link modules
data_mod.field = field_mod
data_mod.utils = utils_mod
torchtext_mod.data = data_mod
torchtext_mod.vocab = vocab_mod

# Update sys.modules BEFORE dill.load is called anywhere
sys.modules["torchtext"] = torchtext_mod
sys.modules["torchtext.data"] = data_mod
sys.modules["torchtext.data.field"] = field_mod # Critical for dill
sys.modules["torchtext.vocab"] = vocab_mod
sys.modules["torchtext.data.utils"] = utils_mod
# --- End Shim ---


# --- Hugging Face Offline Configuration ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError as e:
    print(f"CRITICAL ERROR: spaCy model 'en_core_web_sm' could not be loaded.")
    print(f"Spacy version: {spacy.__version__}. Error: {e}")
    raise RuntimeError("spaCy model 'en_core_web_sm' not found or incompatible.") from e

from transformers import AutoTokenizer, DistilBertModel, AutoConfig, DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util
# Assuming model.py is in the same directory or PYTHONPATH
from model import ConvNet, AttnNet, HAN, DistilClsConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True # For reproducibility
    torch.backends.cudnn.benchmark = False
    # torch.manual_seed(0) # Consider seeding if strict reproducibility is needed across runs

print(f"Using device: {device}")

def ensure_model_on_available_device(model: torch.nn.Module) -> torch.nn.Module:
    """Moves model to CUDA if available, otherwise CPU. Sets to eval mode."""
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = model.to(target_device)
        # Minimal check to see if CUDA ops work if target is CUDA
        if target_device == 'cuda':
            _ = next(model.parameters()).to(target_device) # Try moving a parameter
        print(f"Model successfully moved to {target_device}.")
    except Exception as e:
        warnings.warn(
            f"Failed to move model to {target_device} (reason: {e}). Falling back to CPU.",
            RuntimeWarning,
        )
        model = model.to('cpu')
        print("Model moved to CPU as fallback.")
    model.eval()
    return model

ROB_ITEM_DESCRIPTIONS = {
    'RandomizationTreatmentControl': 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment',
    'BlindedOutcomeAssessment': 'Assessment of an outcome in a blinded fashion. Investigators measuring the outcome do not know which treatment group the animals belongs to and what treatment they had received',
    'SampleSizeCalculation': 'The manuscript reports the performance of a sample size calculation and describes how this number was derived statistically',
    'AnimalExclusions': 'All animals, all data and all outcomes measured are accounted for and presented in the final analysis. Reasons are given for animal exclusions',
    'AllocationConcealment': 'Investigators performing the experiment do not know which treatment an animal is being given',
    'AnimalWelfareRegulations': 'Research investigators complied with animal welfare regulations',
    'ConflictsOfInterest': 'Potential conflict of interest, like funding or affiliation to a pharmaceutical company'
}

HF_HOME_DEFAULT = Path.home() / ".cache" / "huggingface"
HF_HOME_PATH = Path(os.environ.get('HF_HOME', HF_HOME_DEFAULT))
TRANSFORMERS_CACHE_PATH = Path(os.environ.get('TRANSFORMERS_CACHE', HF_HOME_PATH / "hub"))
SENTENCE_TRANSFORMERS_HOME_PATH = Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', HF_HOME_PATH / "sentence_transformers"))

print(f"Expecting Hugging Face models (transformers lib) in TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE_PATH}")
print(f"Expecting SentenceTransformer models in SENTENCE_TRANSFORMERS_HOME: {SENTENCE_TRANSFORMERS_HOME_PATH}")

def load_vocab_info(fld_path: Path) -> tuple[dict, int, int]:
    """
    Loads vocabulary information (stoi, pad_idx, unk_idx) from a pickled .Field file.
    Handles both simple Field and NestedField objects using the shims.
    """
    try:
        with open(fld_path, "rb") as fin:
            # The shims for Field and NestedField should be active here via sys.modules
            loaded_obj = dill.load(fin)

        vocab_source_obj = None
        source_type_msg = "Unknown"

        # Check if it's like a NestedField (common for HAN models)
        if hasattr(loaded_obj, 'nesting_field') and hasattr(loaded_obj.nesting_field, 'vocab'):
            vocab_source_obj = loaded_obj.nesting_field
            source_type_msg = f"NestedField (using nesting_field: {type(vocab_source_obj).__name__})"
        # Check if it's like a simple Field
        elif hasattr(loaded_obj, 'vocab'):
            vocab_source_obj = loaded_obj
            source_type_msg = f"Field (direct: {type(vocab_source_obj).__name__})"
        else:
            raise ValueError(f"Loaded object from {fld_path.name} lacks 'vocab' or 'nesting_field.vocab'. Type: {type(loaded_obj)}")

        if not hasattr(vocab_source_obj, 'vocab') or not hasattr(vocab_source_obj.vocab, 'stoi'):
            raise ValueError(f"Vocab source object (from {source_type_msg}) lacks 'vocab.stoi'. Type: {type(vocab_source_obj.vocab)}")

        vocab_stoi = vocab_source_obj.vocab.stoi
        pad_token = getattr(vocab_source_obj, 'pad_token', "<pad>")
        unk_token = getattr(vocab_source_obj, 'unk_token', "<unk>")

        pad_idx = vocab_stoi.get(pad_token)
        if pad_idx is None: # Fallback if pad_token isn't in stoi (e.g. if stoi is minimal)
            pad_idx = 1 # Common default
            warnings.warn(f"Pad token '{pad_token}' not in vocab.stoi for {fld_path.name}. Using default pad_idx={pad_idx}.", UserWarning)

        unk_idx = vocab_stoi.get(unk_token)
        if unk_idx is None: # Fallback
            unk_idx = 0 # Common default
            warnings.warn(f"Unk token '{unk_token}' not in vocab.stoi for {fld_path.name}. Using default unk_idx={unk_idx}.", UserWarning)

        print(f"Successfully loaded vocab from {fld_path.name} (detected as {source_type_msg}). Vocab size: {len(vocab_stoi)}")
        return vocab_stoi, pad_idx, unk_idx

    except FileNotFoundError:
        print(f"Error: Vocab field file not found at {fld_path}")
        raise
    except Exception as e:
        print(f"Error loading or processing vocab file {fld_path}: {e}")
        raise


def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    """Loads legacy models (ConvNet, AttnNet, HAN)."""
    with open(arg_path) as f:
        args = json.load(f)['args']

    vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
    vocab_size = len(vocab_stoi)
    # Ensure vocab_size for embedding layer is sufficient, matching original logic
    if 'max_vocab_size' in args and (args['max_vocab_size'] + 2) > vocab_size:
         actual_vocab_size_for_model = args['max_vocab_size'] + 2
         # print(f"Using max_vocab_size from args: {actual_vocab_size_for_model} for {Path(arg_path).name}")
    else:
         actual_vocab_size_for_model = vocab_size
         # print(f"Using actual vocab_size: {actual_vocab_size_for_model} for {Path(arg_path).name}")


    net_type = args.get('net_type', 'unknown')
    model: torch.nn.Module
    if net_type == 'cnn':
        sizes = [int(s) for s in args['filter_sizes'].split(',')]
        model = ConvNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args['embed_dim'], n_filters=args['num_filters'],
                        filter_sizes=sizes, output_dim=2, dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'])
    elif net_type == 'attn':
        model = AttnNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args['embed_dim'], rnn_hidden_dim=args['rnn_hidden_dim'],
                        rnn_num_layers=args['rnn_num_layers'], output_dim=2, bidirection=args['bidirection'],
                        rnn_cell_type=args['rnn_cell_type'], dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'],
                        output_attn=args.get('output_attn', False)) # HAN model needs output_attn=True
    elif net_type == 'han':
        model = HAN(vocab_size=actual_vocab_size_for_model, embedding_dim=args['embed_dim'], word_hidden_dim=args['word_hidden_dim'],
                    word_num_layers=args['word_num_layers'], pad_idx=pad_idx, embed_trainable=args['embed_trainable'],
                    batch_norm=args['batch_norm'], sent_hidden_dim=args['sent_hidden_dim'],
                    sent_num_layers=args['sent_num_layers'], output_dim=2,
                    output_attn=args.get('output_attn', True)) # HAN expects to output attention
    else:
        raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path}")

    try:
        # Try loading with weights_only=False first, as these are pickled modules not just state_dicts
        checkpoint = torch.load(pth_path, map_location="cpu") # Load to CPU first
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {pth_path}")
        raise
    except Exception as e: # Catch other torch.load errors
        print(f"Error loading checkpoint {pth_path}: {e}. Trying with weights_only=True as fallback.")
        try:
            checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        except Exception as e_fallback:
            print(f"Fallback with weights_only=True also failed for {pth_path}: {e_fallback}")
            raise e_fallback


    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict):
         warnings.warn(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key. Assuming it IS the state_dict.", UserWarning)
         state_dict = checkpoint # If the whole file is the state_dict
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
         warnings.warn(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key. Assuming the dict itself is the state_dict.", UserWarning)
         state_dict = checkpoint # If the dict is the state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model state_dict from: {pth_path.name} for {net_type} model.")
    except RuntimeError as e:
        print(f"Error loading state_dict into {net_type} model from {pth_path.name} (strict=True): {e}. Trying strict=False.")
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded state_dict with strict=False for {pth_path.name}.")
        except RuntimeError as e_false:
            print(f"Error loading state_dict into {net_type} model from {pth_path.name} (strict=False): {e_false}")
            raise e_false

    model = ensure_model_on_available_device(model)
    return model, args, vocab_stoi, pad_idx, unk_idx


def load_model_bert(arg_path: Path, pth_path: Path):
    """Loads BERT-based DistilClsConv model."""
    with open(arg_path) as f:
        args = json.load(f)['args']

    rob_item = args.get('rob_item')
    rob_sent = args.get('rob_sent')
    if rob_sent is None and rob_item in ROB_ITEM_DESCRIPTIONS:
        rob_sent = ROB_ITEM_DESCRIPTIONS[rob_item]
    elif rob_sent is None:
        raise ValueError(f"RoB item description for '{rob_item}' not found in args or predefined descriptions.")

    distilbert_model_name_or_path = 'distilbert-base-uncased'
    try:
        config = AutoConfig.from_pretrained(
            distilbert_model_name_or_path,
            num_labels=2, # Assuming binary classification
            return_dict=True,
            cache_dir=str(TRANSFORMERS_CACHE_PATH),
            local_files_only=True # Enforce offline usage
        )
        # Initialize custom model with this config
        model = DistilClsConv(config)
        print(f"Initialized DistilClsConv architecture using config from '{distilbert_model_name_or_path}' (cache: {TRANSFORMERS_CACHE_PATH}).")

        # Load pre-trained weights for the DistilBertModel part from Hugging Face cache
        # This step ensures the base DistilBERT weights are loaded before applying fine-tuned weights.
        distilbert_base_model = DistilBertModel.from_pretrained(
            distilbert_model_name_or_path,
            config=config, # Pass the same config
            cache_dir=str(TRANSFORMERS_CACHE_PATH),
            local_files_only=True
        )
        model.distilbert.load_state_dict(distilbert_base_model.state_dict())
        print(f"Loaded pre-trained weights for 'distilbert' part of DistilClsConv from cache.")
        del distilbert_base_model # Free memory

        # Load fine-tuned checkpoint for the whole DistilClsConv model
        checkpoint = torch.load(pth_path, map_location="cpu") # Load to CPU first
        print(f"Loaded fine-tuned model state_dict from: {pth_path.name}")

    except Exception as e:
        print(f"Error during DistilClsConv initialization or loading base DistilBertModel weights: {e}")
        print(f"Checked TRANSFORMERS_CACHE_PATH: {TRANSFORMERS_CACHE_PATH}")
        raise

    state_dict = checkpoint.get('state_dict', checkpoint) # Handle checkpoints that might be raw state_dicts
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict) :
        warnings.warn(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key. Assuming it IS the state_dict.", UserWarning)
        state_dict = checkpoint
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
        warnings.warn(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key. Assuming the dict itself is the state_dict.", UserWarning)
        state_dict = checkpoint


    # Load the state_dict into the DistilClsConv model
    # Using strict=False because the checkpoint might only contain weights for parts of the model
    # (e.g., if only the classifier head was fine-tuned on top of pre-trained DistilBERT)
    # Or, if the checkpoint is from the original training, it should match.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading fine-tuned state_dict for {pth_path.name}: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading fine-tuned state_dict for {pth_path.name}: {unexpected_keys}")
    print(f"Fine-tuned state_dict loaded into DistilClsConv from {pth_path.name}.")


    tokenizer = DistilBertTokenizer.from_pretrained(
        distilbert_model_name_or_path,
        use_fast=True, # Prefer fast tokenizer
        cache_dir=str(TRANSFORMERS_CACHE_PATH),
        local_files_only=True
    )
    try:
        sent_model_name = 'distilbert-base-nli-stsb-mean-tokens'
        sent_model = SentenceTransformer(
            sent_model_name,
            device='cpu', # Load to CPU first, will be moved by ensure_model_on_available_device
            cache_folder=str(SENTENCE_TRANSFORMERS_HOME_PATH)
            # local_files_only=True # SentenceTransformer doesn't have this arg directly, relies on cache_folder
        )
        print(f"Loaded SentenceTransformer model: {sent_model_name} (using cache: {SENTENCE_TRANSFORMERS_HOME_PATH})")
    except Exception as e:
        print(f"Error loading SentenceTransformer model {sent_model_name}: {e}")
        print(f"Checked SENTENCE_TRANSFORMERS_HOME_PATH: {SENTENCE_TRANSFORMERS_HOME_PATH}")
        raise

    model = ensure_model_on_available_device(model)
    sent_model = ensure_model_on_available_device(sent_model) # Also move sentence_model to device
    return model, tokenizer, sent_model, rob_sent


def pred_legacy(doc: str, model: torch.nn.Module, args: dict,
                vocab_stoi: dict, pad_idx: int, unk_idx: int) -> float:
    """Prediction function for legacy models."""
    model.eval() # Ensure model is in evaluation mode
    current_device = next(model.parameters()).device

    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]
    idx = [vocab_stoi.get(t, unk_idx) for t in tokens]

    max_len = args.get('max_token_len', 512) # Default from original, ensure it's in args
    if len(idx) < max_len:
        idx += [pad_idx] * (max_len - len(idx))
    elif len(idx) > max_len:
        idx = idx[:max_len]

    doc_tensor = torch.tensor(idx, dtype=torch.long).to(current_device)
    doc_tensor = doc_tensor.unsqueeze(1) # Original shape [seq_len, batch_size=1]

    with torch.no_grad():
        probs_output = model(doc_tensor) # Model output depends on its structure

    # Assuming model output is [batch_size, num_classes] or similar for classification
    # For binary classification, probs_output[0][1] is prob of positive class
    if probs_output.ndim == 2 and probs_output.shape[0] == 1 and probs_output.shape[1] >= 2:
        prob_positive = probs_output.squeeze(0)[1].item() # .item() to get Python float
    elif probs_output.ndim == 1 and probs_output.shape[0] >=2: # If output was already squeezed
        prob_positive = probs_output[1].item()
    else:
        warnings.warn(f"Unexpected output shape from legacy model: {probs_output.shape}. Cannot extract positive class probability reliably.", UserWarning)
        prob_positive = 0.0 # Fallback

    return float(prob_positive)


def pred_bert(text: str, model: torch.nn.Module, tokenizer,
              sent_model: SentenceTransformer, rob_sent: str, max_n_sent: int = 30) -> float:
    """Prediction function for BERT-based DistilClsConv model."""
    model.eval() # Ensure model is in evaluation mode
    current_device = next(model.parameters()).device
    sent_model.to(current_device) # Ensure sentence_model is on the same device

    # Limit sentence count to prevent OOM, take more initially for better selection
    sents_spacy = list(nlp(text).sents)
    sents = [str(s).strip() for s in sents_spacy if len(str(s).split()) > 3][:max_n_sent * 5] # Increased pool for selection

    if not sents:
        warnings.warn("No sentences found after filtering in pred_bert. Returning 0.0.", UserWarning)
        return 0.0

    try:
        # Encode sentences in batches if too many to avoid OOM with SentenceTransformer
        batch_size_st = 128 # Adjust based on GPU memory
        all_sent_embeds = []
        for i in range(0, len(sents), batch_size_st):
            batch_sents = sents[i:i+batch_size_st]
            batch_embeds = sent_model.encode(batch_sents, convert_to_tensor=True, device=current_device, show_progress_bar=False)
            all_sent_embeds.append(batch_embeds)
        sent_embeds = torch.cat(all_sent_embeds, dim=0)

        rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True, device=current_device, show_progress_bar=False)
    except Exception as e:
        print(f"Error during sentence encoding in pred_bert: {e}")
        return 0.0 # Fallback

    cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed)
    top_k = min(max_n_sent, len(sents))
    if top_k == 0:
        warnings.warn("No sentences to select for top_k in pred_bert (top_k is 0). Returning 0.0.", UserWarning)
        return 0.0

    # Ensure cos_scores is 1D before topk if it's not already
    if cos_scores.ndim > 1:
        cos_scores = cos_scores.squeeze()
        if cos_scores.ndim == 0: # if only one sentence, make it 1D
             cos_scores = cos_scores.unsqueeze(0)


    if cos_scores.numel() < top_k : # If fewer scores than top_k requested
        warnings.warn(f"Fewer sentences available ({cos_scores.numel()}) than top_k ({top_k}) in pred_bert. Using all available.", UserWarning)
        top_k = cos_scores.numel()
        if top_k == 0: return 0.0


    top_vals, top_indices = torch.topk(cos_scores, k=top_k, largest=True, sorted=True)
    top_indices_np = top_indices.cpu().numpy()
    sim_sents = [sents[i] for i in top_indices_np]
    sim_text = " ".join(sim_sents)

    if not sim_text.strip():
        warnings.warn("Sim_text (concatenated similar sentences) is empty before tokenization in pred_bert. Returning 0.0.", UserWarning)
        return 0.0

    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(current_device) for k, v in inputs.items()} # Move inputs to model's device

    with torch.no_grad():
        outputs = model(**inputs) # DistilClsConv returns probabilities directly

    # Assuming outputs is already [batch_size, num_classes] with softmax applied
    # For binary classification, probs_output[0][1] is prob of positive class
    if outputs.ndim == 2 and outputs.shape[0] == 1 and outputs.shape[1] >= 2:
        prob_positive = outputs.squeeze(0)[1].item()
    else:
        warnings.warn(f"Unexpected output shape from BERT model: {outputs.shape}. Cannot extract positive class probability reliably.", UserWarning)
        prob_positive = 0.0 # Fallback

    return float(prob_positive)


def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict,
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, num_sents: int) -> list[str]:
    """Extracts relevant sentences using a HAN model and its attention scores."""
    if num_sents <= 0: return []
    model.eval() # Ensure model is in evaluation mode
    current_device = next(model.parameters()).device

    doc_spacy = nlp(doc_text)
    sents_text_orig = [sent.text.strip() for sent in doc_spacy.sents]

    sents_tokens_for_model = [] # List of token lists for model input
    sents_text_for_reconstruction = [] # Corresponding original sentence text

    max_sent_len = args.get('max_sent_len', 100) # Default from original HAN paper often around 50-100

    for sent_text_item in sents_text_orig:
        tokens = [tok.text.lower() for tok in nlp.tokenizer(sent_text_item)]
        if len(tokens) < 5 or len(tokens) > max_sent_len + 20 : continue # Filter very short/long sentences (original had >10)

        sents_text_for_reconstruction.append(sent_text_item) # Store original text
        # Tokenize and numericalize for model
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        if len(idx) < max_sent_len:
            idx += [pad_idx] * (max_sent_len - len(idx))
        else:
            idx = idx[:max_sent_len] # Truncate
        sents_tokens_for_model.append(idx)

    if not sents_tokens_for_model:
        warnings.warn("No valid sentences found for HAN attention extraction after filtering.", UserWarning)
        return []

    max_doc_len = args.get('max_doc_len', 50) # Max sentences per document for HAN model

    # Prepare batch for HAN model
    actual_sents_indices_for_input = sents_tokens_for_model[:max_doc_len]
    actual_sents_text_for_output = sents_text_for_reconstruction[:max_doc_len]

    # Pad document to max_doc_len if fewer sentences
    if len(actual_sents_indices_for_input) < max_doc_len:
        pad_sentence_tensor_list = [pad_idx] * max_sent_len
        actual_sents_indices_for_input.extend(
            [pad_sentence_tensor_list] * (max_doc_len - len(actual_sents_indices_for_input))
        )

    doc_tensor = torch.tensor(actual_sents_indices_for_input, dtype=torch.long).to(current_device)
    doc_tensor = doc_tensor.unsqueeze(0) # HAN expects [batch_size, num_sents, sent_len]

    with torch.no_grad():
        outputs = model(doc_tensor) # HAN model should return (probs, sentence_attention_scores)

    if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], torch.Tensor)):
        warnings.warn("HAN model did not return expected (probs, attn_score_tensor) tuple "
                      "or attn_score_tensor is not a tensor. Cannot extract sentences by attention.", RuntimeWarning)
        return ["Error: HAN model output format unexpected."]

    _probs, attn_score_tensor = outputs # Probs are not used here, only sentence attention
    attn_scores_np = attn_score_tensor.squeeze().cpu().numpy() # Squeeze batch, move to CPU, to numpy

    # Align attention scores with the original sentences that were fed to the model
    num_sents_fed_to_model = len(actual_sents_text_for_output)
    aligned_scores = attn_scores_np[:num_sents_fed_to_model]

    if len(aligned_scores) != num_sents_fed_to_model:
        warnings.warn(f"Attention score length ({len(aligned_scores)}) mismatch with "
                      f"number of sentences fed to model ({num_sents_fed_to_model}). Cannot reliably extract.", RuntimeWarning)
        return ["Error: Attention score mismatch."]

    # Create a DataFrame to sort sentences by attention score
    df = pd.DataFrame({
        'sent_text': actual_sents_text_for_output, # Use original sentence text
        'attn': aligned_scores
    })
    df = df.sort_values(by=['attn'], ascending=False) # Sort by attention, highest first

    # Get the text of the top N sentences
    top_sents_text = list(df['sent_text'][:num_sents])

    # Basic cleaning of extracted sentences (already done on original text if needed)
    # The original code had regex cleaning here, but it's better if sents_text_for_reconstruction is clean
    # For consistency, apply similar cleaning if desired:
    cleaned_top_sents = []
    for sent in top_sents_text:
        sent = re.sub(r" \.", ".", sent); sent = re.sub(r" \,", ",", sent)
        sent = re.sub(r"\( ", "(", sent); sent = re.sub(r" \)", ")", sent)
        sent = re.sub(r"\[ ", "[", sent); sent = re.sub(r" \]", "]", sent)
        sent = re.sub(r" \- ", "-", sent)
        cleaned_top_sents.append(sent.strip())
        
    return cleaned_top_sents
