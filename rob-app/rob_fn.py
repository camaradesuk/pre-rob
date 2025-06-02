#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated for Python 3.10, PyTorch 2.x, Transformers 4.x, SentenceTransformers 2.x.
Handles legacy torchtext Field and NestedField objects for model loading.
Forces output_attn=True for HAN sentence models for attention extraction.
Includes enhanced error handling, logging, and documentation.
Modified to accept nlp_instance for multiprocessing compatibility.
"""

import json
import dill
import torch
import re
import pandas as pd
# spacy will be imported and used via nlp_instance passed to functions
import warnings
from pathlib import Path
import os
import types
import sys
import logging 

# --- Global constants ---
FILE_PROCESSING_TIMEOUT_SECONDS = 300  # Timeout threshold in seconds (e.g., 5 minutes)
ROB_ITEM_CATEGORIES = ['random', 'blind', 'interest', 'welfare', 'exclusion']


# --- Logger for this module ---
logger = logging.getLogger(__name__)


# --- Shim for legacy torchtext.data.Field and torchtext.data.NestedField ---
# This section remains largely the same.
class _DummyVocab:
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
        self.tokenizer_args = kwargs.get('tokenizer_args', {}) 
        self.tokenize = kwargs.get('tokenize', lambda x: x.split()) 
        self.include_lengths = kwargs.get('include_lengths', False)
        self.batch_first = kwargs.get('batch_first', False)
        self.pad_first = kwargs.get('pad_first', False)
        self.truncate_first = kwargs.get('truncate_first', False)
        self.stop_words = kwargs.get('stop_words', None)
        self.is_target = kwargs.get('is_target', False)
        self.vocab_cls = _DummyVocab
        self.vocab = self.vocab_cls(
            stoi=kwargs.get('vocab_stoi', {"<unk>": 0, "<pad>": 1, "dummy_for_nested": 2}),
            pad_token=self.pad_token,
            unk_token=self.unk_token
        )
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

class _DummyField(_DummyFieldBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class _DummyNestedField(_DummyFieldBase):
    def __init__(self, nesting_field_arg, **kwargs): 
        super().__init__(**kwargs)
        self.nesting_field = nesting_field_arg 
        if hasattr(self.nesting_field, 'vocab') and hasattr(self.nesting_field.vocab, 'stoi'):
            self.vocab = self.nesting_field.vocab
            self.pad_token = getattr(self.nesting_field, 'pad_token', self.pad_token)
            self.unk_token = getattr(self.nesting_field, 'unk_token', self.unk_token)
        else:
            logger.debug("_DummyNestedField: nesting_field does not have expected vocab attributes.")
        self.fixup_attrs = True 

def _split_tokenizer(x: str) -> list[str]:
    return x.split()

torchtext_mod = types.ModuleType("torchtext")
data_mod = types.ModuleType("torchtext.data")
field_mod = types.ModuleType("torchtext.data.field") 
vocab_mod = types.ModuleType("torchtext.vocab")
utils_mod = types.ModuleType("torchtext.data.utils")

field_mod.Field = _DummyField
field_mod.NestedField = _DummyNestedField 
vocab_mod.Vocab = _DummyVocab
utils_mod._split_tokenizer = _split_tokenizer 

data_mod.field = field_mod 
data_mod.utils = utils_mod
torchtext_mod.data = data_mod
torchtext_mod.vocab = vocab_mod

sys.modules["torchtext"] = torchtext_mod
sys.modules["torchtext.data"] = data_mod
sys.modules["torchtext.data.field"] = field_mod 
sys.modules["torchtext.vocab"] = vocab_mod
sys.modules["torchtext.data.utils"] = utils_mod
# --- End Shim ---

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
logger.info("TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE set to 1.")

# --- Transformer and SentenceTransformer Imports ---
try:
    from transformers import AutoTokenizer, DistilBertModel, AutoConfig, DistilBertTokenizer
    from sentence_transformers import SentenceTransformer, util
    from model import ConvNet, AttnNet, HAN, DistilClsConv 
    logger.info("Successfully imported Transformers and SentenceTransformer modules.")
except ImportError as e:
    logger.critical(f"Failed to import Hugging Face or local model modules: {e}", exc_info=True)
    raise

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    logger.info(f"Using device: {device} (CUDA). CuDNN deterministic=True, benchmark=False.")
else:
    logger.info(f"Using device: {device} (CPU).")


def ensure_model_on_available_device(model: torch.nn.Module) -> torch.nn.Module:
    target_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = model.to(target_device_str)
        if target_device_str == 'cuda' and next(model.parameters()).device.type != 'cuda':
             warnings.warn(f"Model {type(model).__name__} parameters not on CUDA after .to('cuda'). Forcing.", RuntimeWarning)
             model.cuda() 
        logger.debug(f"Model {type(model).__name__} moved to {target_device_str}.")
    except Exception as e:
        warnings.warn(
            f"Failed to move model {type(model).__name__} to {target_device_str} (reason: {e}). Falling back to CPU.",
            RuntimeWarning,
        )
        logger.warning(f"Falling back to CPU for model {type(model).__name__} due to error: {e}", exc_info=True)
        model = model.to('cpu')
    model.eval() 
    return model

# --- Constants for RoB Item Descriptions and Cache Paths ---
# ROB_ITEM_DESCRIPTIONS is defined in rob.py's main section if needed for model loading args.
# For rob_fn.py, the direct rob_sent string is passed to pred_bert.
# However, if it were needed here for some reason:
# ROB_ITEM_DESCRIPTIONS_LEGACY = {
#     'RandomizationTreatmentControl': 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment',
#     # ... other descriptions
# }

HF_HOME_DEFAULT = Path.home() / ".cache" / "huggingface"
HF_HOME_PATH = Path(os.environ.get('HF_HOME', HF_HOME_DEFAULT))
TRANSFORMERS_CACHE_PATH = Path(os.environ.get('TRANSFORMERS_CACHE', HF_HOME_PATH / "hub"))
SENTENCE_TRANSFORMERS_HOME_PATH = Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', HF_HOME_PATH / "sentence_transformers"))
logger.info(f"Hugging Face cache paths: TRANSFORMERS_CACHE_PATH={TRANSFORMERS_CACHE_PATH}, SENTENCE_TRANSFORMERS_HOME_PATH={SENTENCE_TRANSFORMERS_HOME_PATH}")


def load_vocab_info(fld_path: Path) -> tuple[dict, int, int]:
    if not fld_path.exists():
        logger.error(f"Vocab field file not found at {fld_path}")
        raise FileNotFoundError(f"Vocab field file not found at {fld_path}")
    try:
        with open(fld_path, "rb") as fin:
            loaded_obj = dill.load(fin)
        vocab_source_obj = None
        source_type_msg = "Unknown"
        if hasattr(loaded_obj, 'nesting_field') and hasattr(loaded_obj.nesting_field, 'vocab'):
            vocab_source_obj = loaded_obj.nesting_field
            source_type_msg = f"NestedField (using nesting_field: {type(vocab_source_obj).__name__})"
        elif hasattr(loaded_obj, 'vocab'):
            vocab_source_obj = loaded_obj
            source_type_msg = f"Field (direct: {type(vocab_source_obj).__name__})"
        else:
            err_msg = f"Loaded object from {fld_path.name} (type: {type(loaded_obj)}) lacks 'vocab' or 'nesting_field.vocab'."
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.debug(f"Vocab source determined: {source_type_msg} for {fld_path.name}")
        if not hasattr(vocab_source_obj, 'vocab') or not hasattr(vocab_source_obj.vocab, 'stoi'):
            err_msg = f"Vocab source object (from {source_type_msg}, type: {type(vocab_source_obj.vocab if hasattr(vocab_source_obj, 'vocab') else None)}) lacks 'vocab.stoi'."
            logger.error(err_msg)
            raise ValueError(err_msg)
        vocab_stoi = vocab_source_obj.vocab.stoi
        pad_token = getattr(vocab_source_obj, 'pad_token', "<pad>")
        unk_token = getattr(vocab_source_obj, 'unk_token', "<unk>")
        pad_idx = vocab_stoi.get(pad_token)
        unk_idx = vocab_stoi.get(unk_token)
        if pad_idx is None:
            logger.warning(f"Pad token '{pad_token}' not found in vocab_stoi from {fld_path.name}. Defaulting to 1.")
            pad_idx = vocab_stoi.get("<pad>", 1) 
            if pad_idx not in vocab_stoi.values():
                 raise ValueError(f"Default pad_idx 1 not found in vocab values after '{pad_token}' was missing.")
        if unk_idx is None:
            logger.warning(f"Unknown token '{unk_token}' not found in vocab_stoi from {fld_path.name}. Defaulting to 0.")
            unk_idx = vocab_stoi.get("<unk>", 0) 
            if unk_idx not in vocab_stoi.values():
                raise ValueError(f"Default unk_idx 0 not found in vocab values after '{unk_token}' was missing.")
        logger.info(f"Successfully loaded vocab from {fld_path.name}. Vocab size: {len(vocab_stoi)}, pad_idx: {pad_idx}, unk_idx: {unk_idx}")
        return vocab_stoi, pad_idx, unk_idx
    except FileNotFoundError: 
        raise
    except Exception as e:
        logger.error(f"Error loading or processing vocab file {fld_path}: {e}", exc_info=True)
        raise


def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    logger.info(f"Loading legacy model: args='{arg_path.name}', weights='{pth_path.name}', vocab='{fld_path.name}'")
    if not arg_path.exists(): raise FileNotFoundError(f"Argument file not found: {arg_path}")
    if not pth_path.exists(): raise FileNotFoundError(f"Model weights file not found: {pth_path}")
    with open(arg_path) as f:
        args_json = json.load(f).get('args')
        if not args_json:
            raise ValueError(f"Could not find 'args' key in JSON file: {arg_path}")
    vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
    vocab_size = len(vocab_stoi)
    actual_vocab_size_for_model = vocab_size
    if 'max_vocab_size' in args_json and args_json['max_vocab_size'] > 0 : 
        configured_max_size = args_json['max_vocab_size'] + 2 
        if configured_max_size < vocab_size:
            actual_vocab_size_for_model = configured_max_size
            logger.info(f"Using configured 'max_vocab_size' from args: {configured_max_size} (original vocab size: {vocab_size})")
    net_type = args_json.get('net_type', 'unknown').lower()
    model: torch.nn.Module
    default_output_attn = True if net_type == 'han' else args_json.get('output_attn', False)
    current_output_attn_setting = args_json.get('output_attn', default_output_attn)
    fn_lower = arg_path.name.lower()
    is_sentence_han_model = net_type == 'han' and any(fn_lower.startswith(p) for p in ['hr_', 'hb_', 'hi_', 'hw_', 'he_'])
    if is_sentence_han_model:
        if not current_output_attn_setting:
            logger.info(f"OVERRIDE_HAN_LOAD: For sentence model config '{arg_path.name}', 'output_attn' was {args_json.get('output_attn', 'Not Set')}. Forcing to True for sentence extraction.")
        current_output_attn_setting = True 
    elif net_type == 'han':
         logger.debug(f"DEBUG_HAN_LOAD: For HAN (non-sentence) model '{arg_path.name}', 'output_attn' resolved to: {current_output_attn_setting} (from JSON: {args_json.get('output_attn')}, default if key missing: {default_output_attn})")
    if net_type == 'cnn':
        sizes_str = args_json.get('filter_sizes', "3,4,5") 
        sizes = [int(s) for s in sizes_str.split(',')]
        model = ConvNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], 
                        n_filters=args_json['num_filters'], filter_sizes=sizes, output_dim=2, 
                        dropout=args_json['dropout'], pad_idx=pad_idx,
                        embed_trainable=args_json.get('embed_trainable', False), 
                        batch_norm=args_json.get('batch_norm', False)) 
    elif net_type == 'attn':
        model = AttnNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], 
                        rnn_hidden_dim=args_json['rnn_hidden_dim'], rnn_num_layers=args_json['rnn_num_layers'], 
                        output_dim=2, bidirection=args_json.get('bidirection', True), 
                        rnn_cell_type=args_json.get('rnn_cell_type', 'gru'), 
                        dropout=args_json['dropout'], pad_idx=pad_idx, 
                        embed_trainable=args_json.get('embed_trainable', False), 
                        batch_norm=args_json.get('batch_norm', False),
                        output_attn=current_output_attn_setting)
    elif net_type == 'han':
        model = HAN(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], 
                    word_hidden_dim=args_json['word_hidden_dim'], word_num_layers=args_json['word_num_layers'], 
                    pad_idx=pad_idx, embed_trainable=args_json.get('embed_trainable', False),
                    batch_norm=args_json.get('batch_norm', False), 
                    sent_hidden_dim=args_json['sent_hidden_dim'], sent_num_layers=args_json['sent_num_layers'], 
                    output_dim=2, output_attn=current_output_attn_setting,
                    dropout_rate=args_json.get('dropout', 0.5)) # Pass dropout to HAN
    else:
        err_msg = f"Unsupported net_type '{net_type}' in args file {arg_path.name}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    logger.debug(f"Instantiated legacy model type: {net_type} for {arg_path.name}")
    try:
        checkpoint = torch.load(pth_path, map_location="cpu") 
    except Exception as e:
        logger.warning(f"Standard torch.load failed for {pth_path.name}: {e}. Trying with weights_only=True.")
        try:
            checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        except Exception as e_fallback:
            logger.error(f"Fallback torch.load with weights_only=True also failed for {pth_path.name}: {e_fallback}", exc_info=True)
            raise RuntimeError(f"Failed to load model checkpoint from {pth_path.name}") from e_fallback
    state_dict = checkpoint.get('state_dict', checkpoint) 
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict):
         logger.warning(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key. Type: {type(checkpoint)}")
         state_dict = checkpoint
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
         logger.warning(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key.")
         state_dict = checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.debug(f"Successfully loaded state_dict into {net_type} model (strict=True) from {pth_path.name}")
    except RuntimeError as e_strict:
        logger.warning(f"Error loading state_dict (strict=True) for {net_type} model from {pth_path.name}: {e_strict}. Trying strict=False.")
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded state_dict (strict=False) for {net_type} model from {pth_path.name}.")
        except RuntimeError as e_false:
            logger.error(f"Error loading state_dict (strict=False) for {net_type} model from {pth_path.name}: {e_false}", exc_info=True)
            raise RuntimeError(f"Failed to load state_dict into {net_type} model from {pth_path.name}") from e_false
    model = ensure_model_on_available_device(model)
    logger.info(f"Legacy model '{arg_path.name}' loaded and moved to device: {next(model.parameters()).device}")
    return model, args_json, vocab_stoi, pad_idx, unk_idx


def load_model_bert(arg_path: Path, pth_path: Path):
    logger.info(f"Loading BERT model: args='{arg_path.name}', weights='{pth_path.name}'")
    if not arg_path.exists(): raise FileNotFoundError(f"Argument file not found: {arg_path}")
    if not pth_path.exists(): raise FileNotFoundError(f"Model weights file not found: {pth_path}")
    with open(arg_path) as f:
        args = json.load(f).get('args')
        if not args:
            raise ValueError(f"Could not find 'args' key in JSON file: {arg_path}")
    rob_item = args.get('rob_item') # This might be used to fetch rob_sent if not directly in args
    rob_sent = args.get('rob_sent') 
    # In rob.py, ROB_ITEM_DESCRIPTIONS is used to populate rob_sent if it's None.
    # Here, we assume rob_sent is provided or handled before calling this.
    # If rob_sent is critical and might be None, add a check.
    if rob_sent is None:
        # Attempt to use ROB_ITEM_DESCRIPTIONS_LEGACY if defined and rob_item exists
        # This part depends on how rob_sent is expected to be derived if not in args.
        # For now, assume rob_sent is correctly passed or derived by the caller.
        logger.warning(f"rob_sent is None for BERT model {arg_path.name}. Ensure it's correctly set by the caller.")
        # raise ValueError(f"RoB item description (rob_sent) is missing for BERT model {arg_path.name}")

    distilbert_model_name_or_path = 'distilbert-base-uncased' 
    logger.debug(f"Using base DistilBert model: {distilbert_model_name_or_path}")
    try:
        config = AutoConfig.from_pretrained(
            distilbert_model_name_or_path, num_labels=2, return_dict=True,
            cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
        )
        model = DistilClsConv(config)
        logger.debug(f"Instantiated DistilClsConv model with config from {distilbert_model_name_or_path}.")
        distilbert_base_model = DistilBertModel.from_pretrained(
            distilbert_model_name_or_path, config=config,
            cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
        )
        model.distilbert.load_state_dict(distilbert_base_model.state_dict())
        del distilbert_base_model 
        logger.debug(f"Loaded base DistilBert weights into DistilClsConv.distilbert.")
        checkpoint = torch.load(pth_path, map_location="cpu") 
    except Exception as e:
        logger.error(f"Error during DistilClsConv initialization or loading base weights: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize DistilClsConv or load base weights.") from e
    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict) :
        logger.warning(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key.")
        state_dict = checkpoint
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
        logger.warning(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key.")
        state_dict = checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading fine-tuned state_dict for {pth_path.name}: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading fine-tuned state_dict for {pth_path.name}: {unexpected_keys}")
    logger.debug(f"Loaded fine-tuned state_dict from {pth_path.name} into DistilClsConv (strict=False).")
    tokenizer = DistilBertTokenizer.from_pretrained(
        distilbert_model_name_or_path, use_fast=True,
        cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
    )
    logger.debug(f"Loaded DistilBertTokenizer for {distilbert_model_name_or_path}.")
    try:
        sent_model_name = 'distilbert-base-nli-stsb-mean-tokens' 
        sent_model = SentenceTransformer(
            sent_model_name, device='cpu', 
            cache_folder=str(SENTENCE_TRANSFORMERS_HOME_PATH) 
        )
        logger.debug(f"Loaded SentenceTransformer model: {sent_model_name}.")
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model {sent_model_name}: {e}. Ensure it's available locally.", exc_info=True)
        raise RuntimeError(f"Failed to load SentenceTransformer model {sent_model_name}.") from e
    model = ensure_model_on_available_device(model)
    sent_model = ensure_model_on_available_device(sent_model) 
    logger.info(f"BERT model '{arg_path.name}' and SentenceTransformer loaded. Devices: Classifier='{next(model.parameters()).device}', SentTransformer='{next(sent_model.parameters()).device}'")
    return model, tokenizer, sent_model, rob_sent


def pred_legacy(doc: str, model: torch.nn.Module, args: dict,
                vocab_stoi: dict, pad_idx: int, unk_idx: int, 
                nlp_instance: Any) -> float: # Added nlp_instance
    if not doc.strip():
        logger.warning("pred_legacy received empty document. Returning 0.0 probability.")
        return 0.0
    if nlp_instance is None:
        logger.error("pred_legacy received None for nlp_instance. Cannot tokenize. Returning 0.0.")
        return 0.0

    model.eval() 
    current_device = next(model.parameters()).device 
    try:
        tokens = [tok.text.lower() for tok in nlp_instance.tokenizer(doc)] # Use passed nlp_instance
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        max_len = args.get('max_token_len', 512) 
        if len(idx) == 0: 
            logger.warning(f"Tokenization resulted in zero tokens for input to pred_legacy. Doc (start): '{doc[:100]}...'")
            idx = [pad_idx] 
        if len(idx) < max_len:
            idx += [pad_idx] * (max_len - len(idx))
        elif len(idx) > max_len:
            idx = idx[:max_len]
        doc_tensor = torch.tensor(idx, dtype=torch.long).to(current_device)
        if model.__class__.__name__ not in ['HAN']: 
             doc_tensor = doc_tensor.unsqueeze(1) 
        with torch.no_grad():
            probs_output = model(doc_tensor)
        if isinstance(probs_output, tuple): 
            probs_output = probs_output[0]
        if probs_output.ndim == 2 and probs_output.shape[0] == 1 and probs_output.shape[1] >= 2:
            prob_positive = probs_output.squeeze(0)[1].item() 
        elif probs_output.ndim == 1 and probs_output.shape[0] >= 2: 
            prob_positive = probs_output[1].item()
        else:
            logger.warning(f"Unexpected output shape from legacy model {type(model).__name__}: {probs_output.shape}. Returning 0.0.")
            prob_positive = 0.0
        return float(prob_positive)
    except Exception as e:
        logger.error(f"Error during pred_legacy for model {type(model).__name__}: {e}", exc_info=True)
        return 0.0 


def pred_bert(text: str, model: torch.nn.Module, tokenizer,
              sent_model: SentenceTransformer, rob_sent_desc: str, 
              nlp_instance: Any, # Added nlp_instance
              max_n_sent: int = 30) -> float:
    if not text.strip():
        logger.warning("pred_bert received empty text. Returning 0.0 probability.")
        return 0.0
    if nlp_instance is None:
        logger.error("pred_bert received None for nlp_instance. Cannot process. Returning 0.0.")
        return 0.0

    model.eval() 
    sent_model.eval() 
    current_device = next(model.parameters()).device 
    sent_model_device = next(sent_model.parameters()).device 
    try:
        sents_spacy = list(nlp_instance(text).sents) # Use passed nlp_instance
        sents = [str(s).strip() for s in sents_spacy if len(str(s).split()) > 3][:max_n_sent * 5] 
        if not sents:
            logger.warning("No sentences found after filtering in pred_bert for text. Returning 0.0.")
            return 0.0
        batch_size_st = 128 
        all_sent_embeds = []
        for i in range(0, len(sents), batch_size_st):
            batch_sents = sents[i:i+batch_size_st]
            batch_embeds = sent_model.encode(batch_sents, convert_to_tensor=True, device=sent_model_device, show_progress_bar=False)
            all_sent_embeds.append(batch_embeds)
        if not all_sent_embeds: 
            logger.warning("Sentence embedding resulted in no embeddings. Returning 0.0.")
            return 0.0
        sent_embeds = torch.cat(all_sent_embeds, dim=0)
        rob_embed = sent_model.encode([rob_sent_desc], convert_to_tensor=True, device=sent_model_device, show_progress_bar=False)
        cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed) 
        top_k = min(max_n_sent, len(sents))
        if top_k == 0: 
            logger.warning("No sentences to select for top_k in pred_bert (top_k is 0). Returning 0.0.")
            return 0.0
        if cos_scores.ndim > 1: 
            cos_scores = cos_scores.squeeze() 
            if cos_scores.ndim == 0: 
                 cos_scores = cos_scores.unsqueeze(0) 
        if cos_scores.numel() == 0: 
            logger.warning("Cosine scores tensor is empty after processing. Returning 0.0.")
            return 0.0
        actual_k_for_topk = min(top_k, cos_scores.numel())
        if actual_k_for_topk == 0:
            logger.warning("actual_k_for_topk is 0 (no scores to pick from). Returning 0.0.")
            return 0.0
        _, top_indices = torch.topk(cos_scores, k=actual_k_for_topk, largest=True, sorted=True)
        top_indices_np = top_indices.cpu().numpy() 
        sim_sents = [sents[i] for i in top_indices_np]
        sim_text = " ".join(sim_sents) 
        if not sim_text.strip():
            logger.warning("Combined similar text (sim_text) is empty before tokenization in pred_bert. Returning 0.0.")
            return 0.0
        inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(current_device) for k, v in inputs.items()} 
        with torch.no_grad():
            outputs = model(**inputs) 
        if outputs.ndim == 2 and outputs.shape[0] == 1 and outputs.shape[1] >= 2:
            prob_positive = outputs.squeeze(0)[1].item()
        else:
            logger.warning(f"Unexpected output shape from BERT model {type(model).__name__}: {outputs.shape}. Returning 0.0.")
            prob_positive = 0.0
        return float(prob_positive)
    except Exception as e:
        logger.error(f"Error during pred_bert for model {type(model).__name__}: {e}", exc_info=True)
        return 0.0 


def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict,
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, 
                     num_sents_to_extract: int, 
                     nlp_instance: Any) -> list[str]: # Added nlp_instance
    if num_sents_to_extract <= 0:
        return []
    if not doc_text.strip():
        logger.warning("extract_sent_han received empty document text.")
        return ["Error: Document text is empty."]
    if nlp_instance is None:
        logger.error("extract_sent_han received None for nlp_instance. Cannot process. Returning error message.")
        return ["Error: NLP instance not available for sentence extraction."]

    model.eval() 
    current_device = next(model.parameters()).device
    try:
        doc_spacy = nlp_instance(doc_text) # Use passed nlp_instance
        sents_text_orig = [sent.text.strip() for sent in doc_spacy.sents]
        sents_tokens_for_model = [] 
        sents_text_for_reconstruction = [] 
        max_sent_len = args.get('max_sent_len', 100) 
        min_sent_len_heuristic = 5 
        for sent_text_item in sents_text_orig:
            tokens = [tok.text.lower() for tok in nlp_instance.tokenizer(sent_text_item)] # Use passed nlp_instance
            if not (min_sent_len_heuristic <= len(tokens) <= max_sent_len + 20): 
                continue 
            sents_text_for_reconstruction.append(sent_text_item) 
            idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
            if len(idx) < max_sent_len:
                idx += [pad_idx] * (max_sent_len - len(idx))
            else:
                idx = idx[:max_sent_len]
            sents_tokens_for_model.append(idx)
        if not sents_tokens_for_model:
            logger.warning("No valid sentences found for HAN attention extraction after filtering by length.")
            return ["Error: No valid sentences for HAN after filtering."]
        max_doc_len = args.get('max_doc_len', 50) 
        actual_sents_indices_for_input = sents_tokens_for_model[:max_doc_len]
        actual_sents_text_for_output = sents_text_for_reconstruction[:max_doc_len] 
        if len(actual_sents_indices_for_input) < max_doc_len:
            pad_sentence_token_indices = [pad_idx] * max_sent_len 
            num_padding_sentences = max_doc_len - len(actual_sents_indices_for_input)
            actual_sents_indices_for_input.extend([pad_sentence_token_indices] * num_padding_sentences)
        doc_tensor_sents = torch.tensor(actual_sents_indices_for_input, dtype=torch.long).to(current_device)
        doc_tensor_batch = doc_tensor_sents.unsqueeze(0) 
        with torch.no_grad():
            outputs = model(doc_tensor_batch) 
        if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], torch.Tensor)):
            model_type_name = type(model).__name__
            err_msg_parts = [f"Error: Output format from HAN model '{model_type_name}' unexpected."]
            # ... (rest of error message construction as before) ...
            output_attn_flag = getattr(model, 'output_attn', 'Not Set/Applicable')
            err_msg_parts.append(f"Model's 'output_attn' flag: {output_attn_flag}. Must be True.")
            full_err_msg = " ".join(err_msg_parts)
            logger.error(full_err_msg)
            return [full_err_msg]
        _probs, sentence_attn_score_tensor = outputs 
        attn_scores_np = sentence_attn_score_tensor.squeeze(0).cpu().numpy() 
        num_actual_sents_fed_to_model = len(actual_sents_text_for_output)
        aligned_scores = attn_scores_np[:num_actual_sents_fed_to_model]
        if len(aligned_scores) != num_actual_sents_fed_to_model: 
            logger.error(f"Attention score length ({len(aligned_scores)}) mismatch with actual sentences ({num_actual_sents_fed_to_model}).")
            return ["Error: Attention score length mismatch after processing."]
        df = pd.DataFrame({'sent_text': actual_sents_text_for_output, 'attn': aligned_scores})
        df = df.sort_values(by=['attn'], ascending=False)
        top_sents_text = list(df['sent_text'][:num_sents_to_extract]) 
        cleaned_top_sents = []
        for sent in top_sents_text:
            sent = re.sub(r" \.", ".", sent); sent = re.sub(r" \,", ",", sent)
            sent = re.sub(r"\( ", "(", sent); sent = re.sub(r" \)", ")", sent)
            sent = re.sub(r"\[ ", "[", sent); sent = re.sub(r" \]", "]", sent)
            sent = re.sub(r" \- ", "-", sent)
            cleaned_top_sents.append(sent.strip())
        return cleaned_top_sents
    except Exception as e:
        logger.error(f"Error during extract_sent_han for model {type(model).__name__}: {e}", exc_info=True)
        return [f"Unhandled Error during HAN sentence extraction: {e}"]

