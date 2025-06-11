#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated to make BERT model loading more robust by requiring the RoB
sentence description as an argument.
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
from typing import Any 

# --- Global constants ---
FILE_PROCESSING_TIMEOUT_SECONDS = 300
ROB_ITEM_CATEGORIES = ['random', 'blind', 'interest', 'welfare', 'exclusion']

# --- Logger for this module ---
logger = logging.getLogger(__name__)

# --- Shim for legacy torchtext (remains unchanged) ---
class _DummyVocab:
    def __init__(self, stoi=None, itos=None, pad_token="<pad>", unk_token="<unk>", specials=None):
        self.stoi = stoi if stoi is not None else {"<unk>": 0, "<pad>": 1}
        if itos is None: self.itos = {i: s for s, i in self.stoi.items()}
        else: self.itos = itos
        self.pad_token, self.unk_token, self.specials = pad_token, unk_token, specials if specials is not None else [pad_token, unk_token]
class _DummyFieldBase:
    def __init__(self, **kwargs):
        self.sequential = kwargs.get('sequential', True); self.use_vocab = kwargs.get('use_vocab', True); self.init_token = kwargs.get('init_token', None); self.eos_token = kwargs.get('eos_token', None); self.unk_token = kwargs.get('unk_token', "<unk>"); self.pad_token = kwargs.get('pad_token', "<pad>"); self.fix_length = kwargs.get('fix_length', None); self.dtype = kwargs.get('dtype', torch.long); self.preprocessing = kwargs.get('preprocessing', None); self.postprocessing = kwargs.get('postprocessing', None); self.lower = kwargs.get('lower', False); self.tokenizer_args = kwargs.get('tokenizer_args', {}); self.tokenize = kwargs.get('tokenize', lambda x: x.split()); self.include_lengths = kwargs.get('include_lengths', False); self.batch_first = kwargs.get('batch_first', False); self.pad_first = kwargs.get('pad_first', False); self.truncate_first = kwargs.get('truncate_first', False); self.stop_words = kwargs.get('stop_words', None); self.is_target = kwargs.get('is_target', False); self.vocab_cls = _DummyVocab; self.vocab = self.vocab_cls(stoi=kwargs.get('vocab_stoi', {"<unk>": 0, "<pad>": 1}), pad_token=self.pad_token, unk_token=self.unk_token)
        for k, v in kwargs.items():
            if not hasattr(self, k): setattr(self, k, v)
class _DummyField(_DummyFieldBase):
    def __init__(self, **kwargs): super().__init__(**kwargs)
class _DummyNestedField(_DummyFieldBase):
    def __init__(self, nesting_field_arg, **kwargs): 
        super().__init__(**kwargs); self.nesting_field = nesting_field_arg
        if hasattr(self.nesting_field, 'vocab') and hasattr(self.nesting_field.vocab, 'stoi'):
            self.vocab = self.nesting_field.vocab; self.pad_token = getattr(self.nesting_field, 'pad_token', self.pad_token); self.unk_token = getattr(self.nesting_field, 'unk_token', self.unk_token)
def _split_tokenizer(x: str) -> list[str]: return x.split()
torchtext_mod, data_mod, field_mod, vocab_mod, utils_mod = types.ModuleType("torchtext"), types.ModuleType("torchtext.data"), types.ModuleType("torchtext.data.field"), types.ModuleType("torchtext.vocab"), types.ModuleType("torchtext.data.utils")
field_mod.Field, field_mod.NestedField, vocab_mod.Vocab, utils_mod._split_tokenizer = _DummyField, _DummyNestedField, _DummyVocab, _split_tokenizer
data_mod.field, data_mod.utils, torchtext_mod.data, torchtext_mod.vocab = field_mod, utils_mod, data_mod, vocab_mod
sys.modules["torchtext"], sys.modules["torchtext.data"], sys.modules["torchtext.data.field"], sys.modules["torchtext.vocab"], sys.modules["torchtext.data.utils"] = torchtext_mod, data_mod, field_mod, vocab_mod, utils_mod
# --- End Shim ---

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
logger.info("TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE set to 1.")

try:
    from transformers import AutoTokenizer, DistilBertModel, AutoConfig, DistilBertTokenizer
    from sentence_transformers import SentenceTransformer, util
    from model import ConvNet, AttnNet, HAN, DistilClsConv 
    logger.info("Successfully imported Transformers and SentenceTransformer modules.")
except ImportError as e:
    logger.critical(f"Failed to import Hugging Face or local model modules: {e}", exc_info=True)
    raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    logger.info(f"Using device: {device} (CUDA). CuDNN deterministic=True, benchmark=False.")
else:
    logger.info(f"Using device: {device} (CPU).")

def ensure_model_on_available_device(model: torch.nn.Module) -> torch.nn.Module:
    # ... (function remains unchanged) ...
    target_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = model.to(target_device_str)
        if target_device_str == 'cuda' and next(model.parameters()).device.type != 'cuda':
             warnings.warn(f"Model {type(model).__name__} parameters not on CUDA after .to('cuda'). Forcing.", RuntimeWarning)
             model.cuda() 
        logger.debug(f"Model {type(model).__name__} moved to {target_device_str}.")
    except Exception as e:
        warnings.warn(f"Failed to move model {type(model).__name__} to {target_device_str} (reason: {e}). Falling back to CPU.", RuntimeWarning)
        logger.warning(f"Falling back to CPU for model {type(model).__name__} due to error: {e}", exc_info=True)
        model = model.to('cpu')
    model.eval() 
    return model

HF_HOME_DEFAULT = Path.home() / ".cache" / "huggingface"
HF_HOME_PATH = Path(os.environ.get('HF_HOME', HF_HOME_DEFAULT))
TRANSFORMERS_CACHE_PATH = Path(os.environ.get('TRANSFORMERS_CACHE', HF_HOME_PATH / "hub"))
SENTENCE_TRANSFORMERS_HOME_PATH = Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', HF_HOME_PATH / "sentence_transformers"))
logger.info(f"Hugging Face cache paths: TRANSFORMERS_CACHE_PATH={TRANSFORMERS_CACHE_PATH}, SENTENCE_TRANSFORMERS_HOME_PATH={SENTENCE_TRANSFORMERS_HOME_PATH}")

def load_vocab_info(fld_path: Path) -> tuple[dict, int, int]:
    # ... (function remains unchanged) ...
    if not fld_path.exists(): raise FileNotFoundError(f"Vocab field file not found at {fld_path}")
    try:
        with open(fld_path, "rb") as fin: loaded_obj = dill.load(fin)
        vocab_source_obj = None
        if hasattr(loaded_obj, 'nesting_field') and hasattr(loaded_obj.nesting_field, 'vocab'): vocab_source_obj = loaded_obj.nesting_field
        elif hasattr(loaded_obj, 'vocab'): vocab_source_obj = loaded_obj
        else: raise ValueError(f"Loaded object from {fld_path.name} lacks 'vocab' structure.")
        if not hasattr(vocab_source_obj, 'vocab') or not hasattr(vocab_source_obj.vocab, 'stoi'): raise ValueError(f"Vocab source object lacks 'vocab.stoi'.")
        vocab_stoi = vocab_source_obj.vocab.stoi
        pad_token, unk_token = getattr(vocab_source_obj, 'pad_token', "<pad>"), getattr(vocab_source_obj, 'unk_token', "<unk>")
        pad_idx, unk_idx = vocab_stoi.get(pad_token), vocab_stoi.get(unk_token)
        if pad_idx is None: pad_idx = vocab_stoi.get("<pad>", 1)
        if unk_idx is None: unk_idx = vocab_stoi.get("<unk>", 0)
        return vocab_stoi, pad_idx, unk_idx
    except Exception as e:
        logger.error(f"Error loading or processing vocab file {fld_path}: {e}", exc_info=True)
        raise

def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    # ... (function remains unchanged, it was loading correctly) ...
    with open(arg_path) as f: args_json = json.load(f).get('args')
    vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
    vocab_size = len(vocab_stoi)
    actual_vocab_size_for_model = vocab_size
    if 'max_vocab_size' in args_json and args_json['max_vocab_size'] > 0 and (args_json['max_vocab_size'] + 2) < vocab_size:
        actual_vocab_size_for_model = args_json['max_vocab_size'] + 2
    net_type = args_json.get('net_type', 'unknown').lower()
    model: torch.nn.Module
    default_output_attn = True if net_type == 'han' else args_json.get('output_attn', False)
    current_output_attn_setting = args_json.get('output_attn', default_output_attn)
    if net_type == 'han' and any(arg_path.name.lower().startswith(p) for p in ['hr_', 'hb_', 'hi_', 'hw_', 'he_']):
        if not current_output_attn_setting:
             logger.info(f"OVERRIDE_HAN_LOAD: Forcing 'output_attn' to True for sentence model '{arg_path.name}'.")
        current_output_attn_setting = True 
    if net_type == 'cnn':
        sizes = [int(s) for s in args_json.get('filter_sizes', "3,4,5").split(',')]
        model = ConvNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], n_filters=args_json['num_filters'], filter_sizes=sizes, output_dim=2, dropout=args_json['dropout'], pad_idx=pad_idx, embed_trainable=args_json.get('embed_trainable', False), batch_norm=args_json.get('batch_norm', False))
    elif net_type == 'attn':
        model = AttnNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], rnn_hidden_dim=args_json['rnn_hidden_dim'], rnn_num_layers=args_json['rnn_num_layers'], output_dim=2, bidirection=args_json.get('bidirection', True), rnn_cell_type=args_json.get('rnn_cell_type', 'gru'), dropout=args_json['dropout'], pad_idx=pad_idx, embed_trainable=args_json.get('embed_trainable', False), batch_norm=args_json.get('batch_norm', False), output_attn=current_output_attn_setting)
    elif net_type == 'han':
        model = HAN(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], word_hidden_dim=args_json['word_hidden_dim'], word_num_layers=args_json['word_num_layers'], pad_idx=pad_idx, embed_trainable=args_json.get('embed_trainable', False), batch_norm=args_json.get('batch_norm', False), sent_hidden_dim=args_json['sent_hidden_dim'], sent_num_layers=args_json['sent_num_layers'], output_dim=2, output_attn=current_output_attn_setting, dropout_rate=args_json.get('dropout', 0.5))
    else: raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path.name}")
    checkpoint = torch.load(pth_path, map_location="cpu") 
    state_dict = checkpoint.get('state_dict', checkpoint) 
    model.load_state_dict(state_dict, strict=True)
    model = ensure_model_on_available_device(model)
    return model, args_json, vocab_stoi, pad_idx, unk_idx

def load_model_bert(arg_path: Path, pth_path: Path, rob_sent_description: str):
    """
    Loads a BERT-based model. Now requires the rob_sent_description to be provided.
    """
    logger.info(f"Loading BERT model: args='{arg_path.name}', weights='{pth_path.name}'")
    if not rob_sent_description or not isinstance(rob_sent_description, str):
        raise ValueError("load_model_bert requires a valid rob_sent_description string.")

    distilbert_model_name_or_path = 'distilbert-base-uncased' 
    try:
        config = AutoConfig.from_pretrained(
            distilbert_model_name_or_path, num_labels=2, return_dict=True,
            cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
        )
        model = DistilClsConv(config)
        distilbert_base_model = DistilBertModel.from_pretrained(
            distilbert_model_name_or_path, config=config,
            cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
        )
        model.distilbert.load_state_dict(distilbert_base_model.state_dict())
        del distilbert_base_model 
        checkpoint = torch.load(pth_path, map_location="cpu") 
    except Exception as e:
        logger.error(f"Error during DistilClsConv initialization or loading base weights: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize DistilClsConv or load base weights.") from e
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys: logger.warning(f"Missing keys loading fine-tuned state_dict for {pth_path.name}: {missing_keys}")
    if unexpected_keys: logger.warning(f"Unexpected keys loading fine-tuned state_dict for {pth_path.name}: {unexpected_keys}")

    tokenizer = DistilBertTokenizer.from_pretrained(
        distilbert_model_name_or_path, use_fast=True,
        cache_dir=str(TRANSFORMERS_CACHE_PATH), local_files_only=True
    )
    try:
        sent_model_name = 'distilbert-base-nli-stsb-mean-tokens' 
        sent_model = SentenceTransformer(
            sent_model_name, device='cpu', 
            cache_folder=str(SENTENCE_TRANSFORMERS_HOME_PATH) 
        )
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model {sent_model_name}: {e}. Ensure it's available locally.", exc_info=True)
        raise RuntimeError(f"Failed to load SentenceTransformer model {sent_model_name}.") from e
    
    model = ensure_model_on_available_device(model)
    sent_model = ensure_model_on_available_device(sent_model) 
    
    # Return the description that was passed in.
    return model, tokenizer, sent_model, rob_sent_description

def pred_legacy(doc: str, model: torch.nn.Module, args: dict,
                vocab_stoi: dict, pad_idx: int, unk_idx: int, 
                nlp_instance: Any) -> float: 
    # ... (function remains unchanged) ...
    if not doc.strip(): return 0.0
    if nlp_instance is None: return 0.0
    model.eval(); current_device = next(model.parameters()).device 
    try:
        tokens = [tok.text.lower() for tok in nlp_instance.tokenizer(doc)] 
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        max_len = args.get('max_token_len', 512) 
        if len(idx) == 0: idx = [pad_idx] 
        if len(idx) < max_len: idx += [pad_idx] * (max_len - len(idx))
        elif len(idx) > max_len: idx = idx[:max_len]
        doc_tensor = torch.tensor(idx, dtype=torch.long).to(current_device)
        if model.__class__.__name__ not in ['HAN']: doc_tensor = doc_tensor.unsqueeze(1) 
        with torch.no_grad(): probs_output = model(doc_tensor)
        if isinstance(probs_output, tuple): probs_output = probs_output[0]
        if probs_output.ndim == 2: prob_positive = probs_output.squeeze(0)[1].item() 
        elif probs_output.ndim == 1: prob_positive = probs_output[1].item()
        else: prob_positive = 0.0
        return float(prob_positive)
    except Exception as e:
        logger.error(f"Error during pred_legacy for model {type(model).__name__}: {e}", exc_info=True)
        return 0.0 

def pred_bert(text: str, model: torch.nn.Module, tokenizer,
              sent_model: SentenceTransformer, rob_sent_desc: str, 
              nlp_instance: Any, 
              max_n_sent: int = 30) -> float:
    # Add guard clause for rob_sent_desc
    if not rob_sent_desc or not isinstance(rob_sent_desc, str):
        logger.warning(f"pred_bert received an invalid or empty rob_sent_desc. Value: '{rob_sent_desc}'. Returning 0.0.")
        return 0.0
    if not text.strip(): return 0.0
    if nlp_instance is None: return 0.0

    model.eval(); sent_model.eval() 
    current_device = next(model.parameters()).device 
    sent_model_device = next(sent_model.parameters()).device 
    try:
        sents = [str(s).strip() for s in nlp_instance(text).sents if len(str(s).split()) > 3][:max_n_sent * 5] 
        if not sents: return 0.0
        
        all_sent_embeds = []
        for i in range(0, len(sents), 128):
            batch_embeds = sent_model.encode(sents[i:i+128], convert_to_tensor=True, device=sent_model_device, show_progress_bar=False)
            all_sent_embeds.append(batch_embeds)
        
        if not all_sent_embeds: return 0.0
        sent_embeds = torch.cat(all_sent_embeds, dim=0)
        rob_embed = sent_model.encode([rob_sent_desc], convert_to_tensor=True, device=sent_model_device, show_progress_bar=False)
        
        cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed) 
        top_k = min(max_n_sent, len(sents))
        if top_k == 0: return 0.0
        
        if cos_scores.ndim > 1: cos_scores = cos_scores.squeeze() 
        if cos_scores.ndim == 0: cos_scores = cos_scores.unsqueeze(0) 
        if cos_scores.numel() == 0: return 0.0
        
        actual_k_for_topk = min(top_k, cos_scores.numel())
        if actual_k_for_topk == 0: return 0.0
        
        _, top_indices = torch.topk(cos_scores, k=actual_k_for_topk, largest=True, sorted=True)
        sim_text = " ".join([sents[i] for i in top_indices.cpu().numpy()]) 
        
        if not sim_text.strip(): return 0.0
        
        inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(current_device) for k, v in inputs.items()} 
        with torch.no_grad(): outputs = model(**inputs) 
        
        if outputs.ndim == 2: prob_positive = outputs.squeeze(0)[1].item()
        else: prob_positive = 0.0
        
        return float(prob_positive)
    except Exception as e:
        logger.error(f"Error during pred_bert for model {type(model).__name__}: {e}", exc_info=True)
        return 0.0 

def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict,
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, 
                     num_sents_to_extract: int, 
                     nlp_instance: Any) -> list[str]: 
    # ... (function remains unchanged) ...
    if num_sents_to_extract <= 0: return []
    if not doc_text.strip(): return ["Error: Document text is empty."]
    if nlp_instance is None: return ["Error: NLP instance not available."]
    model.eval(); current_device = next(model.parameters()).device
    try:
        sents_text_orig = [sent.text.strip() for sent in nlp_instance(doc_text).sents]
        sents_tokens_for_model, sents_text_for_reconstruction = [], []
        max_sent_len = args.get('max_sent_len', 100)
        for sent_text_item in sents_text_orig:
            tokens = [tok.text.lower() for tok in nlp_instance.tokenizer(sent_text_item)]
            if not (5 <= len(tokens) <= max_sent_len + 20): continue 
            sents_text_for_reconstruction.append(sent_text_item) 
            idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
            if len(idx) < max_sent_len: idx += [pad_idx] * (max_sent_len - len(idx))
            else: idx = idx[:max_sent_len]
            sents_tokens_for_model.append(idx)
        if not sents_tokens_for_model: return ["Error: No valid sentences for HAN."]
        max_doc_len = args.get('max_doc_len', 50) 
        actual_sents_indices_for_input = sents_tokens_for_model[:max_doc_len]
        actual_sents_text_for_output = sents_text_for_reconstruction[:max_doc_len] 
        if len(actual_sents_indices_for_input) < max_doc_len:
            pad_sentence = [pad_idx] * max_sent_len
            actual_sents_indices_for_input.extend([pad_sentence] * (max_doc_len - len(actual_sents_indices_for_input)))
        doc_tensor = torch.tensor(actual_sents_indices_for_input, dtype=torch.long).to(current_device).unsqueeze(0) 
        with torch.no_grad(): outputs = model(doc_tensor) 
        if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], torch.Tensor)):
            err_msg = f"Error: Output from model '{type(model).__name__}' is not (probs, attn_tensor)."
            logger.error(err_msg)
            return [err_msg]
        _probs, attn_score_tensor = outputs 
        attn_scores_np = attn_score_tensor.squeeze(0).cpu().numpy()[:len(actual_sents_text_for_output)]
        df = pd.DataFrame({'sent_text': actual_sents_text_for_output, 'attn': attn_scores_np})
        df = df.sort_values(by=['attn'], ascending=False)
        top_sents_text = list(df['sent_text'][:num_sents_to_extract]) 
        cleaned_top_sents = [re.sub(r" \.", ".", s).strip() for s in top_sents_text] # simplified cleaning
        return cleaned_top_sents
    except Exception as e:
        logger.error(f"Error during extract_sent_han for model {type(model).__name__}: {e}", exc_info=True)
        return [f"Unhandled Error in extract_sent_han: {e}"]

