#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated for Python 3.10, PyTorch 2.x, Transformers 4.x, SentenceTransformers 2.x
Handles legacy torchtext Field and NestedField objects for model loading.
Forces output_attn=True for HAN sentence models.
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
        self.tokenize = kwargs.get('tokenize', lambda x: x.split()) # Simplified default
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
        self.fixup_attrs = True

def _split_tokenizer(x):
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

try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError as e:
    print(f"CRITICAL ERROR: spaCy model 'en_core_web_sm' could not be loaded.")
    print(f"Spacy version: {spacy.__version__}. Error: {e}")
    raise RuntimeError("spaCy model 'en_core_web_sm' not found or incompatible.") from e

from transformers import AutoTokenizer, DistilBertModel, AutoConfig, DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util
from model import ConvNet, AttnNet, HAN, DistilClsConv # Assuming model.py is in the same directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {device}")

def ensure_model_on_available_device(model: torch.nn.Module) -> torch.nn.Module:
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = model.to(target_device)
        if target_device == 'cuda':
            _ = next(model.parameters()).to(target_device)
    except Exception as e:
        warnings.warn(
            f"Failed to move model {type(model).__name__} to {target_device} (reason: {e}). Falling back to CPU.",
            RuntimeWarning,
        )
        model = model.to('cpu')
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


def load_vocab_info(fld_path: Path) -> tuple[dict, int, int]:
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
            raise ValueError(f"Loaded object from {fld_path.name} lacks 'vocab' or 'nesting_field.vocab'. Type: {type(loaded_obj)}")
        if not hasattr(vocab_source_obj, 'vocab') or not hasattr(vocab_source_obj.vocab, 'stoi'):
            raise ValueError(f"Vocab source object (from {source_type_msg}) lacks 'vocab.stoi'. Type: {type(vocab_source_obj.vocab)}")
        vocab_stoi = vocab_source_obj.vocab.stoi
        pad_token = getattr(vocab_source_obj, 'pad_token', "<pad>")
        unk_token = getattr(vocab_source_obj, 'unk_token', "<unk>")
        pad_idx = vocab_stoi.get(pad_token, 1)
        unk_idx = vocab_stoi.get(unk_token, 0)
        return vocab_stoi, pad_idx, unk_idx
    except FileNotFoundError:
        print(f"Error: Vocab field file not found at {fld_path}")
        raise
    except Exception as e:
        print(f"Error loading or processing vocab file {fld_path}: {e}")
        raise

def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    with open(arg_path) as f:
        args_json = json.load(f)['args'] # Renamed to avoid conflict with function args

    vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
    vocab_size = len(vocab_stoi)
    if 'max_vocab_size' in args_json and (args_json['max_vocab_size'] + 2) > vocab_size:
         actual_vocab_size_for_model = args_json['max_vocab_size'] + 2
    else:
         actual_vocab_size_for_model = vocab_size

    net_type = args_json.get('net_type', 'unknown')
    model: torch.nn.Module

    # Determine the output_attn setting
    # Default to False for AttnNet, True for HAN (unless overridden by JSON)
    default_output_attn = True if net_type == 'han' else False
    current_output_attn_setting = args_json.get('output_attn', default_output_attn)

    # --- Force output_attn to True for sentence extraction HAN models ---
    fn_lower = arg_path.name.lower()
    is_sentence_han_model = net_type == 'han' and fn_lower.startswith(('hr_', 'hb_', 'hi_', 'hw_', 'he_'))
    
    if is_sentence_han_model:
        if not current_output_attn_setting: # If JSON had it as False or it defaulted to False (which it wouldn't for HAN by default)
            print(f"OVERRIDE_HAN_LOAD: For sentence model config '{arg_path.name}', JSON 'output_attn' was {args_json.get('output_attn')}. Forcing to True for sentence extraction.")
        current_output_attn_setting = True # Force it
    elif net_type == 'han': # For other HAN models, print the debug info
         print(f"DEBUG_HAN_LOAD: For HAN (non-sentence) model config '{arg_path.name}', 'output_attn' resolved to: {current_output_attn_setting} (from JSON args: {args_json.get('output_attn')}, default if key missing: True)")


    if net_type == 'cnn':
        sizes = [int(s) for s in args_json['filter_sizes'].split(',')]
        model = ConvNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], n_filters=args_json['num_filters'],
                        filter_sizes=sizes, output_dim=2, dropout=args_json['dropout'], pad_idx=pad_idx,
                        embed_trainable=args_json['embed_trainable'], batch_norm=args_json['batch_norm'])
    elif net_type == 'attn':
        model = AttnNet(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], rnn_hidden_dim=args_json['rnn_hidden_dim'],
                        rnn_num_layers=args_json['rnn_num_layers'], output_dim=2, bidirection=args_json['bidirection'],
                        rnn_cell_type=args_json['rnn_cell_type'], dropout=args_json['dropout'], pad_idx=pad_idx,
                        embed_trainable=args_json['embed_trainable'], batch_norm=args_json['batch_norm'],
                        output_attn=current_output_attn_setting) # Use resolved setting
    elif net_type == 'han':
        model = HAN(vocab_size=actual_vocab_size_for_model, embedding_dim=args_json['embed_dim'], word_hidden_dim=args_json['word_hidden_dim'],
                    word_num_layers=args_json['word_num_layers'], pad_idx=pad_idx, embed_trainable=args_json['embed_trainable'],
                    batch_norm=args_json['batch_norm'], sent_hidden_dim=args_json['sent_hidden_dim'],
                    sent_num_layers=args_json['sent_num_layers'], output_dim=2,
                    output_attn=current_output_attn_setting) # Use resolved (and possibly overridden) setting
    else:
        raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path}")

    try:
        checkpoint = torch.load(pth_path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {pth_path}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint {pth_path}: {e}. Trying with weights_only=True as fallback.")
        try:
            checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        except Exception as e_fallback:
            print(f"Fallback with weights_only=True also failed for {pth_path}: {e_fallback}")
            raise e_fallback

    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict):
         warnings.warn(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key. Assuming it IS the state_dict.", UserWarning)
         state_dict = checkpoint
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
         warnings.warn(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key. Assuming the dict itself is the state_dict.", UserWarning)
         state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading state_dict into {net_type} model from {pth_path.name} (strict=True): {e}. Trying strict=False.")
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e_false:
            print(f"Error loading state_dict into {net_type} model from {pth_path.name} (strict=False): {e_false}")
            raise e_false

    model = ensure_model_on_available_device(model)
    return model, args_json, vocab_stoi, pad_idx, unk_idx # Return args_json

def load_model_bert(arg_path: Path, pth_path: Path):
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
        print(f"Error during DistilClsConv init or loading base weights: {e}")
        raise
    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint and not isinstance(checkpoint, dict) :
        warnings.warn(f"Checkpoint {pth_path.name} is not a dict and has no 'state_dict' key. Assuming it IS the state_dict.", UserWarning)
        state_dict = checkpoint
    elif 'state_dict' not in checkpoint and isinstance(checkpoint, dict):
        warnings.warn(f"Checkpoint {pth_path.name} is a dict but has no 'state_dict' key. Assuming the dict itself is the state_dict.", UserWarning)
        state_dict = checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys loading fine-tuned state_dict for {pth_path.name}: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys loading fine-tuned state_dict for {pth_path.name}: {unexpected_keys}")
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
        print(f"Error loading SentenceTransformer model {sent_model_name}: {e}")
        raise
    model = ensure_model_on_available_device(model)
    sent_model = ensure_model_on_available_device(sent_model)
    return model, tokenizer, sent_model, rob_sent

def pred_legacy(doc: str, model: torch.nn.Module, args: dict,
                vocab_stoi: dict, pad_idx: int, unk_idx: int) -> float:
    model.eval()
    current_device = next(model.parameters()).device
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]
    idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
    max_len = args.get('max_token_len', 512)
    if len(idx) < max_len:
        idx += [pad_idx] * (max_len - len(idx))
    elif len(idx) > max_len:
        idx = idx[:max_len]
    doc_tensor = torch.tensor(idx, dtype=torch.long).to(current_device)
    doc_tensor = doc_tensor.unsqueeze(1)
    with torch.no_grad():
        probs_output = model(doc_tensor)
    if probs_output.ndim == 2 and probs_output.shape[0] == 1 and probs_output.shape[1] >= 2:
        prob_positive = probs_output.squeeze(0)[1].item()
    elif probs_output.ndim == 1 and probs_output.shape[0] >=2: # If output was already squeezed (e.g. from AttnNet with output_attn=False)
        prob_positive = probs_output[1].item()
    else: # Fallback for unexpected shapes
        warnings.warn(f"Unexpected output shape from legacy model: {probs_output.shape}. Cannot extract positive class probability reliably.", UserWarning)
        prob_positive = 0.0
    return float(prob_positive)

def pred_bert(text: str, model: torch.nn.Module, tokenizer,
              sent_model: SentenceTransformer, rob_sent: str, max_n_sent: int = 30) -> float:
    model.eval()
    current_device = next(model.parameters()).device
    sent_model.to(current_device)
    sents_spacy = list(nlp(text).sents)
    sents = [str(s).strip() for s in sents_spacy if len(str(s).split()) > 3][:max_n_sent * 5]
    if not sents:
        warnings.warn("No sentences found after filtering in pred_bert. Returning 0.0.", UserWarning)
        return 0.0
    try:
        batch_size_st = 128
        all_sent_embeds = []
        for i in range(0, len(sents), batch_size_st):
            batch_sents = sents[i:i+batch_size_st]
            batch_embeds = sent_model.encode(batch_sents, convert_to_tensor=True, device=current_device, show_progress_bar=False)
            all_sent_embeds.append(batch_embeds)
        sent_embeds = torch.cat(all_sent_embeds, dim=0)
        rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True, device=current_device, show_progress_bar=False)
    except Exception as e:
        print(f"Error during sentence encoding in pred_bert: {e}")
        return 0.0
    cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed)
    top_k = min(max_n_sent, len(sents))
    if top_k == 0:
        warnings.warn("No sentences to select for top_k in pred_bert (top_k is 0). Returning 0.0.", UserWarning)
        return 0.0
    if cos_scores.ndim > 1:
        cos_scores = cos_scores.squeeze()
        if cos_scores.ndim == 0:
             cos_scores = cos_scores.unsqueeze(0)
    if cos_scores.numel() < top_k :
        warnings.warn(f"Fewer sentences available ({cos_scores.numel()}) than top_k ({top_k}) in pred_bert. Using all available.", UserWarning)
        top_k = cos_scores.numel()
        if top_k == 0: return 0.0
    top_vals, top_indices = torch.topk(cos_scores, k=top_k, largest=True, sorted=True)
    top_indices_np = top_indices.cpu().numpy()
    sim_sents = [sents[i] for i in top_indices_np]
    sim_text = " ".join(sim_sents)
    if not sim_text.strip():
        warnings.warn("Sim_text is empty before tokenization in pred_bert. Returning 0.0.", UserWarning)
        return 0.0
    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(current_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.ndim == 2 and outputs.shape[0] == 1 and outputs.shape[1] >= 2:
        prob_positive = outputs.squeeze(0)[1].item()
    else:
        warnings.warn(f"Unexpected output shape from BERT model: {outputs.shape}. Cannot extract positive class probability reliably.", UserWarning)
        prob_positive = 0.0
    return float(prob_positive)

def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict,
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, num_sents: int) -> list[str]:
    if num_sents <= 0: return []
    model.eval()
    current_device = next(model.parameters()).device
    doc_spacy = nlp(doc_text)
    sents_text_orig = [sent.text.strip() for sent in doc_spacy.sents]
    sents_tokens_for_model = []
    sents_text_for_reconstruction = []
    max_sent_len = args.get('max_sent_len', 100)

    for sent_text_item in sents_text_orig:
        tokens = [tok.text.lower() for tok in nlp.tokenizer(sent_text_item)]
        if len(tokens) < 5 or len(tokens) > max_sent_len + 20 : continue
        sents_text_for_reconstruction.append(sent_text_item)
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        if len(idx) < max_sent_len:
            idx += [pad_idx] * (max_sent_len - len(idx))
        else:
            idx = idx[:max_sent_len]
        sents_tokens_for_model.append(idx)

    if not sents_tokens_for_model:
        warnings.warn("No valid sentences found for HAN attention extraction after filtering.", UserWarning)
        return ["Error: No valid sentences for HAN after filtering."]

    max_doc_len = args.get('max_doc_len', 50)
    actual_sents_indices_for_input = sents_tokens_for_model[:max_doc_len]
    actual_sents_text_for_output = sents_text_for_reconstruction[:max_doc_len]

    if len(actual_sents_indices_for_input) < max_doc_len:
        pad_sentence_tensor_list = [pad_idx] * max_sent_len
        actual_sents_indices_for_input.extend(
            [pad_sentence_tensor_list] * (max_doc_len - len(actual_sents_indices_for_input))
        )

    doc_tensor = torch.tensor(actual_sents_indices_for_input, dtype=torch.long).to(current_device)
    doc_tensor = doc_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(doc_tensor)

    if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], torch.Tensor)):
        model_type_name = type(model).__name__
        debug_msg_parts = [f"Error: Output format from model '{model_type_name}' unexpected."]
        debug_msg_parts.append(f"Expected tuple (probs, attn_tensor). Got type: {type(outputs)}.")
        if isinstance(outputs, tuple):
            debug_msg_parts.append(f"Tuple length: {len(outputs)} (expected 2).")
            if len(outputs) > 0 and not isinstance(outputs[0], torch.Tensor):
                 debug_msg_parts.append(f"Type of outputs[0]: {type(outputs[0])} (expected Tensor).")
            if len(outputs) > 1 and not isinstance(outputs[1], torch.Tensor):
                 debug_msg_parts.append(f"Type of outputs[1]: {type(outputs[1])} (expected Tensor).")
        elif isinstance(outputs, torch.Tensor):
            debug_msg_parts.append(f"Output is a Tensor with shape: {outputs.shape}.")
        else:
            debug_msg_parts.append(f"Output value (first 100 chars): {str(outputs)[:100]}.")

        if hasattr(model, 'output_attn'):
            debug_msg_parts.append(f"Model's internal 'output_attn' flag: {model.output_attn}.")
        
        full_debug_msg = " ".join(debug_msg_parts)
        warnings.warn(full_debug_msg, RuntimeWarning)
        return [full_debug_msg]

    _probs, attn_score_tensor = outputs
    attn_scores_np = attn_score_tensor.squeeze().cpu().numpy()
    num_sents_fed_to_model = len(actual_sents_text_for_output)
    aligned_scores = attn_scores_np[:num_sents_fed_to_model]

    if len(aligned_scores) != num_sents_fed_to_model:
        warnings.warn(f"Attention score length ({len(aligned_scores)}) mismatch with "
                      f"number of sentences fed to model ({num_sents_fed_to_model}). Cannot reliably extract.", RuntimeWarning)
        return ["Error: Attention score mismatch."]

    df = pd.DataFrame({'sent_text': actual_sents_text_for_output, 'attn': aligned_scores})
    df = df.sort_values(by=['attn'], ascending=False)
    top_sents_text = list(df['sent_text'][:num_sents])
    cleaned_top_sents = []
    for sent in top_sents_text:
        sent = re.sub(r" \.", ".", sent); sent = re.sub(r" \,", ",", sent)
        sent = re.sub(r"\( ", "(", sent); sent = re.sub(r" \)", ")", sent)
        sent = re.sub(r"\[ ", "[", sent); sent = re.sub(r" \]", "]", sent)
        sent = re.sub(r" \- ", "-", sent)
        cleaned_top_sents.append(sent.strip())
        
    return cleaned_top_sents
