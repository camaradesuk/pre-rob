#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function definitions for RoB prediction.
Updated for Python 3.9, PyTorch 2.6, Transformers 4.x, SentenceTransformers 2.x
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

# ---- add this tiny helper so legacy pickles can resolve it
import types, sys

try:
    from torchtext.legacy.data import Field as _LegacyField   # real import if it works
except ModuleNotFoundError:
    class _LegacyField:                                       # dummy replacement
        def __init__(self, *_, **__): pass
        vocab = types.SimpleNamespace(stoi={})
        pad_token = "<pad>"
        unk_token = "<unk>"

def _split_tokenizer(x):
    return x.split()

# build fake module hierarchy: torchtext → torchtext.data → torchtext.vocab
# and add a dummy _split_tokenizer in torchtext.data.utils

torchtext_mod                       = types.ModuleType("torchtext")
data_mod                            = types.ModuleType("torchtext.data")
field_mod                           = types.ModuleType("torchtext.data.field")
vocab_mod                           = types.ModuleType("torchtext.vocab")
utils_mod                           = types.ModuleType("torchtext.data.utils")

class _StubVocab:                       # just enough for .stoi / .itos
    def __init__(self):
        self.stoi, self.itos = {}, []

vocab_mod.Vocab                   = _StubVocab
field_mod.Field                   = _LegacyField
data_mod.field                    = field_mod
torchtext_mod.data                = data_mod
torchtext_mod.vocab               = vocab_mod
utils_mod._split_tokenizer        = _split_tokenizer
data_mod.utils                    = utils_mod
torchtext_mod.data.utils          = utils_mod

sys.modules.update({
    "torchtext"              : torchtext_mod,
    "torchtext.data"         : data_mod,
    "torchtext.data.field"   : field_mod,
    "torchtext.vocab"        : vocab_mod,
    "torchtext.data.utils"   : utils_mod,
})
# ---------------------------------------------------------------
# ---------------------------------------------------------------

# --- Hugging Face Offline Configuration ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1" # Good practice, even if datasets lib isn't directly used for downloading

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError as e:
    print(f"CRITICAL ERROR: spaCy model 'en_core_web_sm' could not be loaded.")
    print(f"Attempted to load 'en_core_web_sm' using spacy.load().")
    print(f"Spacy version: {spacy.__version__}")
    print(f"Error details: {e}")
    print("This model should have been pip-installed directly into the Conda environment's site-packages "
          "during the image build (e.g., via setup.sh using a direct wheel URL).")
    print("If this error occurs, it means the model package is missing or incompatible in the environment.")
    print("Please check the Dockerfile (specifically the setup.sh execution and Conda environment consistency).")
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found or incompatible in the environment. "
        "The image build process may have failed to install it correctly."
    ) from e

from transformers import AutoTokenizer, DistilBertModel, AutoConfig, DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util
from model import ConvNet, AttnNet, HAN, DistilClsConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

print(f"Using device: {device}")

def ensure_model_on_available_device(model):
    """
    Move model to CUDA if available, otherwise CPU. Fallback to CPU if moving to CUDA fails.
    """
    try:
        if torch.cuda.is_available():
            try:
                model = model.to('cuda')
                # Test with a dummy tensor to see if this CUDA op is supported on this GPU
                _ = next(model.parameters()).to('cuda')
            except Exception as cuda_e:
                warnings.warn(
                    f"CUDA op unsupported on this GPU (reason: {cuda_e}). Falling back to CPU for this model.",
                    RuntimeWarning,
                )
                model = model.to('cpu')
        else:
            model = model.to('cpu')
    except Exception as e:
        warnings.warn(f"Device transfer failed: {e}. Forcing CPU.", RuntimeWarning)
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

print(f"Expecting Hugging Face models (transformers lib) in TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE_PATH}")
print(f"Expecting SentenceTransformer models in SENTENCE_TRANSFORMERS_HOME: {SENTENCE_TRANSFORMERS_HOME_PATH}")

def load_vocab_info(fld_path: Path):
    try:
        with open(fld_path, "rb") as fin:
            import sys, types
            try:
                from torchtext.legacy.data import Field as _LegacyField
            except ModuleNotFoundError:
                class _LegacyField:
                    def __init__(self, *_, **__): pass
                    vocab = types.SimpleNamespace(stoi={})
                    pad_token = "<pad>"
                    unk_token = "<unk>"
            torchtext_mod          = types.ModuleType("torchtext")
            data_mod               = types.ModuleType("torchtext.data")
            field_mod              = types.ModuleType("torchtext.data.field")
            field_mod.Field        = _LegacyField
            data_mod.field         = field_mod
            torchtext_mod.data     = data_mod
            sys.modules["torchtext"]               = torchtext_mod
            sys.modules["torchtext.data"]          = data_mod
            sys.modules["torchtext.data.field"]    = field_mod

            loaded_obj = dill.load(fin)

        if hasattr(loaded_obj, 'vocab') and hasattr(loaded_obj.vocab, 'stoi') and \
           hasattr(loaded_obj, 'pad_token') and hasattr(loaded_obj, 'unk_token'):
            warnings.warn(f"Loaded legacy torchtext.Field object from {Path(fld_path).name}. ", DeprecationWarning)
            vocab_stoi = loaded_obj.vocab.stoi
            pad_token = loaded_obj.pad_token
            unk_token = loaded_obj.unk_token
            pad_idx = vocab_stoi.get(pad_token, 1)
            unk_idx = vocab_stoi.get(unk_token, 0)
            print(f"Loaded vocab info from legacy Field: {Path(fld_path).name}")
            return vocab_stoi, pad_idx, unk_idx
        else:
            raise ValueError(f"Cannot extract vocab info from {fld_path}. "
                             "Loaded object does not have expected legacy Field object attributes (vocab.stoi, pad_token, unk_token).")
    except Exception as e:
        print(f"Error loading or processing vocab file {fld_path}: {e}")
        raise

def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    with open(arg_path) as f:
        args = json.load(f)['args']
    try:
        vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
        vocab_size = len(vocab_stoi)
        if 'max_vocab_size' in args and (args['max_vocab_size'] + 2) > vocab_size:
            vocab_size = args['max_vocab_size'] + 2
    except Exception as e:
        print(f"Failed to load vocabulary from {fld_path} for legacy model. Error: {e}")
        raise

    net_type = args.get('net_type', 'unknown')
    model: torch.nn.Module
    if net_type == 'cnn':
        sizes = [int(s) for s in args['filter_sizes'].split(',')]
        model = ConvNet(vocab_size=vocab_size, embedding_dim=args['embed_dim'], n_filters=args['num_filters'],
                        filter_sizes=sizes, output_dim=2, dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'])
    elif net_type == 'attn':
        model = AttnNet(vocab_size=vocab_size, embedding_dim=args['embed_dim'], rnn_hidden_dim=args['rnn_hidden_dim'],
                        rnn_num_layers=args['rnn_num_layers'], output_dim=2, bidirection=args['bidirection'],
                        rnn_cell_type=args['rnn_cell_type'], dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'],
                        output_attn=args.get('output_attn', False))
    elif net_type == 'han':
        model = HAN(vocab_size=vocab_size, embedding_dim=args['embed_dim'], word_hidden_dim=args['word_hidden_dim'],
                    word_num_layers=args['word_num_layers'], pad_idx=pad_idx, embed_trainable=args['embed_trainable'],
                    batch_norm=args['batch_norm'], sent_hidden_dim=args['sent_hidden_dim'],
                    sent_num_layers=args['sent_num_layers'], output_dim=2,
                    output_attn=args.get('output_attn', True))
    else:
        raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path}")

    try:
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    except RuntimeError as e:
        if "weights_only" in str(e) and "Argument" in str(e):
            warnings.warn(f"torch.load failed with weights_only=False for {pth_path}. Trying without 'weights_only' argument. Error: {e}", UserWarning)
            checkpoint = torch.load(pth_path, map_location=device)
        else:
            raise e
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {pth_path}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint {pth_path}: {e}")
        raise

    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint:
        warnings.warn(f"Checkpoint {pth_path} does not contain a 'state_dict' key. Assuming the checkpoint itself is the state_dict.", UserWarning)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model state_dict from: {Path(pth_path).name}")
    except RuntimeError as e:
        print(f"Error loading state_dict into {net_type} model from {pth_path}: {e}")
        raise e

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    model = ensure_model_on_available_device(model)
    return model, args, vocab_stoi, pad_idx, unk_idx  # **fixed: return 5 items only**


def load_model_bert(arg_path: Path, pth_path: Path):
    with open(arg_path) as f:
        args = json.load(f)['args']

    rob_item = args.get('rob_item')
    rob_sent = args.get('rob_sent')
    if rob_sent is None and rob_item in ROB_ITEM_DESCRIPTIONS:
        rob_sent = ROB_ITEM_DESCRIPTIONS[rob_item]
    elif rob_sent is None:
        raise ValueError(f"RoB item description for '{rob_item}' not found in args or ROB_ITEM_DESCRIPTIONS.")

    distilbert_model_name_or_path = 'distilbert-base-uncased'
    try:
        config = AutoConfig.from_pretrained(
            distilbert_model_name_or_path,
            num_labels=2,
            return_dict=True,
            cache_dir=str(TRANSFORMERS_CACHE_PATH)
        )
        model = DistilClsConv(config)
        print(f"Initialized DistilClsConv architecture using config from '{distilbert_model_name_or_path}' (cache: {TRANSFORMERS_CACHE_PATH}).")

        distilbert_base_model = DistilBertModel.from_pretrained(
            distilbert_model_name_or_path,
            config=config,
            cache_dir=str(TRANSFORMERS_CACHE_PATH)
        )
        model.distilbert.load_state_dict(distilbert_base_model.state_dict())
        print(f"Loaded pre-trained weights for 'distilbert' part of DistilClsConv from cache.")
        del distilbert_base_model

        checkpoint = torch.load(pth_path, map_location=device)
        print(f"Loaded fine-tuned model state_dict from: {Path(pth_path).name}")

    except Exception as e:
        print(f"Error during DistilClsConv initialization or loading base DistilBertModel weights: {e}")
        print(f"Checked TRANSFORMERS_CACHE_PATH: {TRANSFORMERS_CACHE_PATH}")
        raise

    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint:
        warnings.warn(f"Checkpoint {pth_path} does not contain 'state_dict' key. Assuming checkpoint is state_dict.", UserWarning)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading fine-tuned state_dict for {Path(pth_path).name}: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading fine-tuned state_dict for {Path(pth_path).name}: {unexpected_keys}")
    except RuntimeError as e:
        print(f"Error loading fine-tuned state_dict into DistilClsConv model from {pth_path}: {e}")
        raise e

    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained(
        distilbert_model_name_or_path,
        use_fast=False,
        cache_dir=str(TRANSFORMERS_CACHE_PATH)
    )
    try:
        sent_model_name = 'distilbert-base-nli-stsb-mean-tokens'
        sent_model = SentenceTransformer(
            sent_model_name,
            device=device,
            cache_folder=str(SENTENCE_TRANSFORMERS_HOME_PATH)
        )
        print(f"Loaded SentenceTransformer model: {sent_model_name} (using cache: {SENTENCE_TRANSFORMERS_HOME_PATH})")
    except Exception as e:
        print(f"Error loading SentenceTransformer model {sent_model_name}: {e}")
        print(f"Checked SENTENCE_TRANSFORMERS_HOME_PATH: {SENTENCE_TRANSFORMERS_HOME_PATH}")
        raise

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = ensure_model_on_available_device(model)
    return model, tokenizer, sent_model, rob_sent

def pred_legacy(doc: str, model: torch.nn.Module, args: dict,
                vocab_stoi: dict, pad_idx: int, unk_idx: int) -> float:
    model.eval()
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]
    idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
    max_len = args.get('max_token_len', 512)
    if len(idx) < max_len:
        idx += [pad_idx] * (max_len - len(idx))
    elif len(idx) > max_len:
        idx = idx[:max_len]
    doc_tensor = torch.tensor(idx, dtype=torch.long).to(device)
    doc_tensor = doc_tensor.unsqueeze(1)
    with torch.no_grad():
        probs = model(doc_tensor)
    prob_positive = probs.squeeze().cpu().numpy()[1]
    return float(prob_positive)

def pred_bert(text: str, model: torch.nn.Module, tokenizer,
              sent_model: SentenceTransformer, rob_sent: str, max_n_sent: int = 30) -> float:
    model.eval()
    # Limit sentence count to prevent OOM
    sents = [str(s).strip() for s in nlp(text).sents if len(str(s).split()) > 3][:max_n_sent*3]
    if not sents:
        warnings.warn("No sentences found after filtering in pred_bert. Returning 0.0.", UserWarning)
        return 0.0
    try:
        sent_embeds = sent_model.encode(sents, convert_to_tensor=True, device=device, show_progress_bar=False)
        rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True, device=device, show_progress_bar=False)
    except Exception as e:
        print(f"Error during sentence encoding in pred_bert: {e}")
        return 0.0
    cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed)
    top_k = min(max_n_sent, len(sents))
    if top_k == 0:
        warnings.warn("No sentences to select for top_k in pred_bert (top_k is 0). Returning 0.0.", UserWarning)
        return 0.0
    top_vals, top_indices = torch.topk(
        cos_scores.squeeze(dim=-1),
        k=top_k,
        largest=True,
        sorted=True
    )
    top_indices = top_indices.cpu().numpy()
    sim_sents = [sents[i] for i in top_indices]
    sim_text = " ".join(sim_sents)
    if not sim_text.strip():
        warnings.warn("Sim_text (concatenated similar sentences) is empty before tokenization in pred_bert. Returning 0.0.", UserWarning)
        return 0.0
    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs
    prob_positive = probs.squeeze().cpu().numpy()[1]
    return float(prob_positive)

def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict,
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, num_sents: int) -> list[str]:
    if num_sents <= 0: return []
    model.eval()
    doc_spacy = nlp(doc_text)
    sents_text = [sent.text.strip() for sent in doc_spacy.sents]
    sents_tokens_orig = []
    sents_indices = []
    max_sent_len = args.get('max_sent_len', 100)
    for sent_text_item in sents_text:
        tokens = [tok.text.lower() for tok in nlp.tokenizer(sent_text_item)]
        if len(tokens) < 5: continue
        sents_tokens_orig.append(tokens)
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        if len(idx) < max_sent_len:
            idx += [pad_idx] * (max_sent_len - len(idx))
        else:
            idx = idx[:max_sent_len]
        sents_indices.append(idx)
    if not sents_indices:
        warnings.warn("No valid sentences found for HAN attention extraction after filtering.", UserWarning)
        return []
    max_doc_len = args.get('max_doc_len', 50)
    actual_sents_for_model_input = sents_indices[:max_doc_len]
    actual_sents_tokens_for_reconstruction = sents_tokens_orig[:max_doc_len]
    if len(actual_sents_for_model_input) < max_doc_len:
        pad_sentence_tensor_list = [pad_idx] * max_sent_len
        actual_sents_for_model_input.extend([pad_sentence_tensor_list] * (max_doc_len - len(actual_sents_for_model_input)))
    
    # Convert to tensor and add batch dimension
    doc_tensor = torch.tensor(actual_sents_for_model_input, dtype=torch.long).to(device)
    doc_tensor = doc_tensor.unsqueeze(0) # HAN expects [batch_size, num_sents, sent_len]

    # Get predictions and sentence attention scores from HAN model
    with torch.no_grad():
        outputs = model(doc_tensor)
        # Expecting (probabilities, sentence_attention_scores)
        if isinstance(outputs, tuple) and len(outputs) == 2 and hasattr(outputs[1], 'squeeze'):
            _probs, attn_score_tensor = outputs # probs are not used here, only attention
        else:
            warnings.warn("HAN model did not return expected (probs, attn_score_tensor) tuple "
                          "or attn_score_tensor is not a tensor. Cannot extract sentences by attention.", RuntimeWarning)
            return []
    
    # Process attention scores
    attn_scores_np = attn_score_tensor.squeeze().cpu().numpy() # Squeeze batch, move to CPU, to numpy
    
    # Align attention scores with the original sentences that were fed to the model
    num_sents_fed_to_model = len(actual_sents_tokens_for_reconstruction)
    aligned_scores = attn_scores_np[:num_sents_fed_to_model] 
    
    if len(aligned_scores) != num_sents_fed_to_model:
        warnings.warn(f"Attention score length ({len(aligned_scores)}) mismatch with "
                      f"number of sentences fed to model ({num_sents_fed_to_model}). Cannot reliably extract.", RuntimeWarning)
        return []

    # Create a DataFrame to sort sentences by attention score
    df = pd.DataFrame({
        'sent_tokens': actual_sents_tokens_for_reconstruction, 
        'attn': aligned_scores
    })
    df = df.sort_values(by=['attn'], ascending=False) # Sort by attention, highest first
    
    # Get the tokens of the top N sentences
    top_sent_tokens_lists = list(df['sent_tokens'][:num_sents])
    
    # Reconstruct sentences from tokens and apply basic cleaning
    out_sents = []
    for tokens_list_item in top_sent_tokens_lists: # Renamed to avoid conflict
        if not tokens_list_item: continue # Should not happen if actual_sents_tokens_for_reconstruction was correct
        sent = " ".join(tokens_list_item)
        # Simple regex cleaning for punctuation spacing
        sent = re.sub(r" \.", ".", sent); sent = re.sub(r" \,", ",", sent)
        sent = re.sub(r"\( ", "(", sent); sent = re.sub(r" \)", ")", sent)
        sent = re.sub(r"\[ ", "[", sent); sent = re.sub(r" \]", "]", sent)
        sent = re.sub(r" \- ", "-", sent)
        out_sents.append(sent)
        
    return out_sents