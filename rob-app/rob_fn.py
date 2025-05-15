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

# --- Hugging Face Offline Configuration ---
# These environment variables instruct the Hugging Face libraries (transformers, datasets)
# to operate in offline mode, meaning they should not attempt to download files from the Hub.
# They will rely on the pre-cached models available in the specified cache directories.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1" # Good practice, even if datasets lib isn't directly used for downloading

# --- Load spaCy model ---
# This attempts to load the 'en_core_web_sm' model.
# In a containerized environment, this model should be pre-installed into the
# Python environment (e.g., site-packages of the Conda env) during the image build.
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError as e:
    # If spacy.load() fails, it's a critical issue with the environment setup.
    print(f"CRITICAL ERROR: spaCy model 'en_core_web_sm' could not be loaded.")
    print(f"Attempted to load 'en_core_web_sm' using spacy.load().")
    print(f"Spacy version: {spacy.__version__}")
    print(f"Error details: {e}")
    print("This model should have been pip-installed directly into the Conda environment's site-packages "
          "during the image build (e.g., via setup.sh using a direct wheel URL).")
    print("If this error occurs, it means the model package is missing or incompatible in the environment.")
    print("Please check the Dockerfile (specifically the setup.sh execution and Conda environment consistency).")
    # Raising a RuntimeError makes the failure explicit and stops execution,
    # which is preferred over silently trying to download at runtime in a supposedly offline image.
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found or incompatible in the environment. "
        "The image build process may have failed to install it correctly."
    ) from e

# --- Import from updated libraries ---
from transformers import AutoTokenizer, DistilBertModel, AutoConfig # For BERT-based models
from sentence_transformers import SentenceTransformer, util # For sentence embeddings

# Import custom model definitions (ConvNet, AttnNet, HAN, DistilClsConv)
# These should be defined in a 'model.py' file accessible in the Python path.
from model import ConvNet, AttnNet, HAN, DistilClsConv

# Set device: Use GPU (cuda) if available, otherwise fall back to CPU.
# This is a global setting for PyTorch operations in this script.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Constants ---
# ROB_ITEM_DESCRIPTIONS provides default textual descriptions for various Risk of Bias items.
# These are used if specific descriptions are not provided in model argument files.
ROB_ITEM_DESCRIPTIONS = {
    'RandomizationTreatmentControl': 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment',
    'BlindedOutcomeAssessment': 'Assessment of an outcome in a blinded fashion. Investigators measuring the outcome do not know which treatment group the animals belongs to and what treatment they had received',
    'SampleSizeCalculation': 'The manuscript reports the performance of a sample size calculation and describes how this number was derived statistically',
    'AnimalExclusions': 'All animals, all data and all outcomes measured are accounted for and presented in the final analysis. Reasons are given for animal exclusions',
    'AllocationConcealment': 'Investigators performing the experiment do not know which treatment an animal is being given',
    'AnimalWelfareRegulations': 'Research investigators complied with animal welfare regulations',
    'ConflictsOfInterest': 'Potential conflict of interest, like funding or affiliation to a pharmaceutical company'
}

# --- Hugging Face Cache Paths (derived from Dockerfile ENV VARS) ---
# These paths determine where the script expects to find pre-cached Hugging Face models.
# They default to standard local cache locations if the environment variables are not set (e.g., during local non-Docker execution).
HF_HOME_DEFAULT = Path.home() / ".cache" / "huggingface" # Default base for HF cache
HF_HOME_PATH = Path(os.environ.get('HF_HOME', HF_HOME_DEFAULT))

# Specific cache for models downloaded via transformers library (e.g., AutoModel, AutoTokenizer)
TRANSFORMERS_CACHE_PATH = Path(os.environ.get('TRANSFORMERS_CACHE', HF_HOME_PATH / "hub"))
# Specific cache for models downloaded via sentence-transformers library
SENTENCE_TRANSFORMERS_HOME_PATH = Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', HF_HOME_PATH / "sentence_transformers"))

print(f"Expecting Hugging Face models (transformers lib) in TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE_PATH}")
print(f"Expecting SentenceTransformer models in SENTENCE_TRANSFORMERS_HOME: {SENTENCE_TRANSFORMERS_HOME_PATH}")


# --- Model Loading Functions ---

def load_vocab_info(fld_path: Path):
    """
    Loads vocabulary information (stoi, pad_idx, unk_idx) from a dill/pickled legacy torchtext.Field file.

    This function is a compatibility layer to extract necessary vocabulary details
    from older Field objects without needing the full torchtext.legacy.data module at runtime
    if the Field object was saved in a specific way or if a minimal mock is used.

    Args:
        fld_path (Path): Path to the .Field file.

    Returns:
        tuple: (vocab_stoi, pad_idx, unk_idx)
            - vocab_stoi (dict): String-to-index mapping for vocabulary.
            - pad_idx (int): Index for the padding token.
            - unk_idx (int): Index for the unknown token.
    Raises:
        ValueError: If the loaded object doesn't have the expected attributes.
        Exception: For other file loading or dill processing errors.
    """
    try:
        with open(fld_path, "rb") as fin:
            # This section attempts to mock the torchtext.legacy.data.Field structure
            # sufficiently for dill to load old pickles, even if torchtext.legacy is not
            # fully available or if its version changed.
            import sys, types
            try:
                from torchtext.legacy.data import Field as _LegacyField # Try importing if available
            except ModuleNotFoundError:
                # If torchtext.legacy.data.Field is not found, create a minimal stand-in class.
                # Dill will populate its attributes from the pickle.
                class _LegacyField:
                    def __init__(self, *_, **__): pass # Accept any arguments during unpickling
                    # Define attributes that dill will overwrite, ensuring they exist.
                    vocab = types.SimpleNamespace(stoi={}) # Mock vocab object with stoi
                    pad_token = "<pad>" # Default pad token string
                    unk_token = "<unk>" # Default unknown token string
            
            # Register the (potentially mocked) _LegacyField class under the name
            # that dill expects to find when unpickling the Field object.
            _field_mod = types.ModuleType("torchtext.data.field")
            _field_mod.Field = _LegacyField
            sys.modules["torchtext.data.field"] = _field_mod
            
            loaded_obj = dill.load(fin) # Unpickle the object

        # Check if the unpickled object has the necessary attributes of a Field object.
        if hasattr(loaded_obj, 'vocab') and hasattr(loaded_obj.vocab, 'stoi') and \
           hasattr(loaded_obj, 'pad_token') and hasattr(loaded_obj, 'unk_token'):
            
            warnings.warn(f"Loaded legacy torchtext.Field object from {Path(fld_path).name}. ", DeprecationWarning)
            vocab_stoi = loaded_obj.vocab.stoi
            pad_token = loaded_obj.pad_token
            unk_token = loaded_obj.unk_token
            # Get pad/unk indices from vocab, with defaults if tokens aren't in stoi (e.g. if stoi is minimal)
            pad_idx = vocab_stoi.get(pad_token, 1) # Default pad_idx for torchtext.legacy
            unk_idx = vocab_stoi.get(unk_token, 0) # Default unk_idx for torchtext.legacy
            print(f"Loaded vocab info from legacy Field: {Path(fld_path).name}")
            return vocab_stoi, pad_idx, unk_idx
        else:
            # If the object doesn't look like a Field object.
            raise ValueError(f"Cannot extract vocab info from {fld_path}. "
                             "Loaded object does not have expected legacy Field object attributes (vocab.stoi, pad_token, unk_token).")
    except Exception as e:
        # Catch-all for other errors during file I/O or unpickling.
        print(f"Error loading or processing vocab file {fld_path}: {e}")
        raise

def load_model_legacy(arg_path: Path, pth_path: Path, fld_path: Path):
    """
    Loads legacy models (ConvNet, AttnNet, HAN).
    These models typically use a pre-defined vocabulary and embeddings.

    Args:
        arg_path (Path): Path to the JSON file containing model arguments/hyperparameters.
        pth_path (Path): Path to the .pth.tar file containing trained model weights (state_dict).
        fld_path (Path): Path to the .Field file containing vocabulary information.

    Returns:
        tuple: (model, args, vocab_stoi, pad_idx, unk_idx)
            - model (torch.nn.Module): The loaded PyTorch model, moved to the appropriate device and set to eval mode.
            - args (dict): Dictionary of model arguments from the JSON file.
            - vocab_stoi (dict): String-to-index vocabulary mapping.
            - pad_idx (int): Padding token index.
            - unk_idx (int): Unknown token index.
    """
    # Load model arguments from JSON file
    with open(arg_path) as f:
        args = json.load(f)['args'] # Assumes 'args' key exists
    
    # Load vocabulary information
    try:
        vocab_stoi, pad_idx, unk_idx = load_vocab_info(fld_path)
        # Determine vocabulary size. Original script used max_vocab_size + 2 (for pad/unk).
        # Using len(vocab_stoi) is more robust if stoi is complete.
        vocab_size = len(vocab_stoi) 
        # Fallback to original logic if max_vocab_size implies a larger embedding matrix was trained
        if 'max_vocab_size' in args and (args['max_vocab_size'] + 2) > vocab_size:
             vocab_size = args['max_vocab_size'] + 2
    except Exception as e:
        print(f"Failed to load vocabulary from {fld_path} for legacy model. Error: {e}")
        raise

    # Instantiate model based on 'net_type' specified in args
    net_type = args.get('net_type', 'unknown') # Default to 'unknown' if not specified
    model: torch.nn.Module # Type hint
    if net_type == 'cnn':
        sizes = [int(s) for s in args['filter_sizes'].split(',')] # Filter sizes for ConvNet
        model = ConvNet(vocab_size=vocab_size, embedding_dim=args['embed_dim'], n_filters=args['num_filters'],
                        filter_sizes=sizes, output_dim=2, dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'])
    elif net_type == 'attn':
        model = AttnNet(vocab_size=vocab_size, embedding_dim=args['embed_dim'], rnn_hidden_dim=args['rnn_hidden_dim'],
                        rnn_num_layers=args['rnn_num_layers'], output_dim=2, bidirection=args['bidirection'],
                        rnn_cell_type=args['rnn_cell_type'], dropout=args['dropout'], pad_idx=pad_idx,
                        embed_trainable=args['embed_trainable'], batch_norm=args['batch_norm'], 
                        output_attn=args.get('output_attn', False)) # HAN needs attention output
    elif net_type == 'han':
        model = HAN(vocab_size=vocab_size, embedding_dim=args['embed_dim'], word_hidden_dim=args['word_hidden_dim'],
                    word_num_layers=args['word_num_layers'], pad_idx=pad_idx, embed_trainable=args['embed_trainable'],
                    batch_norm=args['batch_norm'], sent_hidden_dim=args['sent_hidden_dim'],
                    sent_num_layers=args['sent_num_layers'], output_dim=2, # Assuming binary classification
                    output_attn=args.get('output_attn', True)) # HAN typically outputs attention for sentence extraction
    else:
        raise ValueError(f"Unsupported net_type '{net_type}' in args file {arg_path}")

    # Load model weights (state_dict) from checkpoint file
    try:
        # PyTorch 2.6+ defaults weights_only=True. Set to False if pickle contains more than just weights.
        # Older PyTorch versions might not have 'weights_only' argument.
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False) 
    except RuntimeError as e:
        # Handle cases where 'weights_only' is not a valid argument (older PyTorch)
        if "weights_only" in str(e) and "Argument" in str(e): 
            warnings.warn(f"torch.load failed with weights_only=False for {pth_path}. Trying without 'weights_only' argument. Error: {e}", UserWarning)
            checkpoint = torch.load(pth_path, map_location=device)
        else: # Re-raise other RuntimeErrors
            raise e 
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {pth_path}")
        raise
    except Exception as e: # Catch other potential errors during torch.load
        print(f"Error loading checkpoint {pth_path}: {e}")
        raise

    # Extract state_dict from checkpoint (common to save it under 'state_dict' key)
    state_dict = checkpoint.get('state_dict', checkpoint) # Fallback if checkpoint is the state_dict itself
    if 'state_dict' not in checkpoint:
        warnings.warn(f"Checkpoint {pth_path} does not contain a 'state_dict' key. Assuming the checkpoint itself is the state_dict.", UserWarning)

    # Load the state_dict into the instantiated model
    try:
        model.load_state_dict(state_dict, strict=True) # strict=True ensures all keys match
        print(f"Loaded model state_dict from: {Path(pth_path).name}")
    except RuntimeError as e:
        print(f"Error loading state_dict into {net_type} model from {pth_path}: {e}")
        # Potentially print missing/unexpected keys here for debugging if needed
        raise e

    # Finalize model setup
    if device.type == 'cuda': torch.cuda.empty_cache() # Clear unused GPU memory
    model.to(device) # Move model to the target device
    model.eval()     # Set model to evaluation mode (disables dropout, batchnorm updates etc.)
    
    return model, args, vocab_stoi, pad_idx, unk_idx


def load_model_bert(arg_path: Path, pth_path: Path):
    """
    Loads BERT-based models (specifically DistilClsConv) using pre-cached Hugging Face models.
    Ensures that model loading respects offline settings and uses specified cache directories.

    Args:
        arg_path (Path): Path to the JSON file containing model arguments.
        pth_path (Path): Path to the .pth.tar file containing the fine-tuned DistilClsConv weights.

    Returns:
        tuple: (model, tokenizer, sent_model, rob_sent)
            - model (DistilClsConv): The loaded DistilClsConv model.
            - tokenizer (PreTrainedTokenizer): The tokenizer for the BERT model.
            - sent_model (SentenceTransformer): The sentence embedding model.
            - rob_sent (str): The RoB item description sentence.
    """
    # Load model arguments
    with open(arg_path) as f:
        args = json.load(f)['args']

    # Determine RoB item sentence (used for similarity in pred_bert)
    rob_item = args.get('rob_item')
    rob_sent = args.get('rob_sent') # Get from args if provided
    if rob_sent is None and rob_item in ROB_ITEM_DESCRIPTIONS: # Fallback to default descriptions
        rob_sent = ROB_ITEM_DESCRIPTIONS[rob_item]
    elif rob_sent is None: # Error if no description can be found
        raise ValueError(f"RoB item description for '{rob_item}' not found in args or ROB_ITEM_DESCRIPTIONS.")

    # Name of the base DistilBERT model to load (should be pre-cached)
    distilbert_model_name_or_path = 'distilbert-base-uncased' 
    
    try:
        # 1. Load the configuration for the base DistilBERT model.
        #    `cache_dir` ensures it uses the pre-cached files.
        #    `TRANSFORMERS_OFFLINE=1` prevents network calls.
        config = AutoConfig.from_pretrained(
            distilbert_model_name_or_path,
            num_labels=2, # For binary classification head in DistilClsConv
            return_dict=True, # Standard for newer transformers models
            cache_dir=str(TRANSFORMERS_CACHE_PATH) # Point to the pre-cached location
        )
        
        # 2. Initialize the custom DistilClsConv model architecture using this config.
        #    This sets up the layers but doesn't load any weights yet.
        model = DistilClsConv(config) 
        print(f"Initialized DistilClsConv architecture using config from '{distilbert_model_name_or_path}' (cache: {TRANSFORMERS_CACHE_PATH}).")

        # 3. Load pre-trained weights into the `distilbert` part of our `DistilClsConv` model.
        #    This ensures the BERT backbone has its standard weights.
        distilbert_base_model = DistilBertModel.from_pretrained(
            distilbert_model_name_or_path,
            config=config, # Use the same config
            cache_dir=str(TRANSFORMERS_CACHE_PATH)
        )
        model.distilbert.load_state_dict(distilbert_base_model.state_dict())
        print(f"Loaded pre-trained weights for 'distilbert' part of DistilClsConv from cache.")
        del distilbert_base_model # Free up memory

    except Exception as e:
        print(f"Error during DistilClsConv initialization or loading base DistilBertModel weights: {e}")
        print(f"Checked TRANSFORMERS_CACHE_PATH: {TRANSFORMERS_CACHE_PATH}")
        # For debugging, one might check if the expected model directory exists in the cache, e.g.:
        # expected_model_dir = TRANSFORMERS_CACHE_PATH / f"models--{distilbert_model_name_or_path.replace('/', '--')}"
        # print(f"Checking for model directory: {expected_model_dir}, Exists: {expected_model_dir.exists()}")
        raise

    # 4. Load the fine-tuned checkpoint for the *entire* DistilClsConv model.
    #    This will overwrite the base distilbert weights if they were fine-tuned,
    #    and load weights for the custom convolutional head.
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Fine-tuned model checkpoint file not found at {pth_path}")
        raise
    except Exception as e:
        print(f"Error loading fine-tuned checkpoint {pth_path}: {e}")
        raise

    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'state_dict' not in checkpoint:
        warnings.warn(f"Checkpoint {pth_path} does not contain 'state_dict' key. Assuming checkpoint is state_dict.", UserWarning)
    
    try:
        # Use strict=False because:
        # a) We've already loaded base distilbert weights. If the fine-tuned checkpoint
        #    also contains them, they'll be overwritten (which is fine).
        # b) If the checkpoint *only* contains the head and modified distilbert layers,
        #    strict=True would fail. strict=False allows loading a partial state_dict.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading fine-tuned state_dict for {Path(pth_path).name}: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading fine-tuned state_dict for {Path(pth_path).name}: {unexpected_keys}")
        print(f"Loaded fine-tuned model state_dict from: {Path(pth_path).name}")
    except RuntimeError as e:
        print(f"Error loading fine-tuned state_dict into DistilClsConv model from {pth_path}: {e}")
        raise e

    model.to(device)
    model.eval()

    # Load tokenizer (will use pre-cached files due to TRANSFORMERS_OFFLINE=1 and cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        distilbert_model_name_or_path,
        cache_dir=str(TRANSFORMERS_CACHE_PATH)
    )

    # Load SentenceTransformer model (will use pre-cached files)
    try:
        sent_model_name = 'distilbert-base-nli-stsb-mean-tokens' # Or a more modern alternative like 'all-MiniLM-L6-v2'
        sent_model = SentenceTransformer(
            sent_model_name,
            device=device,
            cache_folder=str(SENTENCE_TRANSFORMERS_HOME_PATH) # Specific cache for sentence-transformers
        )
        print(f"Loaded SentenceTransformer model: {sent_model_name} (using cache: {SENTENCE_TRANSFORMERS_HOME_PATH})")
    except Exception as e:
        print(f"Error loading SentenceTransformer model {sent_model_name}: {e}")
        print(f"Checked SENTENCE_TRANSFORMERS_HOME_PATH: {SENTENCE_TRANSFORMERS_HOME_PATH}")
        # For debugging, check if SENTENCE_TRANSFORMERS_HOME_PATH / "sentence-transformers_distilbert-base-nli-stsb-mean-tokens" exists
        raise

    if device.type == 'cuda': torch.cuda.empty_cache()
    return model, tokenizer, sent_model, rob_sent

# --- Prediction Functions ---

def pred_legacy(doc: str, model: torch.nn.Module, args: dict, 
                vocab_stoi: dict, pad_idx: int, unk_idx: int) -> float:
    """
    Generates a prediction score using a legacy model (ConvNet, AttnNet).

    Args:
        doc (str): The input document text.
        model (torch.nn.Module): The pre-loaded legacy PyTorch model.
        args (dict): Model-specific arguments (e.g., max_token_len).
        vocab_stoi (dict): Vocabulary string-to-index mapping.
        pad_idx (int): Padding token index.
        unk_idx (int): Unknown token index.

    Returns:
        float: The prediction probability for the positive class.
    """
    model.eval() # Ensure model is in evaluation mode

    # Tokenize document using spaCy and convert to lowercase
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]
    
    # Convert tokens to numerical indices using the vocabulary
    idx = [vocab_stoi.get(t, unk_idx) for t in tokens] # Use unk_idx for out-of-vocabulary tokens
    
    # Pad or truncate the sequence to match model's expected input length
    max_len = args.get('max_token_len', 512) # Get max length from args, default if not present
    if len(idx) < max_len:
        idx += [pad_idx] * (max_len - len(idx)) # Pad
    elif len(idx) > max_len:
        idx = idx[:max_len] # Truncate
    
    # Convert list of indices to a PyTorch tensor
    doc_tensor = torch.tensor(idx, dtype=torch.long).to(device)
    # Add batch dimension (batch_size=1) as models expect [seq_len, batch_size] or [batch_size, seq_len]
    doc_tensor = doc_tensor.unsqueeze(1) # For models expecting [seq_len, batch_size=1]
    
    # Perform prediction
    with torch.no_grad(): # Disable gradient calculations for inference
        probs = model(doc_tensor) # Output shape typically [batch_size=1, num_classes=2]
    
    # Extract the probability of the positive class (assuming index 1)
    prob_positive = probs.squeeze().cpu().numpy()[1] # Squeeze batch dim, move to CPU, convert to numpy, get class 1 prob
    
    return float(prob_positive)

def pred_bert(text: str, model: torch.nn.Module, tokenizer, 
              sent_model: SentenceTransformer, rob_sent: str, max_n_sent: int = 30) -> float:
    """
    Generates a prediction score using a BERT-based model (DistilClsConv).
    This method involves selecting relevant sentences based on similarity to `rob_sent`
    and then classifying the concatenated relevant sentences.

    Args:
        text (str): The input document text.
        model (torch.nn.Module): The pre-loaded DistilClsConv model.
        tokenizer: The Hugging Face tokenizer for the BERT model.
        sent_model (SentenceTransformer): The pre-loaded sentence embedding model.
        rob_sent (str): The reference sentence for the RoB item, used for similarity scoring.
        max_n_sent (int, optional): Maximum number of similar sentences to use for prediction. Defaults to 30.

    Returns:
        float: The prediction probability for the positive class.
    """
    model.eval() # Ensure model is in evaluation mode

    # Split text into sentences using spaCy
    doc_spacy = nlp(text) # Use a different variable name to avoid conflict if 'doc' is used elsewhere
    # Filter sentences: keep those with more than 3 words
    sents = [str(s).strip() for s in doc_spacy.sents if len(str(s).split()) > 3] 
    
    if not sents:
        warnings.warn("No sentences found after filtering in pred_bert. Returning 0.0.", UserWarning)
        return 0.0 
    
    # Encode document sentences and the RoB reference sentence
    try:
        sent_embeds = sent_model.encode(sents, convert_to_tensor=True, device=device, show_progress_bar=False)
        rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True, device=device, show_progress_bar=False)
    except Exception as e:
        print(f"Error during sentence encoding in pred_bert: {e}")
        return 0.0 # Return neutral probability on error
    
    # Compute cosine similarity between document sentences and the RoB sentence
    cos_scores = util.pytorch_cos_sim(sent_embeds, rob_embed) # Shape: [num_doc_sents, 1]
    
    # Select top N most similar sentences
    top_k = min(max_n_sent, len(sents))
    if top_k == 0: # Should not happen if 'sents' is not empty, but as a safeguard
         warnings.warn("No sentences to select for top_k in pred_bert (top_k is 0). Returning 0.0.", UserWarning)
         return 0.0
         
    # Squeeze cos_scores to 1D if it's [N,1] before topk, then get indices
    top_results = torch.topk(cos_scores.squeeze(dim=-1), k=top_k) 
    top_indices = top_results.indices.cpu().numpy()
    
    # Concatenate the most similar sentences
    sim_sents = [sents[i] for i in top_indices]
    sim_text = " ".join(sim_sents) 
    if not sim_text.strip(): # If somehow the joined text is empty
        warnings.warn("Sim_text (concatenated similar sentences) is empty before tokenization in pred_bert. Returning 0.0.", UserWarning)
        return 0.0

    # Tokenize the concatenated similar sentences for the DistilClsConv model
    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs) # DistilClsConv model should return probabilities directly
    
    # The DistilClsConv model in model.py applies softmax internally.
    # If it returned logits, you would need: probs = torch.softmax(outputs.logits, dim=-1)
    probs = outputs 
    
    prob_positive = probs.squeeze().cpu().numpy()[1] # Extract positive class probability
    return float(prob_positive)

def extract_sent_han(doc_text: str, model: torch.nn.Module, args: dict, 
                     vocab_stoi: dict, pad_idx: int, unk_idx: int, num_sents: int) -> list[str]:
    """
    Extracts the most relevant sentences from a document based on attention scores
    from a Hierarchical Attention Network (HAN) model.

    Args:
        doc_text (str): The input document text.
        model (torch.nn.Module): The pre-loaded HAN model (must be configured to output attention).
        args (dict): Model-specific arguments (e.g., max_sent_len, max_doc_len).
        vocab_stoi (dict): Vocabulary string-to-index mapping.
        pad_idx (int): Padding token index.
        unk_idx (int): Unknown token index.
        num_sents (int): The number of top sentences to extract.

    Returns:
        list[str]: A list of the extracted sentences, ordered by relevance.
                   Returns an empty list if num_sents <= 0 or if extraction fails.
    """
    if num_sents <= 0: return [] # No sentences to extract
    model.eval() # Ensure model is in evaluation mode

    # Split document into sentences
    doc_spacy = nlp(doc_text)
    sents_text = [sent.text.strip() for sent in doc_spacy.sents]
    
    sents_tokens_orig = [] # Store original tokens for reconstructing sentences
    sents_indices = []     # Store numericalized sentences for model input
    max_sent_len = args.get('max_sent_len', 100) # Max words per sentence

    # Process each sentence
    for sent_text_item in sents_text: # Renamed to avoid conflict
        tokens = [tok.text.lower() for tok in nlp.tokenizer(sent_text_item)]
        if len(tokens) < 5: continue # Skip very short sentences (heuristic)
        
        sents_tokens_orig.append(tokens) # Store for later reconstruction
        
        # Numericalize and pad/truncate sentence
        idx = [vocab_stoi.get(t, unk_idx) for t in tokens]
        if len(idx) < max_sent_len:
            idx += [pad_idx] * (max_sent_len - len(idx))
        else:
            idx = idx[:max_sent_len]
        sents_indices.append(idx)

    if not sents_indices:
        warnings.warn("No valid sentences found for HAN attention extraction after filtering.", UserWarning)
        return []

    # Prepare document for HAN model (pad/truncate number of sentences)
    max_doc_len = args.get('max_doc_len', 50) # Max sentences per document
    
    # These are the sentences that will actually be fed to the model (after truncation if needed)
    actual_sents_for_model_input = sents_indices[:max_doc_len]
    # Corresponding original tokens for these sentences (for reconstruction)
    actual_sents_tokens_for_reconstruction = sents_tokens_orig[:max_doc_len]

    # Pad the document (list of sentences) if it's shorter than max_doc_len
    if len(actual_sents_for_model_input) < max_doc_len:
        pad_sentence_tensor_list = [pad_idx] * max_sent_len # A sentence full of padding
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

