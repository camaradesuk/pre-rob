#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Risk of Bias (RoB) prediction.
Processes text files, predicts RoB scores using various models,
and optionally extracts relevant sentences.

This script incorporates:
- Enhanced logging and error handling.
- Processing time tracking for each file.
- Hard timeouts for individual file processing using multiprocessing.
- Flattened sentence extraction in the output CSV.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import logging
import multiprocessing
import queue # For queue.Empty exception

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Import functions from rob_fn ---
# NOTE: rob_fn.py will also need modifications for nlp handling with multiprocessing.
try:
    from rob_fn import (
        load_model_legacy,
        load_model_bert,
        pred_legacy, # Will need to accept nlp_instance
        pred_bert,   # Will need to accept nlp_instance
        extract_sent_han, # Will need to accept nlp_instance
        FILE_PROCESSING_TIMEOUT_SECONDS,
        ROB_ITEM_CATEGORIES # Define this in rob_fn.py e.g. ['random', 'blind', 'interest', 'welfare', 'exclusion']
    )
    # We will not use the global nlp from rob_fn in the worker.
    # Each worker process will initialize its own spacy instance.
except ImportError as e:
    logger.critical(f"Critical import error from rob_fn: {e}. Ensure rob_fn.py is accessible, correct, and defines necessary constants/functions.", exc_info=True)
    raise

# Worker function for multiprocessing
def _process_single_file_worker(
    file_path_str: str,
    models_loaded_dict: Dict[str, Any],
    p_ref_compiled: re.Pattern,
    num_sents_val: int,
    output_q: multiprocessing.Queue,
    current_id_val: str,
    # Local process_text_fn, pred_legacy_fn, etc. to avoid pickling issues with rob_fn module if complex
    # Or ensure rob_fn functions are top-level and picklable
    process_text_fn_ref,
    pred_legacy_fn_ref,
    pred_bert_fn_ref,
    extract_sent_han_fn_ref,
    rob_item_categories_list: List[str]
):
    """
    Worker function to process a single text file.
    This function is executed in a separate process.
    Initializes its own spaCy nlp instance.
    """
    # Initialize spaCy nlp instance within the worker process
    # This is crucial for safe multiprocessing with spaCy.
    nlp_instance = None
    try:
        import spacy
        nlp_instance = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # logger.info(f"Worker for {current_id_val}: spaCy nlp instance loaded.") # This logger might not work as expected in subprocess without specific config
        print(f"Worker for {current_id_val}: spaCy nlp instance loaded.") # Simple print for worker
    except Exception as e_nlp:
        # logger.error(f"Worker for {current_id_val}: Failed to load spaCy model: {e_nlp}", exc_info=True)
        print(f"Worker for {current_id_val}: Failed to load spaCy model: {e_nlp}")
        # Construct an error result and put it on the queue
        error_score = {
            "id": current_id_val, "txt_path": file_path_str,
            "message": "Failed to load spaCy in worker", "status": "Error",
            "error_details": f"spaCy load error: {e_nlp}", "processing_time_seconds": 0.0,
        }
        for cat in rob_item_categories_list: error_score[cat] = -3.0 # Error code for NLP fail
        if num_sents_val > 0: error_score['sentences'] = {cat: ["NLP Load Error"] for cat in rob_item_categories_list}
        output_q.put(error_score)
        return

    start_time = time.time()
    file_status = "OK"
    error_msg = ""
    path = Path(file_path_str)

    score: Dict[str, Any] = {
        "id": current_id_val, "txt_path": file_path_str,
        "message": "Processing started in worker",
        "processing_time_seconds": 0.0, "status": "Processing", "error_details": ""
    }
    for cat in rob_item_categories_list: score[cat] = 999.0 # Default
    if num_sents_val > 0:
        score['sentences'] = {cat: [] for cat in rob_item_categories_list}

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
            raw_text = fin.read()

        if not raw_text.strip():
            score["message"] = "File is empty or whitespace only"
            file_status = "Skipped"
            error_msg = "File content empty"
        else:
            processed_text = process_text_fn_ref(raw_text, p_ref_compiled) # Use passed PreRob.process_text
            if not processed_text.strip():
                score["message"] = "Text empty after processing"
                file_status = "Skipped"
                error_msg = "Text empty after processing"
            else:
                # Legacy Model Predictions
                for key in rob_item_categories_list:
                    if key == 'welfare': continue # Welfare is BERT
                    if key in models_loaded_dict:
                        model_data = models_loaded_dict[key]
                        if model_data:
                            model, model_args, vocab_stoi, pad_idx, unk_idx = model_data
                            try:
                                score[key] = pred_legacy_fn_ref(processed_text, model, model_args, vocab_stoi, pad_idx, unk_idx, nlp_instance)
                            except Exception as e:
                                score[key] = -1.0
                                file_status = "Error"
                                error_msg += f"Pred error ({key}): {e}; "
                        else: score[key] = -2.0
                    else: score[key] = -2.0

                # BERT Model Prediction (Welfare)
                if 'welfare' in models_loaded_dict:
                    model_data = models_loaded_dict['welfare']
                    if model_data:
                        model, tokenizer, sent_model, rob_sent_desc = model_data
                        try:
                            score['welfare'] = pred_bert_fn_ref(processed_text, model, tokenizer, sent_model, rob_sent_desc, nlp_instance, max_n_sent=30)
                        except Exception as e:
                            score['welfare'] = -1.0
                            file_status = "Error"
                            error_msg += f"Pred error (welfare): {e}; "
                    else: score['welfare'] = -2.0
                else: score['welfare'] = -2.0 if 'welfare' in rob_item_categories_list else score.get('welfare', 999.0)


                # Sentence Extraction
                if num_sents_val > 0:
                    # score['sentences'] already initialized
                    model_prefix = 's_'
                    for key in rob_item_categories_list:
                        s_key = model_prefix + key
                        if s_key in models_loaded_dict:
                            model_data = models_loaded_dict[s_key]
                            if model_data:
                                s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx = model_data
                                try:
                                    score['sentences'][key] = extract_sent_han_fn_ref(
                                        processed_text, s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx, num_sents_val, nlp_instance
                                    )
                                except Exception as e:
                                    score['sentences'][key] = [f"Extraction Error: {e}"]
                                    file_status = "Error"
                                    error_msg += f"SentExt error ({key}): {e}; "
                            else: score['sentences'][key] = ["Sentence Model not loaded"]
                        else: score['sentences'][key] = ["Sentence Model not in config"]
                
                if file_status == "OK" and not error_msg:
                    score["message"] = "Processing successful in worker"

    except FileNotFoundError:
        score["message"] = "File not found in worker"
        file_status = "Error"
        error_msg = f"File not found: {path}"
    except Exception as e:
        score["message"] = f"General processing error in worker: {e}"
        file_status = "Error"
        error_msg = f"General error in worker: {e}"

    processing_duration = round(time.time() - start_time, 3)
    score["processing_time_seconds"] = processing_duration
    score["status"] = file_status
    score["error_details"] = error_msg.strip()
    # Timeout is handled by the main process, but we log actual duration here.
    # If this worker completes, it means it did *not* hit the hard timeout from the main process's perspective.
    # However, it might have internally taken longer than a soft threshold if one were implemented here.

    output_q.put(score)


class PreRob():
    """
    Handles the preprocessing and prediction of Risk of Bias scores from text files.
    """
    def __init__(self, txt_info: str):
        self.txt_info = Path(txt_info)
        self.txt_paths: List[Path] = []
        self.ids: List[str] = []
        logger.info(f"PreRob initialized with input: {txt_info}")

    def get_txt_paths(self) -> None:
        # This method remains largely the same as in the previous version.
        # Minor logging adjustments might be considered if needed.
        txt_info_path = self.txt_info
        self.txt_paths = []
        self.ids = []

        if not txt_info_path.exists() and isinstance(self.txt_info, str) and ".txt," not in str(self.txt_info):
            logger.error(f"Input path does not exist: {txt_info_path}")
            return

        if txt_info_path.is_dir():
            logger.info(f"Scanning directory: {txt_info_path}")
            self.txt_paths = sorted(list(txt_info_path.rglob("*.txt")))
            self.ids = [p.stem + f"_{i}" for i, p in enumerate(self.txt_paths)] # Ensure unique IDs if names clash
            logger.info(f"Found {len(self.txt_paths)} .txt files in directory.")
        elif txt_info_path.is_file() and txt_info_path.suffix == ".txt":
            logger.info(f"Processing single file: {txt_info_path}")
            self.txt_paths.append(txt_info_path.resolve())
            self.ids.append(Path(txt_info_path).stem)
        elif txt_info_path.is_file() and txt_info_path.suffix == ".csv":
            logger.info(f"Reading paths from CSV: {txt_info_path}")
            try:
                path_df = pd.read_csv(txt_info_path, sep=',')
                if 'path' not in path_df.columns or 'id' not in path_df.columns:
                    logger.error("CSV must contain 'path' and 'id' columns.")
                    raise ValueError("CSV must contain 'path' and 'id' columns.")
                
                base_dir = txt_info_path.parent
                for _, row in path_df.iterrows():
                    relative_path_str = str(row['path']).strip()
                    if not relative_path_str:
                        logger.warning(f"Empty path found in CSV row with id {row['id']}. Skipping.")
                        continue
                    relative_path = Path(relative_path_str)
                    txt_path = relative_path if relative_path.is_absolute() else base_dir / relative_path
                    
                    if txt_path.is_file() and txt_path.suffix == ".txt":
                        self.txt_paths.append(txt_path.resolve())
                        self.ids.append(str(row['id']))
                    else:
                        logger.warning(f"File not found or not a .txt file in CSV row (id: {row['id']}): {txt_path}")
                logger.info(f"Found {len(self.txt_paths)} valid .txt paths from CSV.")
            except pd.errors.EmptyDataError:
                logger.error(f"CSV file is empty: {txt_info_path}")
            except Exception as e:
                logger.error(f"Error reading or processing CSV file {txt_info_path}: {e}", exc_info=True)
        elif isinstance(self.txt_info, (str, Path)) and ".txt," in str(self.txt_info):
            logger.info("Processing comma-separated paths string.")
            paths_str_list = str(self.txt_info).split(".txt,")
            temp_paths = []
            for i, p_str_segment in enumerate(paths_str_list):
                path_str = p_str_segment.strip()
                if not path_str.endswith(".txt") and i < len(paths_str_list) -1 :
                     path_str += ".txt"
                elif not path_str.endswith(".txt") and i == len(paths_str_list) -1 and not Path(path_str).exists():
                     potential_path_with_ext = Path(path_str + ".txt")
                     if potential_path_with_ext.exists(): path_str += ".txt"
                txt_path = Path(path_str)
                if txt_path.is_file() and txt_path.suffix == ".txt":
                    temp_paths.append(txt_path.resolve())
                else:
                    logger.warning(f"File not found or not a .txt in comma-separated list: {txt_path}")
            self.txt_paths = temp_paths
            self.ids = [txt_p.stem + f"_{i}" for i, txt_p in enumerate(self.txt_paths)]
            logger.info(f"Found {len(self.txt_paths)} valid .txt paths from comma-separated string.")
        if not self.txt_paths:
            logger.warning(f"No .txt files found to process based on input: {self.txt_info}")

    def process_text(self, text: str, p_ref: re.Pattern) -> str:
        # This method is passed to the worker, so it must be self-contained or
        # rely only on picklable arguments. Its logic remains the same.
        try:
            processed_text = re.sub(r".*?(Introduction|INTRODUCTION)\s*\n+", " ", text, count=1, flags=re.DOTALL | re.IGNORECASE)
            if processed_text == text: processed_text = text
            s = p_ref.search(processed_text)
            if s and s.start() > 0: processed_text = processed_text[:s.start()]
            elif s and s.start() == 0:
                 logger.debug("Reference pattern matched at the beginning of the text in process_text.") # Changed to debug
                 processed_text = ""
            processed_text = re.sub(r"\s+[\[][^a-zA-Z]+[\]]", "", processed_text)
            processed_text = re.sub(r"https?:/\/\S+", " ", processed_text)
            processed_text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", processed_text, flags=re.MULTILINE)
            processed_text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", processed_text, flags=re.MULTILINE)
            processed_text = processed_text.encode("ascii", errors="ignore").decode()
            processed_text = re.sub(r'\s+', " ", processed_text).strip()
            return processed_text
        except Exception as e:
            logger.error(f"Error during text processing (called by worker or main): {e}", exc_info=True)
            return ""

    def predict_probs(self, models: Dict[str, Any], p_ref: re.Pattern, num_sents: int = 0) -> List[Dict[str, Any]]:
        if not self.txt_paths:
            logger.warning("No text paths found to process in predict_probs.")
            return [{"id": "N/A", "txt_path": "N/A", "processing_time_seconds": 0.0, 
                     "status": "Skipped", "error_details": "No TXT files found", "message": "No TXT files"}]

        output_results: List[Dict[str, Any]] = []
        total_files = len(self.txt_paths)
        
        # Pass references to necessary functions from rob_fn to the worker
        # This assumes these functions in rob_fn are modified to accept nlp_instance
        # and are picklable (top-level functions in a module usually are).
        fn_refs = {
            "process_text_fn_ref": self.process_text, # PreRob's method
            "pred_legacy_fn_ref": pred_legacy,
            "pred_bert_fn_ref": pred_bert,
            "extract_sent_han_fn_ref": extract_sent_han
        }

        for i, path_obj in enumerate(self.txt_paths):
            current_id = self.ids[i] if self.ids and i < len(self.ids) else path_obj.stem
            logger.info(f"Preparing to process file {i+1}/{total_files}: {path_obj.name} (ID: {current_id}) via worker.")

            output_queue = multiprocessing.Queue()
            worker_args = (
                str(path_obj), models, p_ref, num_sents, output_queue, current_id,
                fn_refs["process_text_fn_ref"], fn_refs["pred_legacy_fn_ref"],
                fn_refs["pred_bert_fn_ref"], fn_refs["extract_sent_han_fn_ref"],
                ROB_ITEM_CATEGORIES # Pass the list of categories
            )
            
            process = multiprocessing.Process(target=_process_single_file_worker, args=worker_args)
            process.start()
            
            # Wait for the process to complete or timeout
            # FILE_PROCESSING_TIMEOUT_SECONDS should be defined, e.g., in rob_fn.py
            timeout_duration = FILE_PROCESSING_TIMEOUT_SECONDS 
            process.join(timeout=timeout_duration)

            result_score: Optional[Dict[str, Any]] = None
            if process.is_alive():
                logger.warning(f"File {path_obj.name} (ID: {current_id}) processing TIMED OUT after {timeout_duration}s. Terminating worker.")
                process.terminate()
                process.join(timeout=5) # Wait a bit for termination
                if process.is_alive(): # Force kill if still alive
                    logger.error(f"Worker for {path_obj.name} did not terminate gracefully, attempting kill.")
                    process.kill()
                    process.join()

                result_score = {
                    "id": current_id, "txt_path": str(path_obj),
                    "message": f"Processing timed out after {timeout_duration}s.",
                    "status": "Timeout", "error_details": f"Exceeded {timeout_duration}s limit.",
                    "processing_time_seconds": float(timeout_duration),
                }
                for cat in ROB_ITEM_CATEGORIES: result_score[cat] = -4.0 # Timeout error code
                if num_sents > 0: result_score['sentences'] = {cat: ["Timeout"] for cat in ROB_ITEM_CATEGORIES}

            else: # Process finished (either completed or crashed)
                try:
                    result_score = output_queue.get(timeout=5) # Short timeout for queue
                    logger.info(f"File {path_obj.name} (ID: {current_id}) worker finished. Status: {result_score.get('status', 'Unknown')}")
                except queue.Empty:
                    logger.error(f"File {path_obj.name} (ID: {current_id}) worker finished but no result in queue. Assuming crash.")
                    result_score = {
                        "id": current_id, "txt_path": str(path_obj),
                        "message": "Worker crashed or failed to return result.",
                        "status": "Error", "error_details": "Worker process ended without output.",
                        "processing_time_seconds": timeout_duration, # Could be less, but indicates failure
                    }
                    for cat in ROB_ITEM_CATEGORIES: result_score[cat] = -5.0 # Crash error code
                    if num_sents > 0: result_score['sentences'] = {cat: ["Worker Crash"] for cat in ROB_ITEM_CATEGORIES}
                except Exception as e_q:
                    logger.error(f"Error getting result from queue for {path_obj.name}: {e_q}", exc_info=True)
                    # Similar error structure
                    result_score = {
                        "id": current_id, "txt_path": str(path_obj),
                        "message": f"Queue retrieval error: {e_q}",
                        "status": "Error", "error_details": f"Queue error: {e_q}",
                        "processing_time_seconds": timeout_duration,
                    }
                    for cat in ROB_ITEM_CATEGORIES: result_score[cat] = -5.0
                    if num_sents > 0: result_score['sentences'] = {cat: ["Queue Error"] for cat in ROB_ITEM_CATEGORIES}


            if result_score: # Ensure we have a score dictionary
                 output_results.append(result_score)
            else: # Should not happen if logic above is correct
                 logger.error(f"INTERNAL ERROR: No result_score generated for {path_obj.name}. This indicates a flaw in timeout/crash handling.")
                 output_results.append({
                    "id": current_id, "txt_path": str(path_obj), "status": "SystemError", 
                    "error_details": "Failed to obtain result from worker logic.",
                    "processing_time_seconds": 0.0
                 })
        return output_results

if __name__ == "__main__":
    # --- Argument Parsing (remains similar) ---
    parser = argparse.ArgumentParser(
        description='Predict Risk of Bias from text files with hard timeouts and flattened sentences.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', "--input", required=True, type=str, help='Input: dir, TXT, CSV, or comma-separated TXTs.')
    parser.add_argument('-o', '--output', required=True, type=str, help='Absolute path for the output CSV file.')
    parser.add_argument('-s', "--sent", type=int, default=0, help='Number of relevant sentences to extract (0 to disable).')
    parser.add_argument('--model_dir', type=str, default='pth', help='Directory containing model files.')
    args = parser.parse_args()

    input_path_arg = args.input
    output_path = Path(args.output).resolve()
    num_sents_to_extract = args.sent
    model_dir = Path(args.model_dir).resolve()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.critical(f"Could not create output directory {output_path.parent}: {e}", exc_info=True)
        exit(1)

    logger.info("--- RoB Prediction Script Initializing (with Hard Timeouts) ---")
    logger.info(f"Input: {input_path_arg}, Output CSV: {output_path}")
    logger.info(f"Sentences to extract: {num_sents_to_extract}, Model dir: {model_dir}")
    logger.info(f"File processing timeout: {FILE_PROCESSING_TIMEOUT_SECONDS} seconds per file.")
    logger.info(f"Risk categories: {ROB_ITEM_CATEGORIES}")


    p_ref = re.compile(
        r"(\b(Reference|Bibliography|Literatur)\w*\s*(list)?\b\s*?\n|\bREFERENCES\b|\bBIBLIOGRAPHY\b)",
        flags=re.IGNORECASE # Removed DOTALL as process_text handles it line by line or sectionally
    )

    # --- Model Loading (remains similar, ensure models are picklable if passed directly) ---
    logger.info("Loading models...")
    models_loaded: Dict[str, Any] = {}
    model_errors_encountered = False
    
    model_configs = {
        'random': {'type': 'legacy', 'arg': 'awr_13.json', 'pth': 'awr_13.pth.tar', 'fld': 'awr_13.Field'},
        'blind':  {'type': 'legacy', 'arg': 'awb_32.json', 'pth': 'awb_32.pth.tar', 'fld': 'awb_32.Field'},
        'interest':{'type': 'legacy', 'arg': 'cwi_6.json', 'pth': 'cwi_6.pth.tar', 'fld': 'cwi_6.Field'},
        'welfare':{'type': 'bert',   'arg': 'dsc_w0.json', 'pth': 'dsc_w0.pth.tar', 'fld': None},
        'exclusion':{'type': 'legacy', 'arg': 'awe_8.json', 'pth': 'awe_8.pth.tar', 'fld': 'awe_8.Field'},
    }
    sentence_model_configs = {
        's_random': {'type': 'legacy', 'arg': 'hr_4.json', 'pth': 'hr_4.pth.tar', 'fld': 'hr_4.Field'},
        's_blind':  {'type': 'legacy', 'arg': 'hb_5.json', 'pth': 'hb_5.pth.tar', 'fld': 'hb_5.Field'},
        's_interest':{'type': 'legacy', 'arg': 'hi_4.json', 'pth': 'hi_4.pth.tar', 'fld': 'hi_4.Field'},
        's_welfare':{'type': 'legacy', 'arg': 'hw_17.json', 'pth': 'hw_17.pth.tar', 'fld': 'hw_17.Field'},
        's_exclusion':{'type': 'legacy', 'arg': 'he_26.json', 'pth': 'he_26.pth.tar', 'fld': 'he_26.Field'},
    }

    if num_sents_to_extract > 0:
        model_configs.update(sentence_model_configs)
    else:
        logger.info("Sentence extraction disabled. Skipping sentence model loading.")

    for key, config in model_configs.items():
        logger.info(f"  Attempting to load model '{key}'...")
        try:
            arg_p = model_dir / config['arg']
            pth_p = model_dir / config['pth']
            if not arg_p.exists(): raise FileNotFoundError(f"Arg file missing: {arg_p}")
            if not pth_p.exists(): raise FileNotFoundError(f"Pth file missing: {pth_p}")

            if config['type'] == 'legacy':
                if not config['fld']: raise ValueError(f"Field file missing in config for {key}")
                fld_p = model_dir / config['fld']
                if not fld_p.exists(): raise FileNotFoundError(f"Field file missing: {fld_p}")
                models_loaded[key] = load_model_legacy(arg_p, pth_p, fld_p)
            elif config['type'] == 'bert':
                models_loaded[key] = load_model_bert(arg_p, pth_p)
            else:
                logger.error(f"Unknown model type '{config['type']}' for '{key}'. Skipping.")
                model_errors_encountered = True; continue
            logger.info(f"  Successfully loaded model '{key}'.")
        except FileNotFoundError as e_fnf:
            logger.error(f"  File not found for model '{key}': {e_fnf}")
            model_errors_encountered = True; models_loaded[key] = None
        except Exception as e:
            logger.error(f"  ERROR loading model '{key}': {e}", exc_info=True)
            model_errors_encountered = True; models_loaded[key] = None
            
    if not any(m is not None for m in models_loaded.values()):
        logger.critical("No models loaded successfully. Exiting.")
        exit(1)
    if model_errors_encountered:
        logger.warning("Some models failed to load. Results may be incomplete.")

    # --- Initialize PreRob and get file paths ---
    logger.info("Initializing text processing scanner...")
    rober_instance = PreRob(str(input_path_arg))
    rober_instance.get_txt_paths()

    if not rober_instance.txt_paths:
        logger.warning("No .txt files found. Exiting.")
        # Create empty CSV similar to previous version if needed
        exit(0)
    
    # --- Run predictions using multiprocessing ---
    logger.info(f"Starting prediction loop for {len(rober_instance.txt_paths)} files...")
    # Models are passed to the worker. Ensure they are picklable.
    # If models are very large or not picklable, paths/configs should be passed and models loaded in worker.
    # For now, assuming models_loaded (containing model objects) is picklable.
    output_data_list = rober_instance.predict_probs(models_loaded, p_ref, num_sents_to_extract)

    # --- Process and Save Results ---
    if not output_data_list:
        logger.warning("No results generated from prediction process.")
    else:
        logger.info(f"Processing {len(output_data_list)} results for CSV output...")
        
        processed_output_for_df = []
        for res_dict in output_data_list:
            # Flatten sentences
            if num_sents_to_extract > 0 and 'sentences' in res_dict and isinstance(res_dict['sentences'], dict):
                sentences_data = res_dict.pop('sentences') # Remove original sentences dict
                for category, sents_list in sentences_data.items():
                    for i in range(num_sents_to_extract):
                        col_name = f"{category}_sent_{i+1}"
                        if i < len(sents_list):
                            res_dict[col_name] = sents_list[i]
                        else:
                            res_dict[col_name] = "" # Or pd.NA or None
            elif num_sents_to_extract > 0 : # Ensure sentence columns are present even if no sentences key
                 for category in ROB_ITEM_CATEGORIES: # Assuming ROB_ITEM_CATEGORIES is available
                      for i in range(num_sents_to_extract):
                           col_name = f"{category}_sent_{i+1}"
                           res_dict[col_name] = "" 
            processed_output_for_df.append(res_dict)

        output_df = pd.DataFrame(processed_output_for_df)
        
        # Define column order
        cols_ordered = ['id', 'txt_path'] + ROB_ITEM_CATEGORIES
        if num_sents_to_extract > 0:
            for category in ROB_ITEM_CATEGORIES:
                for i in range(num_sents_to_extract):
                    cols_ordered.append(f"{category}_sent_{i+1}")
        cols_ordered.extend(['processing_time_seconds', 'status', 'error_details', 'message'])
        
        # Ensure all expected columns exist in DataFrame, add if missing, then reorder
        for col in cols_ordered:
            if col not in output_df.columns:
                output_df[col] = pd.NA # Or appropriate default
        output_df = output_df[cols_ordered] # Reorder

        try:
            output_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
            logger.info(f"Results saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving output DataFrame to CSV {output_path}: {e}", exc_info=True)
            
    logger.info("--- RoB Prediction Script Finished ---")
