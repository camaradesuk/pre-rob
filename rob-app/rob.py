#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Risk of Bias (RoB) prediction.
This version implements a robust multiprocessing architecture where each
worker process loads its own models to ensure correctness and avoid
pickling issues with CUDA.
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
import queue
import spacy
import sys

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Import functions from rob_fn ---
# These functions will now be used inside the worker.
try:
    from rob_fn import (
        load_model_legacy,
        load_model_bert,
        pred_legacy, 
        pred_bert,   
        extract_sent_han, 
        FILE_PROCESSING_TIMEOUT_SECONDS,
        ROB_ITEM_CATEGORIES 
    )
except ImportError as e:
    logger.critical(f"Critical import error from rob_fn: {e}.", exc_info=True)
    raise

# --- Worker Function ---
def _process_single_file_worker(
    file_path_str: str,
    model_dir_str: str, # Pass model directory path
    model_configs_dict: Dict[str, Any], # Pass configs, not loaded models
    p_ref_compiled: re.Pattern,
    num_sents_val: int,
    output_q: multiprocessing.Queue,
    current_id_val: str,
    process_text_fn_ref,
    pred_legacy_fn_ref,
    pred_bert_fn_ref,
    extract_sent_han_fn_ref,
    rob_item_categories_list: List[str],
    rob_item_descriptions_for_bert: Dict[str, str] # Pass descriptions
):
    """
    Worker function to process a single text file.
    This function is executed in a separate process.
    It loads its own spaCy instance AND all required models from disk.
    """
    worker_log_prefix = f"Worker_{os.getpid()}_for_{current_id_val}"
    print(f"{worker_log_prefix}: Starting.")

    # --- Step 1: Load spaCy instance in the worker ---
    nlp_instance = None
    try:
        nlp_instance = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        if not nlp_instance.has_pipe("sentencizer"):
            sentencizer_pipe = nlp_instance.create_pipe("sentencizer")
            nlp_instance.add_pipe(sentencizer_pipe, first=True)
        print(f"{worker_log_prefix}: spaCy loaded.")
    except Exception as e_nlp:
        print(f"{worker_log_prefix}: FATAL - Failed to load spaCy: {e_nlp}")
        error_score = {"id": current_id_val, "txt_path": file_path_str, "message": "spaCy load error", "status": "Error", "error_details": f"{e_nlp}", "processing_time_seconds": 0.0}
        output_q.put(error_score)
        return

    # --- Step 2: Load all required models in the worker ---
    models_loaded_in_worker: Dict[str, Any] = {}
    model_dir = Path(model_dir_str)
    try:
        for key, config_item in model_configs_dict.items():
            print(f"{worker_log_prefix}: Loading model '{key}'...")
            arg_p = model_dir / config_item['arg']
            pth_p = model_dir / config_item['pth']
            
            if config_item['type'] == 'legacy':
                fld_p = model_dir / config_item['fld']
                models_loaded_in_worker[key] = load_model_legacy(arg_p, pth_p, fld_p)
            elif config_item['type'] == 'bert':
                rob_item_key = config_item.get('rob_item_key')
                rob_sent_desc = rob_item_descriptions_for_bert.get(rob_item_key)
                if not rob_sent_desc:
                    raise ValueError(f"Description for BERT item '{rob_item_key}' not found.")
                models_loaded_in_worker[key] = load_model_bert(arg_p, pth_p, rob_sent_desc)
        print(f"{worker_log_prefix}: All models loaded successfully.")
    except Exception as e_model_load:
        print(f"{worker_log_prefix}: FATAL - Failed to load models: {e_model_load}")
        error_score = {"id": current_id_val, "txt_path": file_path_str, "message": "Model load error in worker", "status": "Error", "error_details": f"{e_model_load}", "processing_time_seconds": 0.0}
        output_q.put(error_score)
        return
        
    # --- Step 3: Perform predictions (existing logic) ---
    start_time = time.time()
    file_status = "OK"
    error_msg = ""
    score: Dict[str, Any] = {"id": current_id_val, "txt_path": file_path_str, "message": "Processing", "status": "Processing"}
    if num_sents_val > 0: score['sentences'] = {cat: [] for cat in rob_item_categories_list}
    
    try:
        with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as fin: raw_text = fin.read()
        if not raw_text.strip():
            score["message"], file_status, error_msg = "File is empty", "Skipped", "File content empty"
        else:
            processed_text = process_text_fn_ref(raw_text, p_ref_compiled) 
            if not processed_text.strip():
                score["message"], file_status, error_msg = "Text empty after processing", "Skipped", "Text empty"
            else:
                # Prediction for scores
                for key in rob_item_categories_list:
                    model_data = models_loaded_in_worker.get(key)
                    if not model_data: score[key] = -2.0; continue
                    try:
                        if key == 'welfare':
                            model, tokenizer, sent_model, rob_sent_desc = model_data
                            score[key] = pred_bert_fn_ref(processed_text, model, tokenizer, sent_model, rob_sent_desc, nlp_instance, max_n_sent=30)
                        else:
                            model, model_args, vocab_stoi, pad_idx, unk_idx = model_data
                            score[key] = pred_legacy_fn_ref(processed_text, model, model_args, vocab_stoi, pad_idx, unk_idx, nlp_instance)
                    except Exception as e:
                        print(f"{worker_log_prefix}: Error predicting {key}: {e}"); score[key] = -1.0; file_status = "Error"; error_msg += f"Pred({key});"
                
                # Prediction for sentences
                if num_sents_val > 0:
                    for key in rob_item_categories_list:
                        s_key = 's_' + key
                        model_data = models_loaded_in_worker.get(s_key)
                        if not model_data: score['sentences'][key] = ["N/C"]; continue
                        try:
                             s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx = model_data
                             score['sentences'][key] = extract_sent_han_fn_ref(processed_text, s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx, num_sents_val, nlp_instance)
                        except Exception as e:
                            print(f"{worker_log_prefix}: Error extracting for {key}: {e}"); score['sentences'][key] = ["ExtractErr"]; file_status = "Error"; error_msg += f"SentExt({key});"
                
                if file_status == "OK": score["message"] = "Success"
    except Exception as e:
        score["message"], file_status, error_msg = "Worker error", "Error", f"{e}"
        print(f"{worker_log_prefix}: General error: {e}")

    score["processing_time_seconds"] = round(time.time() - start_time, 3)
    score["status"], score["error_details"] = file_status, error_msg.strip()
    print(f"{worker_log_prefix}: Finished. Status: {file_status}, Time: {score['processing_time_seconds']}s")
    output_q.put(score)

# --- PreRob Class (largely unchanged, just calls the new worker setup) ---
class PreRob():
    def __init__(self, txt_info: str):
        self.txt_info = Path(txt_info)
        self.txt_paths: List[Path] = []
        self.ids: List[str] = []
        logger.info(f"PreRob initialized with input: {txt_info}")
    def get_txt_paths(self) -> None: 
        # This method is unchanged
        txt_info_path = self.txt_info
        self.txt_paths = []
        self.ids = []
        if not txt_info_path.exists() and isinstance(self.txt_info, str) and ".txt," not in str(self.txt_info): logger.error(f"Input path does not exist: {txt_info_path}"); return
        if txt_info_path.is_dir():
            self.txt_paths = sorted(list(txt_info_path.rglob("*.txt"))); self.ids = [p.stem + f"_{i}" for i, p in enumerate(self.txt_paths)]; logger.info(f"Found {len(self.txt_paths)} .txt files in directory.")
        elif txt_info_path.is_file() and txt_info_path.suffix == ".txt":
            self.txt_paths.append(txt_info_path.resolve()); self.ids.append(Path(txt_info_path).stem)
        elif txt_info_path.is_file() and txt_info_path.suffix == ".csv":
            try:
                path_df = pd.read_csv(txt_info_path, sep=',')
                if 'path' not in path_df.columns or 'id' not in path_df.columns: raise ValueError("CSV must contain 'path' and 'id' columns.")
                base_dir = txt_info_path.parent
                for _, row in path_df.iterrows():
                    txt_path = Path(str(row['path']).strip())
                    if not txt_path.is_absolute(): txt_path = base_dir / txt_path
                    if txt_path.is_file(): self.txt_paths.append(txt_path.resolve()); self.ids.append(str(row['id']))
                    else: logger.warning(f"File not found in CSV row (id: {row['id']}): {txt_path}")
            except Exception as e: logger.error(f"Error reading CSV {txt_info_path}: {e}", exc_info=True)

    def process_text(self, text: str, p_ref: re.Pattern) -> str: 
        # This method is unchanged
        try:
            processed_text = re.sub(r".*?(Introduction|INTRODUCTION)\s*\n+", " ", text, count=1, flags=re.DOTALL | re.IGNORECASE)
            if processed_text == text: processed_text = text
            s = p_ref.search(processed_text)
            if s and s.start() > 0: processed_text = processed_text[:s.start()]
            processed_text = re.sub(r"\s+[\[][^a-zA-Z]+[\]]", "", processed_text)
            processed_text = re.sub(r"https?:/\/\S+", " ", processed_text)
            processed_text = processed_text.encode("ascii", errors="ignore").decode()
            return re.sub(r'\s+', " ", processed_text).strip()
        except Exception as e:
            print(f"Error during text processing: {e}")
            return ""

    def predict_probs(self, model_dir: Path, model_configs: Dict[str, Any], rob_item_descriptions: Dict[str, str], p_ref: re.Pattern, num_sents: int = 0) -> List[Dict[str, Any]]: 
        if not self.txt_paths: return []
        output_results: List[Dict[str, Any]] = []
        fn_refs = {"process_text_fn_ref": self.process_text, "pred_legacy_fn_ref": pred_legacy, "pred_bert_fn_ref": pred_bert, "extract_sent_han_fn_ref": extract_sent_han}
        
        for i, path_obj in enumerate(self.txt_paths):
            current_id = self.ids[i] if self.ids and i < len(self.ids) else path_obj.stem
            logger.info(f"Preparing file {i+1}/{len(self.txt_paths)}: {path_obj.name} (ID: {current_id})")
            output_queue = multiprocessing.Queue()
            
            # Pass model_configs dict, not loaded models
            worker_args = (
                str(path_obj), str(model_dir), model_configs, p_ref, num_sents, output_queue, current_id, 
                fn_refs["process_text_fn_ref"], fn_refs["pred_legacy_fn_ref"], 
                fn_refs["pred_bert_fn_ref"], fn_refs["extract_sent_han_fn_ref"], 
                ROB_ITEM_CATEGORIES, rob_item_descriptions
            )
            
            process = multiprocessing.Process(target=_process_single_file_worker, args=worker_args)
            process.start()
            process.join(timeout=FILE_PROCESSING_TIMEOUT_SECONDS)
            
            result_score: Optional[Dict[str, Any]] = None
            if process.is_alive():
                logger.warning(f"File {path_obj.name} TIMED OUT. Terminating worker.")
                process.terminate(); process.join(5); process.kill()
                result_score = {"id": current_id, "txt_path": str(path_obj), "message": "Timeout", "status": "Timeout", "error_details": f"Exceeded {FILE_PROCESSING_TIMEOUT_SECONDS}s limit.", "processing_time_seconds": float(FILE_PROCESSING_TIMEOUT_SECONDS)}
            else: 
                try: result_score = output_queue.get(timeout=10)
                except queue.Empty:
                    logger.error(f"Worker for {path_obj.name} finished (exit {process.exitcode}) but queue is empty. Assuming crash.")
                    result_score = {"id": current_id, "txt_path": str(path_obj), "message": "Worker Crash", "status": "Error", "error_details": f"Worker process ended (exit {process.exitcode}) without output.", "processing_time_seconds": 0}
            
            if result_score: output_results.append(result_score)
        
        return output_results

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
            logger.info("Multiprocessing start method set to 'spawn' for CUDA compatibility.")
        except RuntimeError as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

    parser = argparse.ArgumentParser(description='RoB Prediction with Hard Timeouts.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', "--input", required=True, type=str, help='Input path.')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output CSV path.')
    parser.add_argument('-s', "--sent", type=int, default=0, help='Number of sentences to extract.')
    parser.add_argument('--model_dir', type=str, default='pth', help='Model directory.')
    args = parser.parse_args()

    # --- Setup ---
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("--- RoB Prediction Script Initializing ---")
    
    # --- Define Model Configurations (NOT loading them) ---
    ROB_ITEM_DESCRIPTIONS_FOR_BERT = {
        'welfare': 'Research investigators complied with animal welfare regulations'
    }

    model_configs = {
        'random': {'type': 'legacy', 'arg': 'awr_13.json', 'pth': 'awr_13.pth.tar', 'fld': 'awr_13.Field'},
        'blind':  {'type': 'legacy', 'arg': 'awb_32.json', 'pth': 'awb_32.pth.tar', 'fld': 'awb_32.Field'},
        'interest':{'type': 'legacy', 'arg': 'cwi_6.json', 'pth': 'cwi_6.pth.tar', 'fld': 'cwi_6.Field'},
        'welfare':{'type': 'bert',   'arg': 'dsc_w0.json', 'pth': 'dsc_w0.pth.tar', 'rob_item_key': 'welfare'},
        'exclusion':{'type': 'legacy', 'arg': 'awe_8.json', 'pth': 'awe_8.pth.tar', 'fld': 'awe_8.Field'},
    }
    if args.sent > 0:
        model_configs.update({
            's_random': {'type': 'legacy', 'arg': 'hr_4.json', 'pth': 'hr_4.pth.tar', 'fld': 'hr_4.Field'},
            's_blind':  {'type': 'legacy', 'arg': 'hb_5.json', 'pth': 'hb_5.pth.tar', 'fld': 'hb_5.Field'},
            's_interest':{'type': 'legacy', 'arg': 'hi_4.json', 'pth': 'hi_4.pth.tar', 'fld': 'hi_4.Field'},
            's_welfare':{'type': 'legacy', 'arg': 'hw_17.json', 'pth': 'hw_17.pth.tar', 'fld': 'hw_17.Field'},
            's_exclusion':{'type': 'legacy', 'arg': 'he_26.json', 'pth': 'he_26.pth.tar', 'fld': 'he_26.Field'},
        })
    logger.info(f"Prepared {len(model_configs)} model configurations to be used by workers.")

    # --- Prediction and Saving Loop ---
    rober_instance = PreRob(str(args.input))
    rober_instance.get_txt_paths()
    if not rober_instance.txt_paths:
        logger.warning("No .txt files found. Exiting.")
        exit(0)
    
    p_ref = re.compile(r"(\b(Reference|Bibliography|Literatur)\w*\s*(list)?\b\s*?\n|\bREFERENCES\b|\bBIBLIOGRAPHY\b)", flags=re.IGNORECASE)
    
    # Pass model_configs dict to predict_probs, not loaded models
    output_data_list = rober_instance.predict_probs(
        model_dir, 
        model_configs, 
        ROB_ITEM_DESCRIPTIONS_FOR_BERT, 
        p_ref, 
        args.sent
    )

    if output_data_list:
        logger.info(f"Processing {len(output_data_list)} results for CSV output...")
        processed_output_for_df = []
        for res_dict in output_data_list:
            if args.sent > 0:
                sentences_data = res_dict.pop('sentences', {})
                for cat in ROB_ITEM_CATEGORIES:
                    sents_list = sentences_data.get(cat, []) if isinstance(sentences_data, dict) else [str(sentences_data)]
                    if not isinstance(sents_list, list): sents_list = [str(sents_list)]
                    for i in range(args.sent):
                        res_dict[f"{cat}_sent_{i+1}"] = sents_list[i] if i < len(sents_list) else ""
            processed_output_for_df.append(res_dict)
        
        output_df = pd.DataFrame(processed_output_for_df)
        cols_ordered = ['id', 'txt_path'] + ROB_ITEM_CATEGORIES
        if args.sent > 0:
            for cat in ROB_ITEM_CATEGORIES:
                for i in range(args.sent): cols_ordered.append(f"{cat}_sent_{i+1}")
        cols_ordered.extend(['processing_time_seconds', 'status', 'error_details', 'message'])
        for col in cols_ordered:
            if col not in output_df.columns: output_df[col] = "" 
        output_df = output_df.reindex(columns=cols_ordered)
        output_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
        logger.info(f"Results saved successfully to {output_path}")

    logger.info("--- RoB Prediction Script Finished ---")

