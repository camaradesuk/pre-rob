#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Risk of Bias (RoB) prediction.
Processes text files, predicts RoB scores using various models,
and optionally extracts relevant sentences.

Updated for Python 3.9, PyTorch 2.6, Transformers 4.x, SentenceTransformers 2.x
"""

import os
import re
import argparse
import warnings
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- Import updated functions from rob_fn ---
try:
    from rob_fn import (
        load_model_legacy,
        load_model_bert,
        pred_legacy,
        pred_bert,
        extract_sent_han,
        nlp
    )
except ImportError as e:
    print(f"Import error in rob_fn: {e}")
    raise

class PreRob():
    def __init__(self, txt_info: str):
        self.txt_info = Path(txt_info)
        self.txt_paths: List[Path] = []
        self.ids: List[str] = []

    def get_txt_paths(self):
        txt_info_path = self.txt_info
        self.txt_paths = []
        self.ids = []
        if txt_info_path.is_dir():
            print(f"Scanning directory: {txt_info_path}")
            self.txt_paths = sorted(list(txt_info_path.rglob("*.txt")))
            self.ids = [str(i + 1) for i in range(len(self.txt_paths))]
            print(f"Found {len(self.txt_paths)} .txt files.")
        elif txt_info_path.is_file() and txt_info_path.suffix == ".txt":
            print(f"Processing single file: {txt_info_path}")
            self.txt_paths.append(txt_info_path)
            self.ids.append("1")
        elif txt_info_path.is_file() and txt_info_path.suffix == ".csv":
            print(f"Reading paths from CSV: {txt_info_path}")
            try:
                path_df = pd.read_csv(txt_info_path, sep=',')
                if 'path' not in path_df.columns or 'id' not in path_df.columns:
                    raise ValueError("CSV must contain 'path' and 'id' columns.")
                base_dir = txt_info_path.parent
                for i, row in path_df.iterrows():
                    relative_path = Path(row['path'])
                    txt_path = relative_path if relative_path.is_absolute() else base_dir / relative_path
                    if txt_path.is_file() and txt_path.suffix == ".txt":
                        self.txt_paths.append(txt_path)
                        self.ids.append(str(row['id']))
                    else:
                        warnings.warn(f"File not found or not a .txt file in CSV row {i}: {txt_path}", UserWarning)
                print(f"Found {len(self.txt_paths)} valid .txt paths in CSV.")
            except Exception as e:
                print(f"Error reading or processing CSV file {txt_info_path}: {e}")
        elif isinstance(self.txt_info, str) and ".txt," in self.txt_info:
            print("Processing comma-separated paths string (Note: less robust than CSV).")
            paths_str = self.txt_info.split(".txt,")
            for i, p_str in enumerate(paths_str):
                path_str = p_str + ".txt" if i < len(paths_str) - 1 else p_str
                txt_path = Path(path_str.strip())
                if txt_path.is_file():
                    self.txt_paths.append(txt_path)
                else:
                    warnings.warn(f"File not found in comma-separated list: {txt_path}", UserWarning)
            self.ids = [str(i + 1) for i in range(len(self.txt_paths))]
        if not self.txt_paths:
            print(f"Warning: No .txt files found based on input: {self.txt_info}")

    def process_text(self, text: str, p_ref: re.Pattern) -> str:
        text = re.sub(r".*?(Introduction|INTRODUCTION)\s{0,}\n{1,}", " ", text, count=1, flags=re.DOTALL)
        s = p_ref.search(text)
        if s:
            text = s[0]
        text = re.sub(r"\s+[\[][^a-zA-Z]+[\]]", "", text)
        text = re.sub(r"https?:/\/\S+", " ", text)
        text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)
        text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", text)
        text = text.encode("ascii", errors="ignore").decode()
        text = re.sub(r'\s+', " ", text)
        text = re.sub(r'^[\s]', "", text)
        text = re.sub(r'[\s]$', "", text)
        return text

    def predict_probs(self, models: Dict[str, Any], p_ref: re.Pattern, num_sents: int = 0) -> List[Dict[str, Any]]:
        if not self.txt_paths:
            print("No text paths found to process.")
            return [{"message": "No TXT files found based on input.", "id": "N/A"}]
        output_results = []
        file_count = 0
        total_files = len(self.txt_paths)
        for i, path in enumerate(self.txt_paths):
            file_count += 1
            print(f"Processing file {file_count}/{total_files}: {path.name}...")
            current_id = self.ids[i] if self.ids and i < len(self.ids) else str(i + 1)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
                    raw_text = fin.read()
                processed_text = self.process_text(raw_text, p_ref)
                if not processed_text.strip():
                    warnings.warn(f"Text processing resulted in empty content for file: {path.name}", UserWarning)
                    score = {"id": current_id, "txt_path": str(path),
                             "random": 999, "blind": 999, "interest": 999, "welfare": 999, "exclusion": 999,
                             "message": "Text empty after processing"}
                    if num_sents > 0:
                        score['sentences'] = {"random": [], "blind": [], "interest": [], "welfare": [], "exclusion": []}
                    output_results.append(score)
                    continue
                score = {"id": current_id, "txt_path": str(path)}
                for key in ['random', 'blind', 'interest', 'exclusion']:
                    if key in models:
                        model, args, vocab_stoi, pad_idx, unk_idx = models[key]
                        try:
                            score[key] = pred_legacy(processed_text, model, args, vocab_stoi, pad_idx, unk_idx)
                        except Exception as e:
                            print(f"Error predicting '{key}' for {path.name}: {e}")
                            score[key] = -1
                    else:
                        score[key] = -2
                if 'welfare' in models:
                    model, tokenizer, sent_model, rob_sent = models['welfare']
                    try:
                        score['welfare'] = pred_bert(processed_text, model, tokenizer, sent_model, rob_sent, max_n_sent=30)
                    except Exception as e:
                        print(f"Error predicting 'welfare' (BERT) for {path.name}: {e}")
                        score['welfare'] = -1
                else:
                    score['welfare'] = -2
                if num_sents > 0:
                    score['sentences'] = {}
                    sent_keys = ['random', 'blind', 'interest', 'welfare', 'exclusion']
                    model_prefix = 's_'
                    for key in sent_keys:
                        s_key = model_prefix + key
                        if s_key in models:
                            s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx = models[s_key]
                            try:
                                score['sentences'][key] = extract_sent_han(
                                    processed_text, s_model, s_args, s_vocab_stoi, s_pad_idx, s_unk_idx, num_sents
                                )
                            except Exception as e:
                                print(f"Error extracting sentences for '{key}' for {path.name}: {e}")
                                score['sentences'][key] = [f"Error: {e}"]
                        else:
                            score['sentences'][key] = ["Model not loaded"]
                output_results.append(score)
            except FileNotFoundError:
                print(f"Error: File not found during processing loop: {path}")
                output_results.append({"id": current_id, "txt_path": str(path), "message": "File not found"})
            except Exception as e:
                print(f"Error processing file {path.name}: {e}")
                output_results.append({"id": current_id, "txt_path": str(path), "message": f"Processing error: {e}"})
        return output_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Risk of Bias from text files.')
    parser.add_argument('-i', "--input", required=True, type=str,
                        help='Path to input: directory of TXTs, single TXT file, CSV file (with "path" and "id" columns), or comma-separated TXT paths.')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Absolute path for the output CSV file.')
    parser.add_argument('-s', "--sent", type=int, default=0,
                        help='Number of relevant sentences to extract for each category (default: 0).')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    num_sents = int(args.sent)
    model_dir = Path('pth')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("--- RoB Prediction Script ---")
    print(f"Input specified: {args.input}")
    print(f"Output CSV: {output_path}")
    print(f"Sentences to extract: {num_sents}")
    print(f"Model directory: {model_dir}")

    p_ref = re.compile(
        r"(.*Reference\\s{0,}\\n)|(.*References\\s{0,}\\n)|(.*Reference list\\s{0,}\\n)|"
        r"(.*REFERENCE\\s{0,}\\n)|(.*REFERENCES\\s{0,}\\n)|(.*REFERENCE LIST\\s{0,}\\n)",
        flags=re.DOTALL
    )

    print("Loading models...")
    models_loaded: Dict[str, Any] = {}
    model_errors = False
    model_configs = {
        'random': {'type': 'legacy', 'arg': 'awr_13.json', 'pth': 'awr_13.pth.tar', 'fld': 'awr_13.Field'},
        'blind':  {'type': 'legacy', 'arg': 'awb_32.json', 'pth': 'awb_32.pth.tar', 'fld': 'awb_32.Field'},
        'interest':{'type': 'legacy', 'arg': 'cwi_6.json', 'pth': 'cwi_6.pth.tar', 'fld': 'cwi_6.Field'},
        'welfare':{'type': 'bert',   'arg': 'dsc_w0.json', 'pth': 'dsc_w0.pth.tar', 'fld': None},
        'exclusion':{'type': 'legacy', 'arg': 'awe_8.json', 'pth': 'awe_8.pth.tar', 'fld': 'awe_8.Field'},
        's_random': {'type': 'legacy', 'arg': 'hr_4.json', 'pth': 'hr_4.pth.tar', 'fld': 'hr_4.Field'},
        's_blind':  {'type': 'legacy', 'arg': 'hb_5.json', 'pth': 'hb_5.pth.tar', 'fld': 'hb_5.Field'},
        's_interest':{'type': 'legacy', 'arg': 'hi_4.json', 'pth': 'hi_4.pth.tar', 'fld': 'hi_4.Field'},
        's_welfare':{'type': 'legacy', 'arg': 'hw_17.json', 'pth': 'hw_17.pth.tar', 'fld': 'hw_17.Field'},
        's_exclusion':{'type': 'legacy', 'arg': 'he_26.json', 'pth': 'he_26.pth.tar', 'fld': 'he_26.Field'},
    }
    for key, config in model_configs.items():
        if key.startswith('s_') and num_sents <= 0:
            continue
        print(f"  Loading '{key}'...")
        try:
            arg_p = model_dir / config['arg']
            pth_p = model_dir / config['pth']
            fld_p = model_dir / config['fld'] if config['fld'] else None
            if not arg_p.exists(): raise FileNotFoundError(f"Arg file not found: {arg_p}")
            if not pth_p.exists(): raise FileNotFoundError(f"Pth file not found: {pth_p}")
            if config['type'] == 'legacy' and (not fld_p or not fld_p.exists()):
                raise FileNotFoundError(f"Field file not found: {fld_p}")
            if config['type'] == 'legacy':
                models_loaded[key] = load_model_legacy(arg_p, pth_p, fld_p)
            elif config['type'] == 'bert':
                models_loaded[key] = load_model_bert(arg_p, pth_p)
            print(f"  Successfully loaded '{key}'.")
        except Exception as e:
            print(f"  ERROR loading model '{key}': {e}")
            print(f"  Skipping predictions/extractions for '{key}'.")
            model_errors = True
    if not models_loaded:
        print("Critical Error: No models were loaded successfully. Exiting.")
        exit(1)
    if model_errors:
        print("Warning: Some models failed to load. Results may be incomplete.")
    print("Initializing text processing...")
    rober = PreRob(str(input_path))
    rober.get_txt_paths()
    print("Starting prediction loop...")
    output_data = rober.predict_probs(models_loaded, p_ref, num_sents)
    if not output_data:
        print("No results generated.")
    else:
        print(f"Saving results to {output_path}...")
        try:
            output_df = pd.DataFrame(output_data)
            cols = ['id', 'txt_path', 'random', 'blind', 'interest', 'welfare', 'exclusion']
            if num_sents > 0:
                cols.append('sentences')
            cols.append('message')
            output_df = output_df.reindex(columns=[c for c in cols if c in output_df.columns])
            output_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving output DataFrame to CSV: {e}")
    print("--- Script Finished ---")
