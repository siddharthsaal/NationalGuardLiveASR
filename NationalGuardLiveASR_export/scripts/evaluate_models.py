import os
import time
import json
import csv
import subprocess
import signal
import numpy as np
import scipy.io.wavfile as wav
from datasets import load_dataset
from jiwer import wer
from whisper_live.client import TranscriptionClient

# Configuration
MODELS_TO_TEST = ["base", "small", "turbo"]
DATASET_NAME = "MohamedRashad/fleurs-ar-eg"
DATASET_CONFIG = None  # Parquet datasets often don't need config
NUM_SAMPLES = 10  # Increased for final verify
SERVER_PORT = 9091
PYTHON_EXE = "python3.11"
OUTPUT_DIR = "eval_results"

def setup_eval_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists("eval_audio"):
        os.makedirs("eval_audio")

def download_samples():
    print(f"--- Downloading {NUM_SAMPLES} Arabic samples from {DATASET_NAME} ---")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=True)
    
    samples = []
    count = 0
    for item in ds:
        # Check duration
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        duration = len(audio) / sr
        
        if 5.0 <= duration <= 30.0:
            file_path = f"eval_audio/sample_{count}.wav"
            # Normalize and save as WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(file_path, sr, audio_int16)
            
            samples.append({
                "audio_path": file_path,
                "ground_truth": item["transcription"],
                "duration": duration
            })
            count += 1
            if count >= NUM_SAMPLES:
                break
    return samples

def run_benchmark(model_name, samples):
    print(f"\n=== Benchmarking Model: {model_name} ===")
    results = []
    
    server_proc = None
    try:
        # Start server once per model
        print(f"Starting server for {model_name}...")
        log_file_path = f"server_{model_name}.log"
        server_log_file = open(log_file_path, "w")
        print(f"Logging server output to {log_file_path}")
        server_proc = subprocess.Popen([
            PYTHON_EXE, "-m", "run_server",
            "--port", str(SERVER_PORT),
            "--backend", "faster_whisper"
        ], stdout=server_log_file, stderr=subprocess.STDOUT)
        
        # Wait for server to be ready
        time.sleep(10)
        
        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] Transcribing: {sample['audio_path']}")
            
            srt_path = f"eval_results/sample_{i}.srt"
            client = TranscriptionClient(
                "127.0.0.1", 
                SERVER_PORT, 
                lang="ar", 
                model=model_name, 
                use_vad=False,
                output_transcription_path=srt_path
            )
            
            start_time = time.time()
            client(sample["audio_path"])
            end_time = time.time()
            
            # Read from generated SRT
            transcription = ""
            if os.path.exists(srt_path):
                with open(srt_path, "r") as f:
                    lines = f.readlines()
                    # Basic SRT parsing to extract text
                    transcription = " ".join([l.strip() for l in lines if not l.strip().isdigit() and "-->" not in l and l.strip()])

            latency = end_time - start_time
            rtf = latency / sample["duration"]
            
            error = wer(sample["ground_truth"], transcription)
            
            print(f"    GT: {sample['ground_truth']}")
            print(f"    HYP: {transcription}")
            print(f"    WER: {error:.4f} | RTF: {rtf:.2f}")
            
            results.append({
                "sample_id": i,
                "ground_truth": sample["ground_truth"],
                "transcription": transcription,
                "wer": error,
                "rtf": rtf,
                "latency": latency
            })
            
            client.close_all_clients()
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        if server_proc:
            print(f"Stopping server...")
            server_proc.terminate()
            server_proc.wait()
            
    return results

def generate_report(all_results):
    print("\n--- Final Evaluation Report ---")
    
    # 1. Summary Markdown Report
    report_content = "# WhisperLive Arabic Evaluation Report\n\n"
    report_content += "## Overall Summary\n\n"
    report_content += "| Model | Avg WER | Avg RTF | Total Latency (s) |\n"
    report_content += "|-------|---------|---------|-------------------|\n"
    
    detailed_results = []
    
    for model, data in all_results.items():
        avg_wer = sum(r["wer"] for r in data) / len(data)
        avg_rtf = sum(r["rtf"] for r in data) / len(data)
        total_lat = sum(r["latency"] for r in data)
        
        print(f"Model: {model:8} | WER: {avg_wer:.4f} | RTF: {avg_rtf:.2f}")
        report_content += f"| {model} | {avg_wer:.4f} | {avg_rtf:.2f} | {total_lat:.2f} |\n"
        
        for r in data:
            detailed_results.append({
                "Model": model,
                "Sample ID": r["sample_id"],
                "Ground Truth": r["ground_truth"],
                "Transcription": r["transcription"],
                "WER": f"{r['wer']:.4f}",
                "Latency (s)": f"{r['latency']:.2f}",
                "RTF": f"{r['rtf']:.2f}"
            })
    
    with open("evaluation_report.md", "w") as f:
        f.write(report_content)
    print("\nSummary report saved to evaluation_report.md")
    
    # 2. Detailed CSV Report
    csv_file = "evaluation_results_detailed.csv"
    keys = ["Model", "Sample ID", "Ground Truth", "Transcription", "WER", "Latency (s)", "RTF"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(detailed_results)
    print(f"Detailed results saved to {csv_file}")

if __name__ == "__main__":
    setup_eval_dir()
    samples = download_samples()
    
    if not samples:
        print("No samples found matching criteria.")
        exit(1)
        
    all_results = {}
    for model in MODELS_TO_TEST:
        all_results[model] = run_benchmark(model, samples)
    
    generate_report(all_results)
