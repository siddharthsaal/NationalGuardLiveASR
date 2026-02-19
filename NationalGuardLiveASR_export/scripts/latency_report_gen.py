import os
import time
import json
import csv
import subprocess
import uuid
import numpy as np
import scipy.io.wavfile as wav
import av
from datasets import load_dataset
from whisper_live.client import TranscriptionClient

# Configuration
MODEL = "small"
DATASET_NAME = "MohamedRashad/fleurs-ar-eg"
NUM_SAMPLES = 5
SERVER_PORT = 9096
PYTHON_EXE = "python3.11"
OUTPUT_CSV = "latency_details_small.csv"

class LatencyTrackingClient(TranscriptionClient):
    def __init__(self, *args, **kwargs):
        kwargs["transcription_callback"] = self.latency_callback
        super().__init__(*args, **kwargs)
        self.chunk_times = []
        self.total_samples_sent = 0
        self.sample_rate = 16000
        self.latency_results = []
        self.current_sample_id = 0
        self.current_ground_truth = ""
        self._last_logged_end = -1
        # Increase timeout
        self.client.disconnect_if_no_response_for = 45

    def multicast_packet(self, packet, unconditional=False):
        if packet == b"END_OF_AUDIO":
            super().multicast_packet(packet, unconditional)
            return

        wall_clock = time.time()
        num_bytes = len(packet)
        # S16LE is 2 bytes per sample
        num_samples = num_bytes // 2
        self.total_samples_sent += num_samples
        offset_seconds = self.total_samples_sent / self.sample_rate
        self.chunk_times.append((offset_seconds, wall_clock))
        
        super().multicast_packet(packet, unconditional)

    def get_wall_clock_for_offset(self, offset):
        if not self.chunk_times:
            return time.time()
        for stream_offset, wall_clock in self.chunk_times:
            if stream_offset >= offset:
                return wall_clock
        return self.chunk_times[-1][1]

    def latency_callback(self, text, segments):
        arrival_time = time.time()
        if not segments:
            return
            
        for seg in segments:
            end_offset = float(seg["end"])
            # Log segment even if not "completed" if it's new text
            if end_offset > self._last_logged_end:
                sent_wall_clock = self.get_wall_clock_for_offset(end_offset)
                latency = arrival_time - sent_wall_clock
                
                # Check if this looks like a final/stable segment
                # faster_whisper sometimes sends many partials
                is_completed = seg.get("completed", False)
                
                if is_completed or (end_offset - self._last_logged_end > 0.5):
                    self.latency_results.append({
                        "Sample ID": self.current_sample_id,
                        "Ground Truth": self.current_ground_truth,
                        "Transcription": seg["text"].strip(),
                        "Start (s)": seg["start"],
                        "End (s)": seg["end"],
                        "Latency (s)": f"{latency:.4f}",
                        "Completed": is_completed
                    })
                    status = "[FINAL]" if is_completed else "[PARTIAL]"
                    print(f"    {status} Latency: {latency:.4f}s | Text: {seg['text'].strip()}")
                    self._last_logged_end = end_offset

def setup_eval_dir():
    if not os.path.exists("eval_audio"):
        os.makedirs("eval_audio")

def download_samples():
    print(f"--- Downloading {NUM_SAMPLES} Arabic samples ---")
    ds = load_dataset(DATASET_NAME, split="test", streaming=True)
    samples = []
    count = 0
    for item in ds:
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        duration = len(audio) / sr
        if 5.0 <= duration <= 15.0:
            file_path = f"eval_audio/latency_sample_{count}.wav"
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

def run_latency_benchmark():
    setup_eval_dir()
    samples = download_samples()
    
    mediamtx = None
    server_proc = None
    
    try:
        print("Starting MediaMTX...")
        mediamtx = subprocess.Popen(["./mediamtx"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Starting WhisperLive Server (Small Model) on port {SERVER_PORT}...")
        server_log = open("server_latency_small.log", "w")
        server_proc = subprocess.Popen([
            PYTHON_EXE, "-m", "run_server",
            "--port", str(SERVER_PORT),
            "--backend", "faster_whisper"
        ], stdout=server_log, stderr=subprocess.STDOUT)
        
        print("Waiting 30 seconds for server to be ready and model to load...")
        time.sleep(30)
        
        all_latency_data = []
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Processing: {sample['audio_path']} (Duration: {sample['duration']:.2f}s)")
            
            stream_name = f"latency_{uuid.uuid4().hex[:8]}"
            rtsp_url = f"rtsp://127.0.0.1:8554/{stream_name}"
            
            ffmpeg = subprocess.Popen([
                "ffmpeg", "-re", "-i", sample["audio_path"],
                "-c:a", "aac", "-b:a", "32k", "-ar", "16000", "-ac", "1",
                "-f", "rtsp", rtsp_url
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            client = LatencyTrackingClient(
                "127.0.0.1", 
                SERVER_PORT, 
                lang="ar", 
                model=MODEL, 
                use_vad=False,
                log_transcription=False
            )
            client.current_sample_id = i
            client.current_ground_truth = sample["ground_truth"]
            
            client(rtsp_url=rtsp_url)
            
            ffmpeg.wait()
            all_latency_data.extend(client.latency_results)
            client.close_all_clients()
            print(f"  Finished sample {i+1}. Captured {len(client.latency_results)} events.")

        if all_latency_data:
            keys = all_latency_data[0].keys()
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_latency_data)
            print(f"\nDetailed latency report saved to {OUTPUT_CSV}")
            
            latencies = [float(r["Latency (s)"]) for r in all_latency_data]
            print("\nSummary Statistics:")
            print(f"  Avg Latency: {sum(latencies)/len(latencies):.4f}s")
        else:
            print("\nNo latency data collected.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if server_proc: server_proc.terminate()
        if mediamtx: mediamtx.terminate()

if __name__ == "__main__":
    run_latency_benchmark()
