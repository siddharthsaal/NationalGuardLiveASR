import subprocess
import time
import os
import uuid
import signal
from whisper_live.client import TranscriptionClient

# CONFIGURATION
WAV_FILE = "Ar_f1.wav"  # Input file
MODEL = "base"          # Whisper model
SERVER_PORT = 9090      # Port for the transcription server
PYTHON_EXE = "python3.11" # The specific python version with dependencies installed

def run_transcription():
    mediamtx = None
    server = None
    ffmpeg = None
    client = None

    try:
        print(f"--- Starting MediaMTX (RTSP Server) ---")
        mediamtx = subprocess.Popen(["./mediamtx"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"--- Starting WhisperLive Server on port {SERVER_PORT} ---")
        server = subprocess.Popen([
            PYTHON_EXE, "-m", "run_server",
            "--port", str(SERVER_PORT)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Give the server and mediamtx time to initialize
        print("Waiting 15 seconds for servers to be ready...")
        time.sleep(15)

        # Generate unique stream name
        stream_name = f"stream_{uuid.uuid4().hex[:8]}"
        rtsp_url = f"rtsp://127.0.0.1:8554/{stream_name}"

        print(f"--- Starting FFmpeg stream: {WAV_FILE} -> {rtsp_url} ---")
        ffmpeg = subprocess.Popen([
            "ffmpeg", "-re", "-i", WAV_FILE,
            "-c:a", "aac", "-b:a", "32k", "-ar", "16000", "-ac", "1",
            "-f", "rtsp", rtsp_url
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"--- Initializing Transcription Client ---")
        full_transcript = []
        
        # Use lang='ar' for Arabic files if needed, or leave None for auto-detection
        client = TranscriptionClient("127.0.0.1", SERVER_PORT, use_vad=False, model=MODEL)
        client.print_transcript = True

        def collect(segment):
            text = segment.text.strip()
            if text:
                full_transcript.append(text)

        client.callback = collect
        
        print(f"--- Transcribing... ---")
        client(rtsp_url)

        # Wait for FFmpeg to finish playing the file
        ffmpeg.wait()
        print("FFmpeg stream finished. Waiting for final chunks...")
        time.sleep(2)

        final_text = " ".join(full_transcript)
        print(f"\n--- FINAL TRANSCRIPT ---\n{final_text}\n------------------------\n")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    finally:
        print("--- Cleaning up processes ---")
        if client:
            try:
                client.close_all_clients()
            except:
                pass
        
        for proc, name in [(ffmpeg, "FFmpeg"), (server, "Whisper Server"), (mediamtx, "MediaMTX")]:
            if proc and proc.poll() is None:
                print(f"Terminating {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    proc.kill()

        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    run_transcription()
