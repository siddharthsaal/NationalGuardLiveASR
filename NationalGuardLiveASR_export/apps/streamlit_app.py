import streamlit as st
import os
import time
import json
import threading
import numpy as np
import av
import io
from whisper_live.client import Client

st.set_page_config(page_title="WhisperLive Multi-Model Test", layout="wide")

st.title("🎙️ WhisperLive Multi-Model Real-time Test")

# Sidebar Configuration
st.sidebar.header("Configuration")
host = st.sidebar.text_input("Server Host", "localhost")
port = st.sidebar.number_input("Server Port", value=9090)
use_vad = st.sidebar.checkbox("Use VAD", value=False)

models_to_test = st.sidebar.multiselect(
    "Select Models to Test",
    options=["base", "small", "turbo"],
    default=["base", "small", "turbo"]
)

# Session State for Transcriptions
if "transcripts" not in st.session_state:
    st.session_state.transcripts = {model: "" for model in ["base", "small", "turbo"]}

def transcription_callback(model_name):
    def callback(text, segments):
        print(f"[CALLBACK {model_name}] Text: {text[:50] if text else 'None'}... Segments: {len(segments) if segments else 0}")
        if text:
            st.session_state.transcripts[model_name] = text
    return callback

def process_audio(audio_bytes):
    """Resample audio to 16kHz mono float32."""
    input_container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(
        format='flt',
        layout='mono',
        rate=16000,
    )
    
    audio_data = []
    for frame in input_container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            for resampled_frame in resampled_frames:
                audio_data.append(resampled_frame.to_ndarray().flatten())
    
    if not audio_data:
        return None
        
    return np.concatenate(audio_data).flatten()

# UI Layout
input_method = st.radio("Choose Input Method", ["Record Audio", "Upload Audio"])

audio_data_to_transcribe = None

if input_method == "Record Audio":
    audio_recorded = st.audio_input("Record your voice")
    if audio_recorded:
        audio_data_to_transcribe = audio_recorded.read()
else:
    audio_uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    if audio_uploaded:
        audio_data_to_transcribe = audio_uploaded.read()

if st.button("Transcribe", disabled=not (audio_data_to_transcribe and models_to_test)):
    if not audio_data_to_transcribe:
        st.error("Please provide audio first.")
    elif not models_to_test:
        st.error("Please select at least one model.")
    else:
        # Clear previous transcripts
        for model in models_to_test:
            st.session_state.transcripts[model] = "Connecting..."
        
        # Display Areas
        cols = st.columns(len(models_to_test))
        containers = {}
        for i, model in enumerate(models_to_test):
            with cols[i]:
                st.subheader(f"Model: {model}")
                containers[model] = st.empty()
                containers[model].info("Preprocessing audio...")

        # Preprocess Audio
        float_audio = process_audio(audio_data_to_transcribe)
        
        if float_audio is not None:
            clients = []
            
            # Start Clients Sequentially
            for model in models_to_test:
                containers[model].info(f"Connecting to model: {model}...")
                client = Client(
                    host=host,
                    port=port,
                    model=model,
                    use_vad=use_vad,
                    log_transcription=False,
                    transcription_callback=transcription_callback(model)
                )
                clients.append(client)
                
                # Wait for this specific client to be ready
                client_wait_start = time.time()
                client_timeout = 60 # 60 seconds per model
                while not client.recording:
                    if client.server_error:
                        st.error(f"Server error for {model}: {getattr(client, 'error_message', 'Unknown error')}")
                        break
                    if time.time() - client_wait_start > client_timeout:
                        containers[model].error(f"Timeout connecting to {model}")
                        break
                    time.sleep(0.5)
                
                if client.recording:
                    containers[model].success(f"Model {model} Ready!")
                else:
                    st.warning(f"Could not initialize {model}. Skipping.")

            # Filter only properly connected clients
            active_clients = [c for c in clients if c.recording]
            
            if not active_clients:
                st.error("No models could be initialized. Please check the server.")
            else:
                st.info("Starting Transcription...")
                # Stream Audio in chunks to simulate real-time
                chunk_size = 4096  # samples
                for i in range(0, len(float_audio), chunk_size):
                    chunk = float_audio[i:i + chunk_size]
                    chunk_bytes = chunk.tobytes()
                    for client in active_clients:
                        client.send_packet_to_server(chunk_bytes)
                    
                    time.sleep(0.01) # Small delay to allow UI updates and simulate streaming

                # Signal End of Audio to active clients only
                for client in active_clients:
                    client.send_packet_to_server(Client.END_OF_AUDIO.encode('utf-8'))
                
                # Wait for transcription results (give server time to process and send back)
                st.info("Waiting for transcription results...")
                wait_start = time.time()
                max_wait = 15  # Wait up to 15 seconds for results
                
                while time.time() - wait_start < max_wait:
                    # Update UI with any transcription that has come in
                    has_content = False
                    for model in models_to_test:
                        if st.session_state.transcripts[model] and st.session_state.transcripts[model] != "Connecting...":
                            containers[model].write(st.session_state.transcripts[model])
                            has_content = True
                    
                    # If we have content, wait a bit longer for final segments
                    if has_content:
                        time.sleep(2)
                        break
                    
                    time.sleep(0.5)
                
                # Final UI Update
                for model in models_to_test:
                    if st.session_state.transcripts[model] and st.session_state.transcripts[model] != "Connecting...":
                        containers[model].success("Transcription Complete")
                        containers[model].write(st.session_state.transcripts[model])
                    else:
                        containers[model].warning("No transcription received")
                    
                # Cleanup active clients only
                for client in active_clients:
                    try:
                        client.client_socket.close()
                    except:
                        pass
        else:
            st.error("Failed to process audio. Please try another file.")

# Instructions
with st.expander("How to run this?"):
    st.markdown(f"""
    1. **Start the WhisperLive Server:**
       ```bash
       chmod +x start_server.sh
       ./start_server.sh
       ```
    2. **Run the Streamlit App:**
       ```bash
       streamlit run streamlit_app.py
       ```
    3. **Test:** Record or upload audio and click 'Transcribe'.
    """)
