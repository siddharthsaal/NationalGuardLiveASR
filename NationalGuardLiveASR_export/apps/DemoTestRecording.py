import streamlit as st
import sys
import io
import wave
import json
import queue
import threading
import av
import numpy as np
import time
from types import ModuleType

# --- FIX: Mock pyaudio before importing whisper_live ---
if 'pyaudio' not in sys.modules:
    mock_pyaudio = ModuleType('pyaudio')
    mock_pyaudio.paInt16 = 8 
    sys.modules['pyaudio'] = mock_pyaudio
# -------------------------------------------------------
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
from whisper_live.client import TranscriptionClient
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- TRUE GLOBAL STORAGE (survives reruns) ---
if "_global_state" not in globals():
    globals()["_global_state"] = {
        "frames": [],
        "count": 0,
        "energy": 0.0,
        "lock": threading.Lock(),
        "queue": queue.Queue(),
        "last_transcript": "",
    }
state = globals()["_global_state"]

st.set_page_config(page_title="WhisperLive Multi-Lang", layout="wide")
st.title("🎙️ Real-Time Transcription")

st.markdown("""
    <style>
    .rtl-text { direction: RTL; text-align: right; font-family: 'Arial'; font-size: 24px; padding: 20px; background: #f0f2f6; border-radius: 10px; border: 1px solid #ddd; min-height: 100px; }
    .ltr-text { direction: LTR; text-align: left; font-family: 'Arial'; font-size: 24px; padding: 20px; background: #f0f2f6; border-radius: 10px; border: 1px solid #ddd; min-height: 100px; }
    .meter-container { background: #eee; border-radius: 5px; height: 20px; width: 100%; margin: 10px 0; }
    .meter-fill { background: #28a745; height: 100%; border-radius: 5px; transition: width 0.1s; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Config")
    lang_map = {"English": "en", "Arabic": "ar", "Auto-Detect": None}
    selected_lang_name = st.selectbox("Language", options=list(lang_map.keys()))
    lang_code = lang_map[selected_lang_name]
    model_options = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    selected_model = st.selectbox("Whisper Model", options=model_options, index=2)
    
    if st.button("Reset Pipeline"):
        with state["lock"]:
            state["frames"].clear()
            state["count"] = 0
            state["energy"] = 0.0
            state["last_transcript"] = ""
        st.session_state.transcript = ""
        st.session_state.final_wav = None
        st.rerun()

# --- Cached Transcription Client ---
@st.cache_resource
def get_transcription_client(lang, model):
    """Create a TranscriptionClient and wait for the server to be ready."""
    print(f"DEBUG: Initializing new TranscriptionClient for {lang} {model}")
    client = TranscriptionClient("localhost", 9090, lang=lang, model=model)
    
    # IMPORTANT: Do NOT replace on_message! The original on_message handles
    # SERVER_READY which sets client.recording = True.
    # Instead, WRAP it to also capture transcripts for our UI.
    for c in client.clients:
        original_on_message = c.on_message
        
        def make_wrapper(orig):
            def wrapped_on_message(ws, message):
                # Call the ORIGINAL handler first (handles SERVER_READY, recording, segments, etc.)
                orig(ws, message)
                # Then also extract transcript for our UI
                try:
                    data = json.loads(message)
                    if "segments" in data:
                        transcript = " ".join([s['text'] for s in data['segments']])
                        state["last_transcript"] = transcript
                        print(f"DEBUG TRANSCRIPT: {transcript[:100]}")
                except:
                    pass
            return wrapped_on_message
        
        c.on_message = make_wrapper(original_on_message)
    
    # Wait for the server to be ready
    print("DEBUG: Waiting for server to be ready...")
    for c in client.clients:
        timeout = 15
        start_time = time.time()
        while not c.recording:
            if c.server_error:
                print(f"DEBUG: Server error: {getattr(c, 'error_message', 'unknown')}")
                break
            if c.waiting:
                print("DEBUG: Server is full, waiting...")
                break
            if time.time() - start_time > timeout:
                print("DEBUG: Timeout waiting for server ready")
                break
            time.sleep(0.1)
    
    if client.clients[0].recording:
        print("DEBUG: Server is READY, client.recording = True")
    else:
        print("DEBUG: WARNING - Server may not be ready!")
        
    return client

# --- Audio Processor (class-based, works with SENDONLY mode) ---
class WhisperAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        print("DEBUG: WhisperAudioProcessor CREATED")

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        resampled_frames = self.resampler.resample(frame)
        if resampled_frames:
            for f in resampled_frames:
                data = f.to_ndarray()
                energy = float(np.sqrt(np.mean(data.astype(np.float32)**2)))
                
                with state["lock"]:
                    state["count"] += 1
                    state["energy"] = energy
                
                bytes_data = data.tobytes()
                state["queue"].put(bytes_data)
                with state["lock"]:
                    state["frames"].append(bytes_data)
                
                # Debug: print every 50th frame
                if state["count"] % 50 == 0:
                    print(f"DEBUG RECV: frame #{state['count']}, energy={energy:.2f}, queue_size={state['queue'].qsize()}")
        return frame

# --- WebRTC Streamer ---
# SENDONLY mode: the browser sends audio, we don't send anything back.
# In streamlit-webrtc SENDONLY, an AudioReceiver is auto-created,
# and when audio_processor_factory is set, the processor's recv() IS called.
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="whisper-live-v6",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=WhisperAudioProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

# --- Active UI ---
if ctx.state.playing:
    st.info("🎤 Recording in progress... Speak into your microphone!")
    
    if not st.session_state.get("_actively_recording", False):
        st.session_state._actively_recording = True
        with state["lock"]:
            state["frames"].clear()
            state["count"] = 0
            state["energy"] = 0.0
            state["last_transcript"] = ""

    # Get or start the client
    transcription_client = get_transcription_client(lang_code, selected_model)
    
    # Show server connection status
    server_ready = any(c.recording for c in transcription_client.clients)
    if server_ready:
        st.success("✅ Connected to WhisperLive server")
    else:
        st.warning("⏳ Waiting for WhisperLive server connection...")
    
    # Start the sender thread if not alive
    if "sender_thread" not in st.session_state or not st.session_state.sender_thread.is_alive():
        def sender_loop():
            print("DEBUG: Sender thread STARTED")
            sent_count = 0
            while True: 
                try:
                    raw_data = state["queue"].get(timeout=1)
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    float_data = audio_data.astype(np.float32) / 32768.0
                    # Use unconditional=True to bypass the recording check
                    transcription_client.multicast_packet(float_data.tobytes(), unconditional=True)
                    sent_count += 1
                    if sent_count % 50 == 0:
                        print(f"DEBUG SENDER: sent {sent_count} packets to server")
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"SENDER ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        thread = threading.Thread(target=sender_loop, daemon=True)
        add_script_run_ctx(thread)
        st.session_state.sender_thread = thread
        st.session_state.sender_thread.start()

    # Visual Feedback
    with state["lock"]:
        count = state["count"]
        energy = state["energy"]
        transcript = state.get("last_transcript", "")
    
    st.metric("WebRTC Audio Packets", count)
    meter_fill = min(100, int(energy / 50))
    st.markdown(f"**Signal Strength**")
    st.markdown(f'<div class="meter-container"><div class="meter-fill" style="width: {meter_fill}%"></div></div>', unsafe_allow_html=True)
    
    if count == 0:
        st.error("No audio data received yet. Click **Start** and allow Microphone access.")
    else:
        st.success(f"✅ Receiving audio data! ({count} frames)")
        st.write("**Real-time Preview:**")
        st.write(transcript if transcript else "...")

    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Auto-refresh
    auto_refresh = st.checkbox("Enable auto-refresh (every 2s)", value=False)
    if auto_refresh:
        time.sleep(2)
        st.rerun()

else:
    # Stopped - save recording
    if st.session_state.get("_actively_recording", False):
        st.session_state._actively_recording = False
        with state["lock"]:
            frame_count = state["count"]
            frame_energy = state["energy"]
            frames_data = list(state["frames"])
            transcript_text = state.get("last_transcript", "")
        
        print(f"DEBUG STOP: count={frame_count}, frames_len={len(frames_data)}, energy={frame_energy}")
        
        if frames_data:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames_data))
            st.session_state.final_wav = buf.getvalue()
            st.session_state.final_transcript = transcript_text
            st.session_state.final_frame_count = frame_count
            st.session_state.final_energy = frame_energy
            print(f"DEBUG: WAV created, size={len(st.session_state.final_wav)} bytes")
        else:
            print("DEBUG: NO FRAMES captured!")
            st.session_state.final_wav = None
            st.session_state.final_transcript = ""
            st.session_state.final_frame_count = 0
            st.session_state.final_energy = 0.0
        
        st.rerun()

# --- Post-Recording UI ---
if not ctx.state.playing:
    if st.session_state.get("final_wav"):
        st.subheader("📼 Recording Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Frames Captured", st.session_state.get("final_frame_count", 0))
        with col2:
            st.metric("Final Energy Level", f"{st.session_state.get('final_energy', 0):.1f}")
        
        st.audio(st.session_state.final_wav, format="audio/wav")
        
        transcript_text = st.session_state.get("final_transcript", "No transcript available")
        text_class = "rtl-text" if lang_code == "ar" else "ltr-text"
        st.markdown(f'<div class="{text_class}">{transcript_text if transcript_text else "No transcript available"}</div>', unsafe_allow_html=True)
        
        if st.button("🎙️ Start New Test"):
            st.session_state.final_wav = None
            st.session_state.final_frame_count = 0
            st.session_state.final_energy = 0
            st.rerun()
    elif st.session_state.get("_actively_recording") == False and st.session_state.get("final_frame_count", 0) == 0:
        st.warning("⚠️ No audio frames were captured. Please ensure microphone access is allowed and try again.")
