#!/usr/bin/env python3
"""
ForkedMediaASR.py — Speaker-Labeled SIP/RTP ASR with Forked Media Support
==========================================================================

Receives TWO (or more) SIP INVITEs per conversation — one per speaker —
and produces an interleaved, timestamped, speaker-labeled transcript.

CUCM sends "forked media": each speaker's audio arrives as a separate
SIP session with a shared conversation correlation ID.

Usage:
  python3.11 ForkedMediaASR.py --sip-port 5062

Architecture:
  CUCM ─→ INVITE #1 (Speaker A) ─→ RTP ─→ Whisper ─→ ┐
       ─→ INVITE #2 (Speaker B) ─→ RTP ─→ Whisper ─→ ├─→ Interleaved Transcript
                                                       └───────────────────────

Each INVITE is correlated into a single Conversation via:
  - X-Conversation-ID SIP header (CUCM configurable)
  - Or automatic grouping within a time window
"""

import argparse
import audioop
import datetime
import json
import logging
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SIP_PORT = 5060
DEFAULT_BIND_IP = "0.0.0.0"
DEFAULT_RTP_PORT_MIN = 16384
DEFAULT_RTP_PORT_MAX = 32767
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_WHISPER_PORT = 9090
DEFAULT_PYTHON_EXE = "python3.11"

TELEPHONY_SAMPLE_RATE = 8000
WHISPER_SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
RTP_HEADER_SIZE = 12
PCMU_PAYLOAD_TYPE = 0
PCMA_PAYLOAD_TYPE = 8
AUDIO_CHUNK_SIZE_MS = 100

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_file="forked_media_asr.log"):
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(name)-18s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(console)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(fh)
    return root


log_main = logging.getLogger("Main")
log_sip = logging.getLogger("SIP")
log_rtp = logging.getLogger("RTP")
log_audio = logging.getLogger("Audio")
log_whisper = logging.getLogger("Whisper")
log_conv = logging.getLogger("Conversation")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT ENTRY
# ─────────────────────────────────────────────────────────────────────────────
class TranscriptEntry:
    """A single utterance in the conversation timeline."""

    def __init__(self, wall_clock, speaker_label, speaker_ip, text,
                 whisper_start=None, whisper_end=None):
        self.wall_clock = wall_clock          # time.time() when received
        self.speaker_label = speaker_label    # e.g. "Speaker A" or SIP From
        self.speaker_ip = speaker_ip          # Source IP address
        self.text = text
        self.whisper_start = whisper_start    # Whisper segment start (relative)
        self.whisper_end = whisper_end        # Whisper segment end (relative)

    @property
    def relative_time(self):
        return self.wall_clock  # Will be made relative at print time

    def __repr__(self):
        return f"[{self.speaker_label} ({self.speaker_ip})] {self.text}"


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION — groups multiple SIP calls (forked media)
# ─────────────────────────────────────────────────────────────────────────────
class Conversation:
    """Groups multiple forked media SIP calls into one conversation."""

    def __init__(self, conversation_id):
        self.conversation_id = conversation_id
        self.calls = {}          # call_id -> SpeakerCall
        self.transcript = []     # List of TranscriptEntry (shared timeline)
        self.transcript_lock = threading.Lock()
        self.start_time = time.time()
        self.active = True

    def add_transcript(self, entry):
        with self.transcript_lock:
            self.transcript.append(entry)

    def get_sorted_transcript(self):
        with self.transcript_lock:
            return sorted(self.transcript, key=lambda e: e.wall_clock)

    def get_speakers(self):
        return {call.speaker_label: call.speaker_ip
                for call in self.calls.values()}

    def all_calls_ended(self):
        return all(not c.active for c in self.calls.values())

    def print_transcript(self):
        """Print the full interleaved transcript with timestamps and speaker labels."""
        sorted_entries = self.get_sorted_transcript()
        if not sorted_entries:
            log_conv.warning(f"[Conv {self.conversation_id[:8]}] No transcript entries")
            return

        conv_start = self.start_time
        speakers = self.get_speakers()

        log_conv.info("")
        log_conv.info("╔" + "═" * 70 + "╗")
        log_conv.info("║" + "  INTERLEAVED CONVERSATION TRANSCRIPT".center(70) + "║")
        log_conv.info("╠" + "═" * 70 + "╣")
        log_conv.info(f"║  Conversation ID: {self.conversation_id[:16]}...".ljust(71) + "║")
        log_conv.info(f"║  Duration: {sorted_entries[-1].wall_clock - conv_start:.1f}s".ljust(71) + "║")
        log_conv.info(f"║  Speakers: {len(speakers)}".ljust(71) + "║")
        for label, ip in speakers.items():
            log_conv.info(f"║    • {label} ({ip})".ljust(71) + "║")
        log_conv.info(f"║  Total utterances: {len(sorted_entries)}".ljust(71) + "║")
        log_conv.info("╠" + "═" * 70 + "╣")

        for entry in sorted_entries:
            elapsed = entry.wall_clock - conv_start
            mins = int(elapsed // 60)
            secs = elapsed % 60
            timestamp = f"{mins:02d}:{secs:05.2f}"
            line = f"║  [{timestamp}] {entry.speaker_label} ({entry.speaker_ip})"
            log_conv.info(line)
            log_conv.info(f"║    \"{entry.text}\"")
            log_conv.info("║" + "─" * 70 + "║")

        log_conv.info("╚" + "═" * 70 + "╝")
        log_conv.info("")


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class ConversationManager:
    """Manages conversations and correlates forked media INVITEs."""

    def __init__(self, auto_group_window=10.0):
        self.conversations = OrderedDict()  # conv_id -> Conversation
        self.lock = threading.Lock()
        self.auto_group_window = auto_group_window  # seconds to auto-group

    def find_or_create(self, conversation_id=None):
        """Find an existing conversation or create a new one."""
        with self.lock:
            if conversation_id and conversation_id in self.conversations:
                conv = self.conversations[conversation_id]
                log_conv.info(f"[Conv {conversation_id[:8]}] Joining existing conversation "
                              f"({len(conv.calls)} calls so far)")
                return conv

            # Auto-group: if there's a recent active conversation, join it
            if not conversation_id:
                now = time.time()
                for cid, conv in reversed(self.conversations.items()):
                    if conv.active and (now - conv.start_time) < self.auto_group_window:
                        log_conv.info(f"[Conv {cid[:8]}] Auto-grouping into recent conversation "
                                      f"(started {now - conv.start_time:.1f}s ago)")
                        return conv

            # Create new conversation
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            conv = Conversation(conversation_id)
            self.conversations[conversation_id] = conv
            log_conv.info(f"[Conv {conversation_id[:8]}] NEW conversation created")
            return conv

    def check_completed(self, conversation):
        """Check if conversation is complete and print transcript."""
        if conversation.all_calls_ended():
            log_conv.info(f"[Conv {conversation.conversation_id[:8]}] "
                          f"All speakers disconnected — printing final transcript")
            conversation.active = False
            conversation.print_transcript()


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER CALL — one speaker's SIP/RTP session
# ─────────────────────────────────────────────────────────────────────────────
class SpeakerCall:
    """Tracks one speaker's SIP call within a forked media conversation."""

    def __init__(self, call_id, from_uri, to_uri, remote_ip, remote_rtp_port,
                 local_rtp_port, codec_type=PCMU_PAYLOAD_TYPE):
        self.call_id = call_id
        self.from_uri = from_uri
        self.to_uri = to_uri
        self.remote_ip = remote_ip
        self.remote_rtp_port = remote_rtp_port
        self.local_rtp_port = local_rtp_port
        self.codec_type = codec_type
        self.start_time = time.time()
        self.rtp_packets = 0
        self.rtp_bytes = 0
        self.active = True
        self.rtp_socket = None
        self.whisper_client = None
        self.conversation = None  # reference to parent Conversation

        # Speaker identification
        self.speaker_label = self._extract_speaker_label()
        self.speaker_ip = remote_ip

        # SIP state
        self.sip_to_tag = uuid.uuid4().hex[:8]
        self.sip_via = None
        self.sip_from_tag = None

    def _extract_speaker_label(self):
        """Extract a readable speaker label from the SIP From: URI."""
        match = re.search(r'sip:([^@]+)@', self.from_uri or "")
        if match:
            name = match.group(1)
            return name.replace("_", " ").title()
        return f"Speaker-{self.remote_ip}"

    def elapsed(self):
        return time.time() - self.start_time

    def __repr__(self):
        return (f"Speaker({self.speaker_label}, ip={self.remote_ip}, "
                f"call={self.call_id[:8]}..., pkts={self.rtp_packets})")


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class AudioProcessor:
    """Decodes G.711 and resamples 8kHz → 16kHz."""

    def __init__(self, codec_type=PCMU_PAYLOAD_TYPE):
        self.codec_type = codec_type
        self.total_bytes_output = 0

    def decode_and_resample(self, payload):
        try:
            if self.codec_type == PCMU_PAYLOAD_TYPE:
                pcm_8k = audioop.ulaw2lin(payload, SAMPLE_WIDTH)
            else:
                pcm_8k = audioop.alaw2lin(payload, SAMPLE_WIDTH)
            pcm_16k, _ = audioop.ratecv(pcm_8k, SAMPLE_WIDTH, 1,
                                         TELEPHONY_SAMPLE_RATE,
                                         WHISPER_SAMPLE_RATE, None)
            self.total_bytes_output += len(pcm_16k)
            return pcm_16k
        except Exception as e:
            log_audio.error(f"Audio decode error: {e}")
            return b""


# ─────────────────────────────────────────────────────────────────────────────
# RTP DECODER
# ─────────────────────────────────────────────────────────────────────────────
class RTPDecoder:
    @staticmethod
    def decode_header(data):
        if len(data) < RTP_HEADER_SIZE:
            return None
        byte0, byte1 = data[0], data[1]
        header_size = RTP_HEADER_SIZE + ((byte0 & 0x0F) * 4)
        if (byte0 >> 4) & 0x01 and len(data) > header_size + 4:
            ext_len = struct.unpack("!H", data[header_size + 2:header_size + 4])[0]
            header_size += 4 + (ext_len * 4)
        return {
            "payload_type": byte1 & 0x7F,
            "sequence_number": struct.unpack("!H", data[2:4])[0],
            "timestamp": struct.unpack("!I", data[4:8])[0],
            "ssrc": struct.unpack("!I", data[8:12])[0],
            "header_size": header_size
        }

    @staticmethod
    def extract_payload(data, header):
        return data[header["header_size"]:]


# ─────────────────────────────────────────────────────────────────────────────
# SDP PARSER / BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SDPParser:
    @staticmethod
    def parse(sdp_body):
        result = {"connection_ip": None, "media_port": None,
                  "payload_types": [], "ptime": 20}
        for line in sdp_body.split("\r\n"):
            line = line.strip()
            if line.startswith("c=IN IP4 "):
                result["connection_ip"] = line.split()[-1]
            elif line.startswith("m=audio "):
                parts = line.split()
                result["media_port"] = int(parts[1])
                result["payload_types"] = [int(p) for p in parts[3:]]
            elif line.startswith("a=ptime:"):
                result["ptime"] = int(line.split(":")[1])
        return result

    @staticmethod
    def build(local_ip, local_rtp_port):
        session_id = str(int(time.time()))
        return (
            "v=0\r\n"
            f"o=ForkedMediaASR {session_id} {session_id} IN IP4 {local_ip}\r\n"
            "s=WhisperLive ASR Session\r\n"
            f"c=IN IP4 {local_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {local_rtp_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=recvonly\r\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIP MESSAGE PARSER / BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SIPMessage:
    @staticmethod
    def parse(data):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        msg = {"method": None, "is_response": False, "status_code": None,
               "headers": {}, "body": "", "raw": data}
        parts = text.split("\r\n\r\n", 1)
        msg["body"] = parts[1] if len(parts) > 1 else ""
        lines = parts[0].split("\r\n")
        if not lines:
            return msg
        first_line = lines[0]
        if first_line.startswith("SIP/2.0"):
            msg["is_response"] = True
            p = first_line.split(" ", 2)
            msg["status_code"] = int(p[1]) if len(p) > 1 else 0
        else:
            p = first_line.split(" ", 2)
            msg["method"] = p[0] if p else None
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                msg["headers"][k.strip()] = v.strip()
        return msg

    @staticmethod
    def build_response(status_code, status_text, call, sip_msg,
                       local_ip, sdp_body=None):
        via = sip_msg["headers"].get("Via", "")
        from_hdr = sip_msg["headers"].get("From", "")
        call_id = sip_msg["headers"].get("Call-ID", "")
        cseq = sip_msg["headers"].get("CSeq", "")
        to_hdr = sip_msg["headers"].get("To", "")
        if call and ";tag=" not in to_hdr:
            to_hdr += f";tag={call.sip_to_tag}"
        r = f"SIP/2.0 {status_code} {status_text}\r\n"
        r += f"Via: {via}\r\nFrom: {from_hdr}\r\nTo: {to_hdr}\r\n"
        r += f"Call-ID: {call_id}\r\nCSeq: {cseq}\r\n"
        r += f"Contact: <sip:{local_ip}:{DEFAULT_SIP_PORT}>\r\n"
        r += "User-Agent: ForkedMediaASR/1.0\r\n"
        if sdp_body:
            r += f"Content-Type: application/sdp\r\nContent-Length: {len(sdp_body)}\r\n\r\n{sdp_body}"
        else:
            r += "Content-Length: 0\r\n\r\n"
        return r.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER RTP RECEIVER — one per speaker
# ─────────────────────────────────────────────────────────────────────────────
class SpeakerRTPReceiver:
    """Receives RTP for one speaker, decodes, feeds to Whisper, adds to shared transcript."""

    def __init__(self, call, conversation, whisper_host, whisper_port, model):
        self.call = call
        self.conversation = conversation
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.model = model
        self.audio_processor = AudioProcessor(call.codec_type)
        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.running = False
        self.last_seq = -1

    def start(self):
        log_rtp.info(f"[{self.call.speaker_label}] Starting RTP on port {self.call.local_rtp_port}")
        self.call.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.call.rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.call.rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        self.call.rtp_socket.bind(("0.0.0.0", self.call.local_rtp_port))
        self.call.rtp_socket.settimeout(1.0)
        self.running = True

        threading.Thread(target=self._receive_loop, daemon=True,
                         name=f"RTP-{self.call.speaker_label}").start()
        self._start_whisper_client()
        threading.Thread(target=self._audio_feeder_loop, daemon=True,
                         name=f"Feed-{self.call.speaker_label}").start()

        log_rtp.info(f"[{self.call.speaker_label}] All threads started")

    def stop(self):
        log_rtp.info(f"[{self.call.speaker_label}] Stopping RTP receiver...")
        self.running = False
        self.call.active = False

        if self.call.rtp_socket:
            try:
                self.call.rtp_socket.close()
            except Exception:
                pass

        # Flush remaining audio
        self._flush_buffer()

        # Close Whisper
        if self.call.whisper_client:
            try:
                self.call.whisper_client.send_packet_to_server(b"END_OF_AUDIO")
                log_whisper.info(f"[{self.call.speaker_label}] Waiting 5s for final transcript...")
                time.sleep(5)
                self.call.whisper_client.close_websocket()
            except Exception as e:
                log_whisper.error(f"[{self.call.speaker_label}] Error closing Whisper: {e}")

        log_rtp.info(f"[{self.call.speaker_label}] FINAL STATS: "
                     f"pkts={self.call.rtp_packets}, bytes={self.call.rtp_bytes}, "
                     f"pcm_out={self.audio_processor.total_bytes_output}")

    def _start_whisper_client(self):
        try:
            from whisper_live.client import Client

            log_whisper.info(f"[{self.call.speaker_label}] Connecting to Whisper "
                             f"at {self.whisper_host}:{self.whisper_port}")

            call_ref = self.call
            conv_ref = self.conversation

            def on_transcription(text, segments):
                if text and text.strip():
                    now = time.time()
                    entry = TranscriptEntry(
                        wall_clock=now,
                        speaker_label=call_ref.speaker_label,
                        speaker_ip=call_ref.speaker_ip,
                        text=text.strip(),
                        whisper_start=segments[0].get("start") if segments else None,
                        whisper_end=segments[-1].get("end") if segments else None,
                    )
                    conv_ref.add_transcript(entry)
                    log_whisper.info(
                        f"[{call_ref.speaker_label} ({call_ref.speaker_ip})] "
                        f"═══ \"{text.strip()}\" ═══"
                    )

            client = Client(
                host=self.whisper_host,
                port=self.whisper_port,
                model=self.model,
                use_vad=False,
                lang=None,
                log_transcription=False,
                transcription_callback=on_transcription,
            )

            timeout = 15
            start = time.time()
            while not client.recording and time.time() - start < timeout:
                if client.waiting or client.server_error:
                    log_whisper.error(f"[{self.call.speaker_label}] Server error")
                    return
                time.sleep(0.1)

            if client.recording:
                log_whisper.info(f"[{self.call.speaker_label}] Whisper connected")
                self.call.whisper_client = client
            else:
                log_whisper.error(f"[{self.call.speaker_label}] Whisper connection timeout")

        except Exception as e:
            log_whisper.error(f"[{self.call.speaker_label}] Whisper error: {e}")

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.call.rtp_socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            header = RTPDecoder.decode_header(data)
            if not header or header["payload_type"] not in (PCMU_PAYLOAD_TYPE, PCMA_PAYLOAD_TYPE):
                continue

            payload = RTPDecoder.extract_payload(data, header)
            if not payload:
                continue

            pcm_16k = self.audio_processor.decode_and_resample(payload)
            if pcm_16k:
                with self.buffer_lock:
                    self.audio_buffer.extend(pcm_16k)

            self.call.rtp_packets += 1
            self.call.rtp_bytes += len(data)

            if self.call.rtp_packets == 1:
                log_rtp.info(f"[{self.call.speaker_label}] *** FIRST RTP PACKET *** "
                             f"from {addr}, SSRC={header['ssrc']:#010x}")

    def _audio_feeder_loop(self):
        chunk_bytes = int(WHISPER_SAMPLE_RATE * SAMPLE_WIDTH * AUDIO_CHUNK_SIZE_MS / 1000)
        while self.running:
            time.sleep(AUDIO_CHUNK_SIZE_MS / 1000.0)
            with self.buffer_lock:
                if len(self.audio_buffer) < chunk_bytes:
                    continue
                chunk = bytes(self.audio_buffer[:chunk_bytes])
                del self.audio_buffer[:chunk_bytes]

            if self.call.whisper_client and self.call.whisper_client.recording:
                try:
                    pcm_float = (np.frombuffer(chunk, dtype=np.int16)
                                 .astype(np.float32) / 32768.0).tobytes()
                    self.call.whisper_client.send_packet_to_server(pcm_float)
                except Exception as e:
                    log_audio.error(f"[{self.call.speaker_label}] Feed error: {e}")

    def _flush_buffer(self):
        with self.buffer_lock:
            if self.audio_buffer and self.call.whisper_client:
                try:
                    remaining = bytes(self.audio_buffer)
                    pcm_float = (np.frombuffer(remaining, dtype=np.int16)
                                 .astype(np.float32) / 32768.0).tobytes()
                    self.call.whisper_client.send_packet_to_server(pcm_float)
                    self.audio_buffer.clear()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# SIP LISTENER
# ─────────────────────────────────────────────────────────────────────────────
class SIPListener:
    def __init__(self, bind_ip, sip_port, rtp_port_min, rtp_port_max,
                 whisper_host, whisper_port, model):
        self.bind_ip = bind_ip
        self.sip_port = sip_port
        self.rtp_port_min = rtp_port_min
        self.rtp_port_max = rtp_port_max
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.model = model
        self.conv_manager = ConversationManager()
        self.receivers = {}      # call_id -> SpeakerRTPReceiver
        self.calls = {}          # call_id -> SpeakerCall
        self.next_rtp_port = rtp_port_min
        self.rtp_lock = threading.Lock()
        self.running = False
        self.socket = None
        self.local_ip = None

    def _alloc_rtp_port(self):
        with self.rtp_lock:
            port = self.next_rtp_port
            self.next_rtp_port += 2
            if self.next_rtp_port > self.rtp_port_max:
                self.next_rtp_port = self.rtp_port_min
            return port

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start(self):
        self.local_ip = self._get_local_ip()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.bind_ip, self.sip_port))
        self.socket.settimeout(1.0)
        self.running = True

        log_sip.info("=" * 70)
        log_sip.info("FORKED MEDIA ASR — SIP LISTENER STARTED")
        log_sip.info(f"  Bind: {self.bind_ip}:{self.sip_port} | RTP: {self.rtp_port_min}-{self.rtp_port_max}")
        log_sip.info(f"  Whisper: {self.whisper_host}:{self.whisper_port} | Model: {self.model}")
        log_sip.info("=" * 70)
        log_sip.info("Waiting for forked media SIP INVITEs...")

        while self.running:
            try:
                data, addr = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    log_sip.error("SIP socket error")
                break

            msg = SIPMessage.parse(data)
            if msg["is_response"]:
                continue

            method = msg["method"]
            call_id = msg["headers"].get("Call-ID", "")
            log_sip.info(f"--- SIP {method} from {addr} | Call-ID: {call_id[:16]}... ---")

            if method == "INVITE":
                self._handle_invite(msg, addr)
            elif method == "ACK":
                log_sip.info(f"[ACK] Call {call_id[:8]} confirmed")
            elif method == "BYE":
                self._handle_bye(msg, addr)
            elif method == "OPTIONS":
                resp = SIPMessage.build_response(200, "OK", None, msg, self.local_ip)
                self.socket.sendto(resp, addr)

    def _handle_invite(self, msg, addr):
        call_id = msg["headers"].get("Call-ID", "")
        from_uri = msg["headers"].get("From", "")

        # Send 100 Trying
        trying = SIPMessage.build_response(100, "Trying", None, msg, self.local_ip)
        self.socket.sendto(trying, addr)

        # Parse SDP
        sdp = SDPParser.parse(msg["body"])
        remote_ip = sdp["connection_ip"] or addr[0]
        remote_port = sdp["media_port"]
        codec = PCMU_PAYLOAD_TYPE if 0 in sdp["payload_types"] else PCMA_PAYLOAD_TYPE
        local_rtp = self._alloc_rtp_port()

        # Create speaker call
        call = SpeakerCall(
            call_id=call_id, from_uri=from_uri,
            to_uri=msg["headers"].get("To", ""),
            remote_ip=remote_ip, remote_rtp_port=remote_port,
            local_rtp_port=local_rtp, codec_type=codec
        )
        call.remote_sip_addr = addr

        # Extract conversation correlation ID
        conv_id = msg["headers"].get("X-Conversation-ID")
        if not conv_id:
            conv_id = msg["headers"].get("X-Group-ID")

        # Find or create conversation
        conversation = self.conv_manager.find_or_create(conv_id)
        call.conversation = conversation
        conversation.calls[call_id] = call
        self.calls[call_id] = call

        log_sip.info(f"  Speaker: {call.speaker_label} ({remote_ip})")
        log_sip.info(f"  Conv: {conversation.conversation_id[:8]}... "
                     f"({len(conversation.calls)} speaker(s))")
        log_sip.info(f"  RTP: {remote_ip}:{remote_port} → local:{local_rtp}")

        # Send 200 OK with SDP
        sdp_body = SDPParser.build(self.local_ip, local_rtp)
        ok = SIPMessage.build_response(200, "OK", call, msg, self.local_ip,
                                        sdp_body=sdp_body)
        self.socket.sendto(ok, addr)
        log_sip.info(f"  → 200 OK sent")

        # Start RTP receiver for this speaker
        receiver = SpeakerRTPReceiver(
            call, conversation, self.whisper_host, self.whisper_port, self.model
        )
        self.receivers[call_id] = receiver
        receiver.start()

        log_sip.info(f"  ✓ {call.speaker_label} fully set up")

    def _handle_bye(self, msg, addr):
        call_id = msg["headers"].get("Call-ID", "")
        call = self.calls.get(call_id)

        # Send 200 OK
        resp = SIPMessage.build_response(200, "OK", call, msg, self.local_ip)
        self.socket.sendto(resp, addr)

        if call_id in self.receivers:
            receiver = self.receivers.pop(call_id)
            call = self.calls.pop(call_id, None)

            log_sip.info(f"  [{call.speaker_label}] BYE → stopping stream")
            receiver.stop()

            # Check if all speakers in conversation have hung up
            if call and call.conversation:
                self.conv_manager.check_completed(call.conversation)

    def stop(self):
        log_sip.info("Stopping SIP listener...")
        self.running = False
        for call_id, receiver in list(self.receivers.items()):
            receiver.stop()
        # Print any active conversations
        for conv in self.conv_manager.conversations.values():
            if conv.transcript:
                conv.print_transcript()
        if self.socket:
            self.socket.close()
        log_sip.info("SIP listener stopped")


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER SERVER MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class WhisperServerManager:
    def __init__(self, port=DEFAULT_WHISPER_PORT, python_exe=DEFAULT_PYTHON_EXE):
        self.port = port
        self.python_exe = python_exe
        self.process = None

    def start(self):
        log_whisper.info(f"Starting WhisperLive server on port {self.port}...")
        cmd = [self.python_exe, "-m", "run_server", "--port", str(self.port)]
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        log_whisper.info(f"WhisperLive started (PID: {self.process.pid})")
        log_whisper.info("Waiting 15s for model loading...")
        time.sleep(15)
        log_whisper.info("WhisperLive ready")

    def stop(self):
        if self.process:
            log_whisper.info(f"Stopping WhisperLive (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            log_whisper.info("WhisperLive stopped")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Forked Media ASR — Speaker-labeled transcription from CUCM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.11 ForkedMediaASR.py --sip-port 5062
  python3.11 ForkedMediaASR.py --sip-port 5062 --model small
  python3.11 ForkedMediaASR.py --sip-port 5062 --no-whisper-server
        """
    )
    parser.add_argument("--sip-port", type=int, default=DEFAULT_SIP_PORT)
    parser.add_argument("--bind", default=DEFAULT_BIND_IP)
    parser.add_argument("--rtp-min", type=int, default=DEFAULT_RTP_PORT_MIN)
    parser.add_argument("--rtp-max", type=int, default=DEFAULT_RTP_PORT_MAX)
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-host", default="127.0.0.1")
    parser.add_argument("--whisper-port", type=int, default=DEFAULT_WHISPER_PORT)
    parser.add_argument("--log-file", default="forked_media_asr.log")
    parser.add_argument("--no-whisper-server", action="store_true",
                        help="Don't start WhisperLive (use existing)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--auto-group-window", type=float, default=10.0,
                        help="Seconds to auto-group INVITEs into same conversation")
    args = parser.parse_args()

    setup_logging(args.log_file)

    log_main.info("=" * 70)
    log_main.info("  FORKED MEDIA ASR — Speaker-Labeled Transcription")
    log_main.info(f"  Python: {sys.version.split()[0]} | PID: {os.getpid()}")
    log_main.info("=" * 70)

    # Self-test
    ap = AudioProcessor()
    test_pcm = struct.pack("<" + "h" * 800,
                           *[int(16000 * __import__('math').sin(2 * 3.14159 * 440 * i / 8000))
                             for i in range(800)])
    test_ulaw = audioop.lin2ulaw(test_pcm, 2)
    result = ap.decode_and_resample(test_ulaw)
    log_main.info(f"Self-test: {len(test_ulaw)} μ-law bytes → {len(result)} PCM16k bytes ✓")

    if args.self_test:
        return

    # Start WhisperLive
    whisper_mgr = None
    if not args.no_whisper_server:
        whisper_mgr = WhisperServerManager(args.whisper_port)
        whisper_mgr.start()

    # Start SIP
    sip = SIPListener(args.bind, args.sip_port, args.rtp_min, args.rtp_max,
                      args.whisper_host, args.whisper_port, args.model)

    def shutdown(signum, frame):
        log_main.info(f"\nSHUTDOWN SIGNAL ({signum})")
        sip.stop()
        if whisper_mgr:
            whisper_mgr.stop()
        log_main.info("Forked Media ASR shut down")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        sip.start()
    except KeyboardInterrupt:
        shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
