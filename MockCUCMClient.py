#!/usr/bin/env python3
"""
MockCUCMClient.py — Simulates a CUCM SIP Trunk + IP Phone for testing
======================================================================

Sends a SIP INVITE to CUCMRTPStreamASR, negotiates media via SDP,
streams a local WAV/MP3 file as G.711 μ-law RTP packets, and sends
BYE when the file finishes.

Usage:
  # Terminal 1: Start the ASR receiver
  python3.11 CUCMRTPStreamASR.py

  # Terminal 2: Run this mock client
  python3.11 MockCUCMClient.py --file Ar_f1.wav
  python3.11 MockCUCMClient.py --file A_eng_f1.wav --server 10.1.1.50
"""

import argparse
import audioop
import logging
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)-12s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MockCUCM")
log_sip = logging.getLogger("SIP-Client")
log_rtp = logging.getLogger("RTP-Sender")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 8000          # G.711 sample rate
SAMPLE_WIDTH = 2            # 16-bit PCM
PTIME_MS = 20               # 20ms per RTP packet (standard)
SAMPLES_PER_PACKET = int(SAMPLE_RATE * PTIME_MS / 1000)  # 160 samples
BYTES_PER_PACKET = SAMPLES_PER_PACKET * SAMPLE_WIDTH      # 320 bytes of PCM
RTP_PAYLOAD_TYPE = 0        # PCMU
RTP_CLOCK_RATE = 8000


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_as_pcm_8k_mono(filepath):
    """
    Load any audio file and convert to 8kHz mono 16-bit PCM using ffmpeg.
    Returns raw PCM bytes.
    """
    log.info(f"Loading audio file: {filepath}")

    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        sys.exit(1)

    # Use ffmpeg to convert to 8kHz mono 16-bit PCM WAV
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", filepath,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        tmp_wav
    ]
    log.info(f"  Converting with ffmpeg: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  FFmpeg failed: {result.stderr[:500]}")
        sys.exit(1)

    # Read the WAV file
    with wave.open(tmp_wav, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm_data = wf.readframes(n_frames)

    os.unlink(tmp_wav)

    duration = n_frames / framerate
    log.info(f"  Loaded: {n_frames} frames, {framerate}Hz, {n_channels}ch, "
             f"{sample_width*8}bit, {duration:.2f}s")
    log.info(f"  PCM data: {len(pcm_data)} bytes")

    return pcm_data, duration


def pcm_to_ulaw(pcm_data):
    """Convert 16-bit PCM to G.711 μ-law."""
    log.info(f"Encoding PCM ({len(pcm_data)} bytes) to G.711 μ-law...")
    ulaw_data = audioop.lin2ulaw(pcm_data, SAMPLE_WIDTH)
    log.info(f"  μ-law data: {len(ulaw_data)} bytes "
             f"(compression: {len(pcm_data)/len(ulaw_data):.1f}x)")
    return ulaw_data


# ─────────────────────────────────────────────────────────────────────────────
# SIP CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class MockSIPClient:
    """Sends SIP INVITE, receives 200 OK, sends ACK/BYE."""

    def __init__(self, server_ip, server_port, local_rtp_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.local_rtp_port = local_rtp_port
        self.call_id = str(uuid.uuid4())
        self.from_tag = uuid.uuid4().hex[:8]
        self.to_tag = None
        self.cseq = 1
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(10)
        self.local_ip = self._get_local_ip()
        self.remote_rtp_port = None  # Will be learned from SDP in 200 OK

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.server_ip, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def send_invite(self):
        """Send SIP INVITE with SDP offering PCMU."""
        log_sip.info(f"Sending SIP INVITE to {self.server_ip}:{self.server_port}")
        log_sip.info(f"  Call-ID: {self.call_id}")
        log_sip.info(f"  Local IP: {self.local_ip}")
        log_sip.info(f"  Local RTP port: {self.local_rtp_port}")

        sdp_body = (
            "v=0\r\n"
            f"o=MockCUCM 1234 1234 IN IP4 {self.local_ip}\r\n"
            "s=Mock CUCM Call\r\n"
            f"c=IN IP4 {self.local_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.local_rtp_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=sendonly\r\n"
        )

        invite = (
            f"INVITE sip:whisper@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:phone@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: <sip:whisper@{self.server_ip}>\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} INVITE\r\n"
            f"Contact: <sip:phone@{self.local_ip}:5060>\r\n"
            f"Content-Type: application/sdp\r\n"
            f"Max-Forwards: 70\r\n"
            f"User-Agent: MockCUCM/1.0\r\n"
            f"Content-Length: {len(sdp_body)}\r\n"
            "\r\n"
            f"{sdp_body}"
        )

        self.socket.sendto(invite.encode(), (self.server_ip, self.server_port))
        log_sip.info(f"  INVITE sent ({len(invite)} bytes)")
        self.cseq += 1

    def wait_for_response(self):
        """Wait for 100 Trying and 200 OK, extract remote RTP port from SDP."""
        log_sip.info("Waiting for SIP response...")

        got_ok = False
        attempts = 0
        max_attempts = 10

        while not got_ok and attempts < max_attempts:
            attempts += 1
            try:
                data, addr = self.socket.recvfrom(65535)
                text = data.decode("utf-8", errors="replace")
                first_line = text.split("\r\n")[0]

                if "100 Trying" in first_line:
                    log_sip.info(f"  ← Received 100 Trying from {addr}")
                    continue

                elif "200 OK" in first_line:
                    log_sip.info(f"  ← Received 200 OK from {addr}")

                    # Extract To tag
                    to_match = re.search(r"To:.*?;tag=([^\s;>\r\n]+)", text)
                    if to_match:
                        self.to_tag = to_match.group(1)
                        log_sip.info(f"    To-tag: {self.to_tag}")

                    # Extract remote RTP port from SDP
                    m_match = re.search(r"m=audio (\d+)", text)
                    if m_match:
                        self.remote_rtp_port = int(m_match.group(1))
                        log_sip.info(f"    Remote RTP port: {self.remote_rtp_port}")

                    # Extract remote IP from SDP c= line
                    c_match = re.search(r"c=IN IP4 ([\d.]+)", text)
                    if c_match:
                        remote_media_ip = c_match.group(1)
                        log_sip.info(f"    Remote media IP: {remote_media_ip}")

                    got_ok = True

                else:
                    log_sip.warning(f"  ← Unexpected response: {first_line}")

            except socket.timeout:
                log_sip.warning(f"  Timeout waiting for response (attempt {attempts})")

        if not got_ok:
            log_sip.error("  Failed to receive 200 OK — aborting")
            return False

        return True

    def send_ack(self):
        """Send SIP ACK to confirm the call."""
        log_sip.info("Sending SIP ACK...")

        to_hdr = f"<sip:whisper@{self.server_ip}>"
        if self.to_tag:
            to_hdr += f";tag={self.to_tag}"

        ack = (
            f"ACK sip:whisper@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:phone@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: {to_hdr}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: 1 ACK\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n"
            "\r\n"
        )
        self.socket.sendto(ack.encode(), (self.server_ip, self.server_port))
        log_sip.info("  ACK sent")

    def send_bye(self):
        """Send SIP BYE to end the call."""
        log_sip.info("Sending SIP BYE...")

        to_hdr = f"<sip:whisper@{self.server_ip}>"
        if self.to_tag:
            to_hdr += f";tag={self.to_tag}"

        bye = (
            f"BYE sip:whisper@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:phone@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: {to_hdr}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} BYE\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n"
            "\r\n"
        )
        self.socket.sendto(bye.encode(), (self.server_ip, self.server_port))
        log_sip.info("  BYE sent")

        # Wait for 200 OK
        try:
            data, addr = self.socket.recvfrom(65535)
            first_line = data.decode("utf-8", errors="replace").split("\r\n")[0]
            log_sip.info(f"  ← Response: {first_line}")
        except socket.timeout:
            log_sip.warning("  No response to BYE (timeout)")

    def close(self):
        self.socket.close()


# ─────────────────────────────────────────────────────────────────────────────
# RTP SENDER
# ─────────────────────────────────────────────────────────────────────────────
class RTPSender:
    """Sends G.711 μ-law audio as RTP packets to the ASR server."""

    def __init__(self, dest_ip, dest_port, source_port):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", source_port))
        self.ssrc = struct.unpack("!I", os.urandom(4))[0]
        self.sequence = 0
        self.timestamp = 0
        self.packets_sent = 0
        self.bytes_sent = 0

    def build_rtp_packet(self, payload):
        """Build an RTP packet with header + payload."""
        # RTP header: V=2, P=0, X=0, CC=0, M=0, PT=0
        byte0 = 0x80  # Version 2
        byte1 = RTP_PAYLOAD_TYPE  # No marker, PT=0 (PCMU)

        header = struct.pack("!BBHII",
                             byte0, byte1,
                             self.sequence & 0xFFFF,
                             self.timestamp & 0xFFFFFFFF,
                             self.ssrc)

        self.sequence += 1
        self.timestamp += SAMPLES_PER_PACKET  # 160 samples per 20ms

        return header + payload

    def send_audio(self, ulaw_data, audio_duration):
        """
        Send μ-law audio data as RTP packets at real-time pace.
        Each packet contains 160 bytes (20ms at 8kHz).
        """
        # Each μ-law sample is 1 byte, so 160 bytes = 20ms
        ulaw_bytes_per_packet = SAMPLES_PER_PACKET  # 160

        total_packets = len(ulaw_data) // ulaw_bytes_per_packet
        log_rtp.info(f"Starting RTP stream:")
        log_rtp.info(f"  Destination: {self.dest_ip}:{self.dest_port}")
        log_rtp.info(f"  SSRC: {self.ssrc:#010x}")
        log_rtp.info(f"  Total μ-law bytes: {len(ulaw_data)}")
        log_rtp.info(f"  Bytes per packet: {ulaw_bytes_per_packet}")
        log_rtp.info(f"  Total packets: {total_packets}")
        log_rtp.info(f"  Expected duration: {audio_duration:.2f}s")
        log_rtp.info(f"  Pacing: real-time ({PTIME_MS}ms per packet)")

        start_time = time.time()
        last_progress_time = start_time

        for i in range(total_packets):
            if not self._running:
                break

            offset = i * ulaw_bytes_per_packet
            payload = ulaw_data[offset:offset + ulaw_bytes_per_packet]

            packet = self.build_rtp_packet(payload)
            self.socket.sendto(packet, (self.dest_ip, self.dest_port))
            self.packets_sent += 1
            self.bytes_sent += len(packet)

            # Log first packet
            if i == 0:
                log_rtp.info(f"  *** First RTP packet sent ***")
                log_rtp.info(f"    Packet size: {len(packet)} bytes "
                             f"(header=12, payload={len(payload)})")

            # Progress log every 2 seconds
            now = time.time()
            if now - last_progress_time >= 2.0:
                elapsed = now - start_time
                pct = (i + 1) / total_packets * 100
                log_rtp.info(f"  Progress: {pct:.0f}% | "
                             f"Packets: {self.packets_sent} | "
                             f"Elapsed: {elapsed:.1f}s / {audio_duration:.1f}s")
                last_progress_time = now

            # Real-time pacing: sleep 20ms between packets
            expected_time = start_time + (i + 1) * PTIME_MS / 1000.0
            sleep_time = expected_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        elapsed = time.time() - start_time
        log_rtp.info(f"RTP stream complete:")
        log_rtp.info(f"  Packets sent: {self.packets_sent}")
        log_rtp.info(f"  Bytes sent: {self.bytes_sent}")
        log_rtp.info(f"  Elapsed: {elapsed:.2f}s (expected {audio_duration:.2f}s)")
        log_rtp.info(f"  Rate: {self.packets_sent/elapsed:.0f} pkt/s")

    _running = True

    def close(self):
        self._running = False
        self.socket.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Mock CUCM SIP/RTP Client — streams a local audio file "
                    "to CUCMRTPStreamASR for end-to-end testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Arabic audio
  python3.11 MockCUCMClient.py --file Ar_f1.wav

  # Test with English audio
  python3.11 MockCUCMClient.py --file A_eng_f1.wav

  # Custom server address
  python3.11 MockCUCMClient.py --file Ar_f1.wav --server 10.1.1.50 --port 5060
        """
    )
    parser.add_argument("--file", required=True,
                        help="Audio file to stream (WAV/MP3)")
    parser.add_argument("--server", default="127.0.0.1",
                        help="CUCMRTPStreamASR server IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5060,
                        help="SIP port on the server (default: 5060)")
    parser.add_argument("--rtp-port", type=int, default=20000,
                        help="Local RTP port to send from (default: 20000)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait after stream ends before sending BYE (default: 2.0)")

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  MOCK CUCM CLIENT — End-to-End Test")
    log.info("=" * 60)
    log.info(f"  Audio file:   {args.file}")
    log.info(f"  Server:       {args.server}:{args.port}")
    log.info(f"  Local RTP:    {args.rtp_port}")
    log.info("")

    # ── Step 1: Load and encode audio ──────────────────────────────────
    log.info("STEP 1/6: Loading and encoding audio file")
    pcm_data, duration = load_audio_as_pcm_8k_mono(args.file)
    ulaw_data = pcm_to_ulaw(pcm_data)

    # ── Step 2: SIP INVITE ─────────────────────────────────────────────
    log.info("")
    log.info("STEP 2/6: Sending SIP INVITE")
    sip = MockSIPClient(args.server, args.port, args.rtp_port)

    sip.send_invite()

    # ── Step 3: Wait for 200 OK ────────────────────────────────────────
    log.info("")
    log.info("STEP 3/6: Waiting for SIP 200 OK")
    if not sip.wait_for_response():
        log.error("Failed to establish SIP session — exiting")
        sip.close()
        sys.exit(1)

    if not sip.remote_rtp_port:
        log.error("No remote RTP port in SDP — cannot send audio")
        sip.close()
        sys.exit(1)

    # ── Step 4: Send ACK ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 4/6: Sending SIP ACK")
    sip.send_ack()

    # Small delay for ASR to prepare
    time.sleep(1)

    # ── Step 5: Stream RTP audio ───────────────────────────────────────
    log.info("")
    log.info("STEP 5/6: Streaming RTP audio")
    rtp = RTPSender(args.server, sip.remote_rtp_port, args.rtp_port)

    try:
        rtp.send_audio(ulaw_data, duration)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        rtp.close()

    # Wait for final transcription chunks
    log.info(f"  Waiting {args.delay}s for final transcription...")
    time.sleep(args.delay)

    # ── Step 6: Send BYE ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 6/6: Sending SIP BYE")
    sip.send_bye()
    sip.close()

    log.info("")
    log.info("=" * 60)
    log.info("  MOCK CUCM CLIENT — Test Complete")
    log.info("=" * 60)
    log.info("Check the CUCMRTPStreamASR terminal for transcription output.")
    log.info("Check cucm_rtp_asr.log for detailed server-side logs.")


if __name__ == "__main__":
    main()
