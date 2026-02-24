#!/usr/bin/env python3
"""
MockForkedClient.py — Simulates CUCM Forked Media (two speakers)
=================================================================

Sends TWO SIP INVITEs to ForkedMediaASR, each representing a different
speaker. Streams separate audio for each speaker as RTP.

Modes:
  --stereo FILE.wav     Split stereo file: left=Speaker A, right=Speaker B
  --file-a F1 --file-b F2   Two separate mono files for each speaker

Usage:
  # Stereo mode (splits Agent-Client.wav into two speakers)
  python3.11 MockForkedClient.py --stereo Agent-Client.wav --port 5062

  # Two-file mode (two Arabic files as two speakers)
  python3.11 MockForkedClient.py --file-a Ar_f1.wav --file-b Ar_f2.wav --port 5062
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

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)-12s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MockForked")

SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
PTIME_MS = 20
SAMPLES_PER_PACKET = int(SAMPLE_RATE * PTIME_MS / 1000)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_audio_mono(filepath):
    """Load and convert any audio file to 8kHz mono PCM."""
    log.info(f"Loading: {filepath}")
    if not os.path.exists(filepath):
        log.error(f"Not found: {filepath}")
        sys.exit(1)
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-i", filepath, "-ar", str(SAMPLE_RATE),
                     "-ac", "1", "-sample_fmt", "s16", "-f", "wav", tmp],
                    capture_output=True)
    with wave.open(tmp, "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        dur = wf.getnframes() / wf.getframerate()
    os.unlink(tmp)
    log.info(f"  → {dur:.2f}s, {len(pcm)} bytes PCM")
    return pcm, dur


def split_stereo(filepath):
    """Split a stereo file into two mono PCM streams (left + right channel)."""
    log.info(f"Splitting stereo: {filepath}")
    tmp_l = tempfile.mktemp(suffix="_left.wav")
    tmp_r = tempfile.mktemp(suffix="_right.wav")

    subprocess.run(["ffmpeg", "-y", "-i", filepath,
                     "-af", "pan=mono|c0=c0", "-ar", str(SAMPLE_RATE),
                     "-sample_fmt", "s16", tmp_l], capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", filepath,
                     "-af", "pan=mono|c0=c1", "-ar", str(SAMPLE_RATE),
                     "-sample_fmt", "s16", tmp_r], capture_output=True)

    with wave.open(tmp_l, "rb") as wf:
        pcm_l = wf.readframes(wf.getnframes())
        dur_l = wf.getnframes() / wf.getframerate()
    with wave.open(tmp_r, "rb") as wf:
        pcm_r = wf.readframes(wf.getnframes())
        dur_r = wf.getnframes() / wf.getframerate()
    os.unlink(tmp_l)
    os.unlink(tmp_r)

    log.info(f"  Left (Speaker A): {dur_l:.2f}s, {len(pcm_l)} bytes")
    log.info(f"  Right (Speaker B): {dur_r:.2f}s, {len(pcm_r)} bytes")
    return (pcm_l, dur_l), (pcm_r, dur_r)


def pcm_to_ulaw(pcm):
    return audioop.lin2ulaw(pcm, SAMPLE_WIDTH)


# ─────────────────────────────────────────────────────────────────────────────
# SIP CLIENT (for one speaker)
# ─────────────────────────────────────────────────────────────────────────────
class SIPClient:
    def __init__(self, server_ip, server_port, local_rtp_port,
                 speaker_name, conversation_id):
        self.server_ip = server_ip
        self.server_port = server_port
        self.local_rtp_port = local_rtp_port
        self.speaker_name = speaker_name
        self.conversation_id = conversation_id
        self.call_id = str(uuid.uuid4())
        self.from_tag = uuid.uuid4().hex[:8]
        self.to_tag = None
        self.cseq = 1
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(10)
        self.remote_rtp_port = None
        self.local_ip = self._get_local_ip()

    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.server_ip, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def send_invite(self):
        log.info(f"[{self.speaker_name}] Sending INVITE (Call-ID: {self.call_id[:8]}...)")

        sdp = (
            "v=0\r\n"
            f"o={self.speaker_name} 1234 1234 IN IP4 {self.local_ip}\r\n"
            f"s={self.speaker_name} Call\r\n"
            f"c=IN IP4 {self.local_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.local_rtp_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=sendonly\r\n"
        )

        invite = (
            f"INVITE sip:asr@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:{self.speaker_name}@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: <sip:asr@{self.server_ip}>\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} INVITE\r\n"
            f"Contact: <sip:{self.speaker_name}@{self.local_ip}:5060>\r\n"
            f"X-Conversation-ID: {self.conversation_id}\r\n"
            "Content-Type: application/sdp\r\n"
            "Max-Forwards: 70\r\n"
            f"User-Agent: MockForked-{self.speaker_name}/1.0\r\n"
            f"Content-Length: {len(sdp)}\r\n"
            "\r\n"
            f"{sdp}"
        )
        self.socket.sendto(invite.encode(), (self.server_ip, self.server_port))
        self.cseq += 1

    def wait_for_ok(self):
        for _ in range(10):
            try:
                data, addr = self.socket.recvfrom(65535)
                text = data.decode("utf-8", errors="replace")
                first = text.split("\r\n")[0]
                if "100 Trying" in first:
                    continue
                if "200 OK" in first:
                    m = re.search(r"m=audio (\d+)", text)
                    if m:
                        self.remote_rtp_port = int(m.group(1))
                    t = re.search(r"To:.*?;tag=([^\s;>\r\n]+)", text)
                    if t:
                        self.to_tag = t.group(1)
                    log.info(f"[{self.speaker_name}] ← 200 OK (RTP port: {self.remote_rtp_port})")
                    return True
            except socket.timeout:
                pass
        return False

    def send_ack(self):
        to_hdr = f"<sip:asr@{self.server_ip}>"
        if self.to_tag:
            to_hdr += f";tag={self.to_tag}"
        ack = (
            f"ACK sip:asr@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:{self.speaker_name}@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: {to_hdr}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            "CSeq: 1 ACK\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.socket.sendto(ack.encode(), (self.server_ip, self.server_port))

    def send_bye(self):
        to_hdr = f"<sip:asr@{self.server_ip}>"
        if self.to_tag:
            to_hdr += f";tag={self.to_tag}"
        bye = (
            f"BYE sip:asr@{self.server_ip}:{self.server_port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:5060;branch=z9hG4bK{uuid.uuid4().hex[:12]}\r\n"
            f"From: <sip:{self.speaker_name}@{self.local_ip}>;tag={self.from_tag}\r\n"
            f"To: {to_hdr}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} BYE\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.socket.sendto(bye.encode(), (self.server_ip, self.server_port))
        try:
            data, _ = self.socket.recvfrom(65535)
            log.info(f"[{self.speaker_name}] ← {data.decode()[:30]}")
        except socket.timeout:
            pass

    def close(self):
        self.socket.close()


# ─────────────────────────────────────────────────────────────────────────────
# RTP SENDER (for one speaker)
# ─────────────────────────────────────────────────────────────────────────────
class RTPSender:
    def __init__(self, dest_ip, dest_port, source_port, speaker_name):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.speaker_name = speaker_name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", source_port))
        self.ssrc = struct.unpack("!I", os.urandom(4))[0]
        self.seq = 0
        self.ts = 0
        self.sent = 0

    def send_audio(self, ulaw_data, duration):
        total = len(ulaw_data) // SAMPLES_PER_PACKET
        log.info(f"[{self.speaker_name}] Streaming {total} RTP packets ({duration:.1f}s)")
        start = time.time()

        for i in range(total):
            payload = ulaw_data[i * SAMPLES_PER_PACKET:(i + 1) * SAMPLES_PER_PACKET]
            header = struct.pack("!BBHII", 0x80, 0, self.seq & 0xFFFF,
                                 self.ts & 0xFFFFFFFF, self.ssrc)
            self.socket.sendto(header + payload, (self.dest_ip, self.dest_port))
            self.seq += 1
            self.ts += SAMPLES_PER_PACKET
            self.sent += 1

            # Real-time pacing
            expected = start + (i + 1) * PTIME_MS / 1000.0
            sleep = expected - time.time()
            if sleep > 0:
                time.sleep(sleep)

        elapsed = time.time() - start
        log.info(f"[{self.speaker_name}] Done: {self.sent} pkts in {elapsed:.1f}s")

    def close(self):
        self.socket.close()


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER STREAMER THREAD
# ─────────────────────────────────────────────────────────────────────────────
def stream_speaker(server_ip, server_port, local_rtp_port, speaker_name,
                   conversation_id, ulaw_data, duration, delay_before_bye=5):
    """Complete SIP+RTP flow for one speaker (runs in a thread)."""
    sip = SIPClient(server_ip, server_port, local_rtp_port,
                    speaker_name, conversation_id)

    sip.send_invite()
    if not sip.wait_for_ok():
        log.error(f"[{speaker_name}] Failed to get 200 OK")
        sip.close()
        return
    sip.send_ack()
    time.sleep(1)

    rtp = RTPSender(server_ip, sip.remote_rtp_port, local_rtp_port, speaker_name)
    try:
        rtp.send_audio(ulaw_data, duration)
    except Exception as e:
        log.error(f"[{speaker_name}] RTP error: {e}")
    finally:
        rtp.close()

    log.info(f"[{speaker_name}] Waiting {delay_before_bye}s for final transcription...")
    time.sleep(delay_before_bye)

    sip.send_bye()
    sip.close()
    log.info(f"[{speaker_name}] Call ended")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Mock CUCM Forked Media — sends two speaker streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stereo file (splits channels)
  python3.11 MockForkedClient.py --stereo Agent-Client.wav --port 5062

  # Two separate files
  python3.11 MockForkedClient.py --file-a Ar_f1.wav --file-b Ar_f2.wav --port 5062

  # Two Arabic files with speaker names
  python3.11 MockForkedClient.py --file-a Ar_f1.wav --file-b Ar_f2.wav \\
      --speaker-a caller --speaker-b receiver --port 5062
        """
    )
    parser.add_argument("--stereo", help="Stereo WAV file to split into two speakers")
    parser.add_argument("--file-a", help="Audio file for Speaker A")
    parser.add_argument("--file-b", help="Audio file for Speaker B")
    parser.add_argument("--speaker-a", default="agent", help="Speaker A name (default: agent)")
    parser.add_argument("--speaker-b", default="customer", help="Speaker B name (default: customer)")
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5060, help="SIP port")
    parser.add_argument("--rtp-a", type=int, default=20000, help="Local RTP port for Speaker A")
    parser.add_argument("--rtp-b", type=int, default=20002, help="Local RTP port for Speaker B")
    parser.add_argument("--delay", type=float, default=8.0,
                        help="Seconds to wait after stream ends before BYE")

    args = parser.parse_args()

    if not args.stereo and not (args.file_a and args.file_b):
        parser.error("Either --stereo or both --file-a and --file-b required")

    log.info("=" * 60)
    log.info("  MOCK FORKED MEDIA CLIENT — Two-Speaker Test")
    log.info("=" * 60)

    # Load audio
    if args.stereo:
        (pcm_a, dur_a), (pcm_b, dur_b) = split_stereo(args.stereo)
        if not args.speaker_a or args.speaker_a == "agent":
            args.speaker_a = "agent"
        if not args.speaker_b or args.speaker_b == "customer":
            args.speaker_b = "customer"
    else:
        pcm_a, dur_a = load_audio_mono(args.file_a)
        pcm_b, dur_b = load_audio_mono(args.file_b)

    # Encode to μ-law
    ulaw_a = pcm_to_ulaw(pcm_a)
    ulaw_b = pcm_to_ulaw(pcm_b)
    log.info(f"  Speaker A ({args.speaker_a}): {dur_a:.1f}s, {len(ulaw_a)} μ-law bytes")
    log.info(f"  Speaker B ({args.speaker_b}): {dur_b:.1f}s, {len(ulaw_b)} μ-law bytes")

    # Shared conversation ID (correlates the two INVITEs)
    conversation_id = str(uuid.uuid4())
    log.info(f"  Conversation ID: {conversation_id[:16]}...")
    log.info("")

    # Start both speakers in parallel threads
    import threading
    threads = []

    t_a = threading.Thread(
        target=stream_speaker,
        args=(args.server, args.port, args.rtp_a, args.speaker_a,
              conversation_id, ulaw_a, dur_a, args.delay),
        name=f"Speaker-{args.speaker_a}"
    )

    t_b = threading.Thread(
        target=stream_speaker,
        args=(args.server, args.port, args.rtp_b, args.speaker_b,
              conversation_id, ulaw_b, dur_b, args.delay),
        name=f"Speaker-{args.speaker_b}"
    )

    t_a.start()
    time.sleep(0.5)  # Small offset so INVITEs don't collide
    t_b.start()

    t_a.join()
    t_b.join()

    log.info("")
    log.info("=" * 60)
    log.info("  Both speakers done — check ForkedMediaASR for transcript")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
