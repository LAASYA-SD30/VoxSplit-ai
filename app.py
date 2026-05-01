import os
import ssl
import tempfile
import base64
import io
import subprocess
import shutil


from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch
import torchaudio


ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Load Silero VAD
try:
    model, utils = torch.hub.load(
        'snakers4/silero-vad',
        'silero_vad',
        trust_repo=True
    )
except Exception as e:
    print("MODEL LOAD ERROR:", e)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

SAMPLE_RATE = 16000

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ffmpeg_available():
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_to_wav(src_path):
    """
    Use ffmpeg to convert any audio format (including webm/opus) to a
    temporary WAV file. Returns the path to the WAV.
    Raises RuntimeError if ffmpeg is not available or conversion fails.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed or not found on PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html — "
            "on Windows, download a build and add it to your PATH."
        )

    dst_path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-f", "wav",
                dst_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed:\n{result.stderr.decode(errors='replace')}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg executable not found. "
            "Please install ffmpeg and ensure it is on your system PATH. "
            "Download from: https://ffmpeg.org/download.html"
        )
    return dst_path


def load_any_audio(path):
    """
    Load audio from *path*, converting via ffmpeg first if torchaudio
    cannot open the format directly (e.g. webm/opus from MediaRecorder).
    Returns a 1-D float32 tensor at SAMPLE_RATE.
    """
    converted_path = None
    try:
        # Try torchaudio first (fast path for wav/mp3/flac/ogg/m4a)
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as torchaudio_err:
            # Fall back: convert with ffmpeg → load the resulting wav
            converted_path = convert_to_wav(path)
            waveform, sr = torchaudio.load(converted_path)

        if waveform.shape[0] > 1:                        # stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:                             # resample
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform.squeeze(0)                        # (samples,)

    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)


def tensor_to_wav_bytes(tensor):
    """Return mono 16-kHz PCM WAV as a BytesIO."""
    buf = io.BytesIO()
    torchaudio.save(buf, tensor.unsqueeze(0), SAMPLE_RATE, format="wav")
    buf.seek(0)
    return buf


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[-1].lower() or ".wav"
    tmp  = tempfile.mktemp(suffix=ext)
    file.save(tmp)

    # Read tunable parameters from the form
    threshold   = float(request.form.get("threshold", 0.5))
    min_silence = int(request.form.get("min_silence_duration_ms", 300))

    try:
        audio = load_any_audio(tmp)
        timestamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence,
        )

        segments, total_speech = [], 0.0
        for ts in timestamps:
            start = round(ts["start"] / SAMPLE_RATE, 3)
            end   = round(ts["end"]   / SAMPLE_RATE, 3)
            dur   = round(end - start, 3)
            total_speech += dur
            segments.append({
                "start": start, "end": end, "duration": dur,
                "sample_start": ts["start"], "sample_end": ts["end"],
            })

        total_duration  = round(len(audio) / SAMPLE_RATE, 3)
        silence_removed = round(total_duration - total_speech, 3)

        return jsonify({
            "segments":        segments,
            "count":           len(segments),
            "total_duration":  total_duration,
            "total_speech":    round(total_speech, 3),
            "silence_removed": silence_removed,
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/get_segment_audio", methods=["POST"])
def get_segment_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    idx  = int(request.form.get("index", 0))
    threshold   = float(request.form.get("threshold", 0.5))
    min_silence = int(request.form.get("min_silence_duration_ms", 300))
    ext  = os.path.splitext(file.filename)[-1].lower() or ".wav"
    tmp  = tempfile.mktemp(suffix=ext)
    file.save(tmp)

    try:
        audio      = load_any_audio(tmp)
        timestamps = get_speech_timestamps(
            audio, model, sampling_rate=SAMPLE_RATE,
            threshold=threshold, min_silence_duration_ms=min_silence,
        )
        if idx >= len(timestamps):
            return jsonify({"error": "Segment index out of range"})

        ts    = timestamps[idx]
        chunk = audio[ts["start"]:ts["end"]]
        buf   = tensor_to_wav_bytes(chunk)
        b64   = base64.b64encode(buf.read()).decode()
        return jsonify({"audio_b64": b64})

    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/get_trimmed_audio", methods=["POST"])
def get_trimmed_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    threshold   = float(request.form.get("threshold", 0.5))
    min_silence = int(request.form.get("min_silence_duration_ms", 300))
    ext  = os.path.splitext(file.filename)[-1].lower() or ".wav"
    tmp  = tempfile.mktemp(suffix=ext)
    file.save(tmp)

    try:
        audio      = load_any_audio(tmp)
        timestamps = get_speech_timestamps(
            audio, model, sampling_rate=SAMPLE_RATE,
            threshold=threshold, min_silence_duration_ms=min_silence,
        )
        trimmed = collect_chunks(timestamps, audio)
        buf     = tensor_to_wav_bytes(trimmed)
        b64     = base64.b64encode(buf.read()).decode()
        return jsonify({"audio_b64": b64})

    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/download_trimmed", methods=["POST"])
def download_trimmed():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    threshold   = float(request.form.get("threshold", 0.5))
    min_silence = int(request.form.get("min_silence_duration_ms", 300))
    ext  = os.path.splitext(file.filename)[-1].lower() or ".wav"
    tmp  = tempfile.mktemp(suffix=ext)
    file.save(tmp)

    try:
        audio      = load_any_audio(tmp)
        timestamps = get_speech_timestamps(
            audio, model, sampling_rate=SAMPLE_RATE,
            threshold=threshold, min_silence_duration_ms=min_silence,
        )
        trimmed = collect_chunks(timestamps, audio)
        buf     = tensor_to_wav_bytes(trimmed)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=True,
                         download_name="voxsplit_trimmed.wav")
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/download_segment", methods=["POST"])
def download_segment():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    idx  = int(request.form.get("index", 0))
    threshold   = float(request.form.get("threshold", 0.5))
    min_silence = int(request.form.get("min_silence_duration_ms", 300))
    ext  = os.path.splitext(file.filename)[-1].lower() or ".wav"
    tmp  = tempfile.mktemp(suffix=ext)
    file.save(tmp)

    try:
        audio      = load_any_audio(tmp)
        timestamps = get_speech_timestamps(
            audio, model, sampling_rate=SAMPLE_RATE,
            threshold=threshold, min_silence_duration_ms=min_silence,
        )
        ts    = timestamps[idx]
        chunk = audio[ts["start"]:ts["end"]]
        buf   = tensor_to_wav_bytes(chunk)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=True,
                         download_name=f"voxsplit_segment_{idx+1}.wav")
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    app.run(debug=True)