import os
import cv2
import numpy as np
import ffmpeg
import whisper
from pydub import AudioSegment, silence
from better_profanity import profanity
from ultralytics import YOLO
from PIL import Image
import tempfile

# ========== Utilities ==========

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

# ========== Frame Extraction ==========

def extract_frames(video_path, output_dir, frame_size=(224, 224), grayscale=True):
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_frame = frame.astype(np.float32) / 255.0
        frames.append(norm_frame)

        # Save individual frame
        i = len(frames) - 1
        filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        if grayscale:
            cv2.imwrite(filename, (norm_frame * 255).astype(np.uint8))
        else:
            cv2.imwrite(filename, cv2.cvtColor((norm_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    cap.release()
    return np.array(frames)

# ========== Audio Extraction ==========

def extract_audio(video_path, output_path):
    ensure_dir(os.path.dirname(output_path))
    if not os.path.exists(output_path):
        ffmpeg.input(video_path).output(output_path, format='mp3').run(overwrite_output=True)
    return output_path

# ========== Audio Moderation ==========

def moderate_audio(audio_path, output_dir):
    ensure_dir(output_dir)
    model = whisper.load_model("base")
    audio = AudioSegment.from_mp3(audio_path).set_channels(1).set_frame_rate(16000)
    temp_path = os.path.join(output_dir, "temp.wav")
    audio.export(temp_path, format="wav")

    result = model.transcribe(temp_path)
    transcript = result['text']
    censored = profanity.censor(transcript)

    # Save transcript
    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(censored)

    # Mute profane parts
    segments = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    censored_audio = AudioSegment.silent(duration=0)

    for word, segment in zip(transcript.split(), segments):
        if profanity.contains_profanity(word):
            censored_audio += AudioSegment.silent(duration=len(segment))
        else:
            censored_audio += segment

    censored_path = os.path.join(output_dir, "censored_audio.mp3")
    censored_audio.export(censored_path, format="mp3")

    return censored_path

# ========== Frame Moderation ==========

def analyze_frames(frame_paths, model=None, threshold=0.5):
    model = model or YOLO("yolov8n.pt")
    flagged = []

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb)

        for r in results:
            for box in r.boxes:
                if box.conf[0] > threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    flagged.append((i, (x1, y1, x2, y2), conf))

    return flagged

# ========== Blurring ==========

def blur_flagged_frames(frames, flagged, output_dir):
    ensure_dir(output_dir)
    for i, (index, (x1, y1, x2, y2), _) in enumerate(flagged):
        frame = frames[index]
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            frame = (frame * 255).astype(np.uint8)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)
        cv2.imwrite(os.path.join(output_dir, f"blurred_frame_{index:04d}.jpg"), frame)

# ========== Decision Making ==========

def make_decision(flagged, mild=0.6, severe=0.85):
    if not flagged:
        return "safe"
    max_conf = max(flag[2] for flag in flagged)
    if max_conf >= severe:
        return "remove"
    elif max_conf >= mild:
        return "blur"
    return "safe"

# ========== Optional utility ==========

def blur_if_flagged(frame):
    return cv2.GaussianBlur(frame, (51, 51), 0)