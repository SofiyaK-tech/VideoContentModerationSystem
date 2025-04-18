from django.shortcuts import render, redirect, get_object_or_404
from .models import Video
from .forms import VideoForm
from django.conf import settings
from django.http import HttpResponse
import os
import cv2
import numpy as np
import subprocess
import ffmpeg

from .ml_utils import (
    extract_frames,
    extract_audio,
    moderate_audio,
    analyze_frames,
    blur_flagged_frames,
    make_decision
)

import whisper
from pydub import AudioSegment, silence
from better_profanity import profanity


def home(request):
    return render(request, 'home.html')


def video_list(request):
    videos = Video.objects.all()
    return render(request, 'videolist.html', {'videos': videos})


def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('videolist')
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})


def extract_frames(video_path, video_id):
    output_dir = os.path.join(settings.MEDIA_ROOT, f"frames/{video_id}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1

    cap.release()
    return frame_paths


def extract_audio(video_path, video_id):
    output_audio = os.path.join(settings.MEDIA_ROOT, f"audio/{video_id}.mp3")
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    (
        ffmpeg
        .input(video_path)
        .output(output_audio, format="mp3", acodec="libmp3lame")
        .run(overwrite_output=True)
    )

    return output_audio


def preprocess_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = os.path.join(settings.MEDIA_ROOT, str(video.file))

    frame_dir = os.path.join(settings.MEDIA_ROOT, 'frames', str(video_id))
    audio_path = os.path.join(settings.MEDIA_ROOT, 'audio', f"{video_id}.mp3")

    os.makedirs(frame_dir, exist_ok=True)

    # Extract Frames
    if not os.listdir(frame_dir):
        extract_frames(video_path, video_id)

    # Extract Audio
    if not os.path.exists(audio_path):
        extract_audio(video_path, video_id)

    # Prepare frame and audio URLs
    frames = [
        f"{settings.MEDIA_URL}frames/{video_id}/{img}"
        for img in sorted(os.listdir(frame_dir)) if img.endswith(".jpg")
    ]
    audio_url = f"{settings.MEDIA_URL}audio/{video_id}.mp3"

    return render(request, "preprocess.html", {
        "frames": frames,
        "audio": audio_url,
        "video_id": video_id
    })


def moderated_frames_view(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = os.path.join(settings.MEDIA_ROOT, str(video.file))

    frame_dir = os.path.join(settings.MEDIA_ROOT, "frames", str(video_id))
    output_dir = os.path.join(settings.MEDIA_ROOT, "moderated_frames", str(video_id))

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames if not present
    if not os.listdir(frame_dir):
        frame_paths = extract_frames(video_path, video_id)
    else:
        frame_paths = [
            os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir)) if f.endswith(".jpg")
        ]

    if not os.listdir(output_dir):
        # Analyze frames
        flagged = analyze_frames(frame_paths)
        decision = make_decision(flagged)

        if decision == "blur":
            images = [cv2.imread(p) for p in frame_paths]
            blur_flagged_frames(images, flagged, output_dir)
        elif decision == "remove":
            return render(request, "error.html", {"message": "❌ Video is flagged for removal."})
        else:
            for i, path in enumerate(frame_paths):
                frame = cv2.imread(path)
                out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(out_path, frame)

    # Render moderated frames
    frames = [
        f"{settings.MEDIA_URL}moderated_frames/{video_id}/{img}"
        for img in sorted(os.listdir(output_dir)) if img.endswith(".jpg")
    ]

    return render(request, "moderated_frames.html", {
        "frames": frames,
        "video_id": video_id
    })


def transcribe_and_censor_audio(video_id):
    video = get_object_or_404(Video, id=video_id)
    audio_path = os.path.join(settings.MEDIA_ROOT, 'audio', f"{video_id}.wav")

    if not os.path.exists(audio_path):
        return None, None

    output_dir = os.path.join(settings.MEDIA_ROOT, 'processed', str(video_id))
    os.makedirs(output_dir, exist_ok=True)

    try:
        model = whisper.load_model("base")
        audio = AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(16000)
        processed_audio_path = os.path.join(output_dir, "processed_audio.wav")
        audio.export(processed_audio_path, format="wav")

        result = model.transcribe(processed_audio_path)
        transcript = result['text']
        censored_text = profanity.censor(transcript)

        with open(os.path.join(output_dir, "transcribed_text.txt"), "w") as f:
            f.write(censored_text)

        segments = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        censored_audio = AudioSegment.silent(duration=0)

        for word, segment in zip(transcript.split(), segments):
            if profanity.contains_profanity(word):
                censored_audio += AudioSegment.silent(duration=len(segment))
            else:
                censored_audio += segment

        censored_audio_path = os.path.join(output_dir, "censored_audio.wav")
        censored_audio.export(censored_audio_path, format="wav")

        return os.path.join(output_dir, "transcribed_text.txt"), censored_audio_path
    except Exception as e:
        print(f"❌ Error in transcribe_and_censor_audio: {e}")
        return None, None


def transcribe_audio_view(request, video_id):
    transcribed_text_path, censored_audio_path = transcribe_and_censor_audio(video_id)
    transcript = open(transcribed_text_path).read() if transcribed_text_path else None
    audio = f"{settings.MEDIA_URL}processed/{video_id}/censored_audio.wav" if censored_audio_path else None

    return render(request, 'transcribe_result.html', {
        'audio': audio,
        'transcript': transcript
    })


def live_options_view(request):
    return render(request, 'live_options.html')