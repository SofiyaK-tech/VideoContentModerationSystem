import os
import cv2
import time
from django.shortcuts import render, get_object_or_404
from django.http import StreamingHttpResponse
from django.conf import settings
from .models import Video
from .ml_utils import blur_if_flagged , analyze_frames

# 🎥 Frame generator (used for both file and webcam/IP camera)
def generate_stream(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        time.sleep(0.05)  # control frame rate
    cap.release()

# 🎬 Stream a stored video
def live_stream(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = os.path.join(settings.MEDIA_ROOT, str(video.file))
    return StreamingHttpResponse(generate_stream(video_path),
        content_type='multipart/x-mixed-replace; boundary=frame')

# 📡 Stream from webcam or IP camera
def live_camera_stream(request):
    # 0 = default webcam, or replace with IP URL like "http://192.168.1.10:8080/video"
    webcam_url = request.GET.get("url", "0")
    try:
        cam_source = int(webcam_url)
    except ValueError:
        cam_source = webcam_url  # It's probably an IP camera URL

    return StreamingHttpResponse(generate_stream(cam_source),
        content_type='multipart/x-mixed-replace; boundary=frame')

# 📄 Page to select live options
def live_stream_list(request):
    videos = Video.objects.all()
    return render(request, 'live_list.html', {'videos': videos})


# 🔁 Generator for real-time streaming with moderation
def generate_moderated_stream(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Analyze the frame
        flagged = analyze_frames(frame)  # return True/False

        # Step 2: Blur frame if flagged
        if flagged:
            frame = blur_if_flagged(frame)

        # Step 3: Encode & yield as stream
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        time.sleep(0.03)
    cap.release()

# 📡 Moderated Live Stream from Camera
def moderated_camera_stream(request):
    cam_url = request.GET.get("url", "0")
    try:
        source = int(cam_url)
    except ValueError:
        source = cam_url  # e.g. IP cam URL

    return StreamingHttpResponse(generate_moderated_stream(source),
        content_type='multipart/x-mixed-replace; boundary=frame')

# 🔘 Selection UI
def live_stream_options(request):
    return render(request, 'live_options.html')