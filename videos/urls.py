from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import home, video_list, upload_video, preprocess_video,moderated_frames_view  # ✅ Add preprocess_video
from . import views
from . import live_stream

urlpatterns = [
    path('', home, name='home'),
    path('videos/', views.video_list, name='videolist'),
    path('upload/', views.upload_video, name='upload'),
    path('preprocess/<int:video_id>/', preprocess_video, name='preprocess'),
    path('moderated_frames/<int:video_id>/', moderated_frames_view, name='moderated_frames'),
    path('live/', live_stream.live_stream_list, name='live_stream_list'),
    path('live/stream/<int:video_id>/', live_stream.live_stream, name='live_stream'),
    path('live/camera/', live_stream.live_camera_stream, name='live_camera_stream'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)