from django.db import models

class Video(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='videos/')  # Stores video in 'media/videos/'
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
