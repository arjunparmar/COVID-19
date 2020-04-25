from django.db import models
from django_resized import ResizedImageField
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from WEB_APP.settings import BASE_DIR, MEDIA_ROOT
import os

class OverwriteStorage(FileSystemStorage):
    #overwriting image files with same name
    def get_available_name(self, name, max_length = None):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name


class Image(models.Model) :
    uploads = ResizedImageField(size = [500,500], upload_to = 'uploaded_images/' , storage = OverwriteStorage())
    
    # def __str__(self):
    #     return str(self.category)