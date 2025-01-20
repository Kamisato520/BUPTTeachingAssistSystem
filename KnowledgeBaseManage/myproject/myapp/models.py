import uuid

from django.db import models


# Create your models here.
class Knowledge(models.Model):
    title = models.TextField()  # 标题
    content = models.TextField()  # 文本字段
    created_at = models.DateTimeField(auto_now_add=True)  # 时间戳字段
    vector_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
