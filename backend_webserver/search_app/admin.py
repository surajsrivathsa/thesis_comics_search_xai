from django.contrib import admin
from .models import BookMetadata, Drink

# Register your models here.
admin.site.register(BookMetadata)
admin.site.register(Drink)
