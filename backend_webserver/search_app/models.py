from django.template.defaultfilters import slugify
from django.contrib.auth.models import User
from django.urls import reverse
from django.db import models


class BookMetadata(models.Model):
    comic_no = models.CharField(max_length=6, primary_key=True)
    book_title = models.CharField(max_length=50, blank=True)
    volume = models.CharField(max_length=4, null=True, blank=True)
    issue = models.CharField(max_length=4, null=True, blank=True)
    year = models.CharField(max_length=5, null=True, blank=True)
    month = models.CharField(max_length=20, null=True, blank=True)
    genre = models.CharField(max_length=50, null=True, blank=True)
    weblink = models.CharField(max_length=200, null=True, blank=True)

    def __str__(self):
        return self.comic_no + " | " + self.book_title


class Drink(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=500)
