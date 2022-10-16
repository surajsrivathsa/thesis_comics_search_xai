from rest_framework import serializers
from .models import BookMetadata, Drink


class DrinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Drink
        fields = ["id", "name", "description"]


class BookMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = BookMetadata
        fields = [
            "comic_no",
            "book_title",
            "volume",
            "issue",
            "year",
            "month",
            "genre",
            "weblink",
        ]
