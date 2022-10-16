from django.shortcuts import render
import os, sys
import pandas as pd, numpy as np
import re
from django.http import JsonResponse
from .models import BookMetadata, Drink
from .serializers import BookMetadataSerializer, DrinkSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# custom imports
import common_functions.backend_utils as utils
from search.coarse.coarse_search import comics_coarse_search

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()

# # create book instances once in program
# model_instances = [
#     BookMetadata(
#         comic_no=str(int(record["comic_no"])),
#         book_title=record["Book Title"],
#         volume=record["Vol"],
#         issue=record["Issue"],
#         year=record["Year"],
#         month=record["Month"],
#         genre=record["genre"],
#         weblink=record["link"],
#     )
#     for record in comic_book_metadata_df.to_dict("records")
# ]

# model_lst = BookMetadata.objects.bulk_create(model_instances)
# print("Created: {} books".format(len(model_lst)))

# Create your views here.

from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello Geeks")


def display_metadata(request):

    return


@api_view(["GET"])
def get_book_details(request, id, format=None):
    # all_books = BookMetadata.objects.all()
    # all_books_drinks = BookMetadataSerializer(all_books, many=True)
    try:

        book = BookMetadata.objects.all().get(pk=id)
        book_serializer = BookMetadataSerializer(book)
        return Response(book_serializer.data)
    except:
        return Response(status=status.HTTP_404_NOT_FOUND)


@api_view(["POST"])
def get_coarse_search_results(request, format=None):
    print("vvvvv")
    try:
        print(request.data, type(request.data))
        query_book_comic_id = int(request.data["id"])
        top_n = min(request.data["top_n"], 21)
        feature_weight_dict = {
            "cld": request.data["cld_weight"],
            "edh": request.data["edh_weight"],
            "hog": request.data["hog_weight"],
            "text": request.data["text_weight"],
        }
        print(feature_weight_dict)
        print(
            "query book info: {}".format(book_metadata_dict[query_book_comic_id - 3451])
        )
        top_n_results_df = comics_coarse_search(
            query_book_comic_id, feature_weight_dict=feature_weight_dict, top_n=top_n
        )
        print(top_n_results_df)
        return Response(top_n_results_df.to_json())
    except:
        print(traceback.format_exc())
        return Response(status=status.HTTP_404_NOT_FOUND)


@api_view(["GET", "POST"])
def drink_list(request, format=None):
    """
    get all drinks
    serialize them
    return json

    """

    if request.method == "GET":

        all_drinks = Drink.objects.all()
        all_serialized_drinks = DrinkSerializer(all_drinks, many=True)

        return Response(all_serialized_drinks.data, safe=False)

    if request.method == "POST":
        serialized_drink = DrinkSerializer(data=request.data)

        if serialized_drink.is_valid():
            serialized_drink.save()
            return Response(serialized_drink.data, status=status.HTTP_201_CREATED)


@api_view(["GET", "PUT", "DELETE"])
def drink_detail(request, id, format=None):

    try:
        drink = Drink.objects.get(pk=id)
    except:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        drink_serializer = DrinkSerializer(drink)
        return Response(drink_serializer.data)

    elif request.method == "PUT":
        serializer = DrinkSerializer(drink, data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "DELETE":
        drink.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    else:
        pass

