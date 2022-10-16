from django.urls import path

# now import the views.py file into this code
from . import views


urlpatterns = [
    path("index/", views.index),
    path("get_book_details/<str:id>", views.get_book_details),
    path("get_coarse_search_results/", views.get_coarse_search_results),
    path("drinks/", views.drink_list),
    path("drinks/<int:id>", views.drink_detail),
]

