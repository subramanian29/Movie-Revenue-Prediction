from django.urls import path,include
from django.contrib import admin
from .views import movie_form_view, search_titles, search_cast, search_crew, search_director, search_genres, search_keywords,get_movie_details
from .views import handle_movie_form

urlpatterns = [
    path('', movie_form_view, name='home'),
    path('search/director/', search_director, name='search_director'),
    path('search/cast/', search_cast, name='search_cast'),
    path('search/crew/', search_crew, name='search_crew'),
    path('search/genres/', search_genres, name='search_genres'),
    path('search/keywords/', search_keywords, name='search_keywords'),
    path('search/title/', search_titles, name='search_title'),
    path('get_movie_details/', get_movie_details, name='get_movie_details'),
    path('movie-form/', handle_movie_form, name='handle_movie_form'),
]
