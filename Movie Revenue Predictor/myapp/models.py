# models.py
from django.db import models

class Director(models.Model):
    name = models.CharField(max_length=255)

class Cast(models.Model):
    name = models.CharField(max_length=4000)

class Crew(models.Model):
    name = models.CharField(max_length=4000)

class Genre(models.Model):
    name = models.CharField(max_length=255)

class Keyword(models.Model):
    name = models.CharField(max_length=255)

class Language(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class Movie(models.Model):
    title = models.CharField(max_length=255)
    director = models.ForeignKey(Director, on_delete=models.CASCADE)
    cast = models.ManyToManyField(Cast)
    crew = models.ManyToManyField(Crew)
    genres = models.ManyToManyField(Genre)
    keywords = models.ManyToManyField(Keyword)
    original_language = models.ForeignKey(Language, on_delete=models.CASCADE)
    release_date = models.DateField()
    budget = models.DecimalField(max_digits=15, decimal_places=2)
