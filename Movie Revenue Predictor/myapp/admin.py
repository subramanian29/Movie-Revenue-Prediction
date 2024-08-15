
# Register your models here.
from django.contrib import admin
from .models import Movie, Director, Cast, Crew, Genre, Keyword, Language

admin.site.register(Movie)
admin.site.register(Director)
admin.site.register(Cast)
admin.site.register(Crew)
admin.site.register(Genre)
admin.site.register(Keyword)
admin.site.register(Language)
