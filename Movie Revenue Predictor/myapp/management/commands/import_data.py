import csv
from django.core.management.base import BaseCommand
from myapp.models import Director, Language, Movie, Cast, Crew, Genre, Keyword
import ast
class Command(BaseCommand):
    help = 'Import data from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                director, _ = Director.objects.get_or_create(name=row['director'])
                original_language, _ = Language.objects.get_or_create(name=row['original_language'])
                cast_members=[]
                for i in ast.literal_eval(row['cast']):
                    cast,_= Cast.objects.get_or_create(name=i)
                    cast_members.append(cast)

                crew_members=[]
                for i in ast.literal_eval(row['crew']):
                    crew,_= Crew.objects.get_or_create(name=i)
                    crew_members.append(crew)
                if (row['title']=='Avatar'):
                    print(len(crew_members))
                
                genres = []
                for genre_name in ast.literal_eval(row['genres']):
                    genre, _ = Genre.objects.get_or_create(name=genre_name)
                    genres.append(genre)

            
                keywords = []
                for keyword_name in ast.literal_eval(row['keywords']):
                    keyword, _ = Keyword.objects.get_or_create(name=keyword_name)
                    keywords.append(keyword)

                release_date = row['release_date']

                budget = row['budget']

                movie,created=Movie.objects.get_or_create(
                    title=row['title'],
                    director=director,
                    original_language=original_language,
                    release_date=release_date,
                    budget=budget
                )
                movie.cast.set(cast_members)
                movie.crew.set(crew_members)
                movie.genres.set(genres)
                movie.keywords.set(keywords)

                movie.save()
        self.stdout.write(self.style.SUCCESS('Data imported successfully'))
