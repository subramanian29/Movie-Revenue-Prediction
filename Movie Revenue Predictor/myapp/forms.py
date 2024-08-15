from django import forms
from .models import Movie, Director, Language, Cast, Crew, Genre, Keyword

class MovieForm(forms.ModelForm):
    title = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter movie title', 'id': 'id_title'}),
        required=False
    )
    director = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter director name', 'id': 'id_director'}),
        required=False
    )
    cast = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter cast names, separated by commas', 'id': 'id_cast'}),
        required=False
    )
    crew = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter crew names, separated by commas', 'id': 'id_crew'}),
        required=False
    )
    genres = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter genres, separated by commas', 'id': 'id_genres'}),
        required=False
    )
    keywords = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter keywords, separated by commas', 'id': 'id_keywords'}),
        required=False
    )

    class Meta:
        model = Movie
        fields = [
            'title',
            'director',
            'original_language',
            'cast',
            'crew',
            'genres',
            'keywords',
            'release_date',
            'budget',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['original_language'].queryset = Language.objects.all()
