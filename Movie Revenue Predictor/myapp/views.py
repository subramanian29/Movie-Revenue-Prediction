from django.shortcuts import render, get_object_or_404

from .models import Movie, Cast, Crew, Genre, Keyword, Director
from .forms import MovieForm
from django.http import JsonResponse
import pickle
import torch 
from torch import nn 
import pandas as pd
import os

def movie_form_view(request):
    form = MovieForm()
    if request.method == 'POST':
        form = MovieForm(request.POST)
        if form.is_valid():
            form.save()
    return render(request, 'movie_form.html', {'form': form})

from django.http import JsonResponse
from .models import Movie, Director
from django.db.models import Q

def search_titles(request):
    term = request.GET.get('term', '')
    results = Movie.objects.filter(title__icontains=term).values_list('title', flat=True)
    return JsonResponse(list(results), safe=False)


def search_director(request):
    if request.is_ajax() and request.method == "GET":
        query = request.GET.get('term', '')
        results = list(Director.objects.filter(name__icontains=query).values_list('name', flat=True))
        return JsonResponse(results, safe=False)

def search_cast(request):
    term = request.GET.get('term', '')
    results = list(Cast.objects.filter(name__icontains=term).values_list('name', flat=True))
    return JsonResponse(results, safe=False)

def search_crew(request):
    term = request.GET.get('term', '')
    results = list(Crew.objects.filter(name__icontains=term).values_list('name', flat=True))
    return JsonResponse(results, safe=False)

def search_genres(request):
    term = request.GET.get('term', '')
    results = list(Genre.objects.filter(name__icontains=term).values_list('name', flat=True))
    return JsonResponse(results, safe=False)

def search_keywords(request):
    term = request.GET.get('term', '')
    results = list(Keyword.objects.filter(name__icontains=term).values_list('name', flat=True))
    return JsonResponse(results, safe=False)






def submit_movie_form(request):
    if request.method == 'POST':
        print("hi")
        form = MovieForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            
            print(data)
            return render(request, 'success.html', {'movie_data': data})

    else:
        form = MovieForm()

    return render(request, 'movie_form.html', {'form': form})

def get_movie_details(request):
    title = request.GET.get('title')
    try:
        movie = Movie.objects.get(title=title)
        data = {
            'title': movie.title,
            'director': movie.director.name,
            'cast': [cast.name for cast in movie.cast.all()],
            'crew': [crew.name for crew in movie.crew.all()],
            'genres': [genre.name for genre in movie.genres.all()],
            'keywords': [keyword.name for keyword in movie.keywords.all()],
            'original_language': movie.original_language.name,
            'release_date': movie.release_date,
            'budget': movie.budget,
        }
        return JsonResponse(data)
    except Movie.DoesNotExist:
        return JsonResponse({'error': 'Movie not found'}, status=404)


def input_to_tensor(X):
  title=X["title"]
  budget=X["budget"]
  genres=X["genres"]
  keywords=X["keywords"]
  language=X["original_language"]
  release_date=X["release_date"]
  cast=X["cast"]
  crew=X["crew"]
  director=X["director"]
  SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scalers.pkl')
  EN_PATH=os.path.join(os.path.dirname(__file__), 'encoders.pkl')
  IDS_PATH=os.path.join(os.path.dirname(__file__), 'ids.pkl')



  scalers=pickle.load(open(SCALER_PATH,"rb"))
  encoders=pickle.load(open(EN_PATH,"rb"))
  ids=pickle.load(open(IDS_PATH,"rb"))

  title_vectorizer=encoders["title_vectorizer"]
  title_scaler=scalers["title_scaler"]
  title_vector = title_vectorizer.transform([title]).toarray()
  title_vector=title_scaler.transform(title_vector)
  title_tensor = torch.tensor(title_vector, dtype=torch.float32)

  budget_vector=scalers["budget_scaler"].transform([[budget]])
  budget_tensor=torch.tensor(budget_vector,dtype=torch.float32)

  genre_encoder=encoders["genre_encoder"]
  genre_scaler=scalers["genre_scaler"]
  genres_vector=genre_encoder.transform([genres])
  genres_vector=genre_scaler.transform(genres_vector)
  genres_tensor=torch.tensor(genres_vector,dtype=torch.float32)
  
  keyword_id=ids["keyword_id"]

  keywords_vector=pd.Series(sorted(list(set([keyword_id[i] for i in keywords]))) +[0 for i in range(max(0,97-len(keywords)))])
  k_tens=torch.tensor(keywords_vector)
  keywords_vector=scalers["keywords_scaler"].transform([k_tens.numpy()])
  keywords_tensor=torch.tensor(keywords_vector,dtype=torch.float32)
 
  language_encoder=encoders["language_encoder"]
  language_scaler=scalers["language_scaler"]
  language_vector=language_encoder.transform([language])
  language_vector=language_scaler.transform(language_vector.reshape(-1,1))
  language_tensor=torch.tensor(language_vector,dtype=torch.float32)


  dayofyear=pd.to_datetime(release_date).dayofyear
  month=pd.to_datetime(release_date).month
  dayofmonth=pd.to_datetime(release_date).day
  dayofweek=pd.to_datetime(release_date).weekday()
  year=pd.to_datetime(release_date).year
  dayofyear_tensor=torch.tensor(scalers["day_of_week_scaler"].transform([[dayofyear]]),dtype=torch.float32)
  month_tensor=torch.tensor(scalers["month_scaler"].transform([[month]]),dtype=torch.float32)
  dayofmonth_tensor=torch.tensor(scalers["day_of_month_scaler"].transform([[dayofmonth]]),dtype=torch.float32)
  dayofweek_tensor=torch.tensor(scalers["day_of_week_scaler"].transform([[dayofweek]]),dtype=torch.float32)
  year_tensor=torch.tensor(scalers["year_scaler"].transform([[year]]),dtype=torch.float32)
  
  cast_id=ids["cast_id"]
  cast_vector=pd.Series(sorted(list(set([cast_id[i] for i in cast]))) +[0 for i in range(max(0,224-len(cast)))])
  c_tens=torch.tensor(cast_vector)
  cast_vector=scalers["cast_scaler"].transform([c_tens.numpy()])
  cast_tensor=torch.tensor(cast_vector,dtype=torch.float32)
  
  crew_id=ids["crew_id"]
  crew=list(set([crew_id[i] for i in crew]))
  crew_vector=pd.Series(sorted(crew) +[0 for i in range(max(0,428-len(crew)))])
  cr_tens=torch.tensor(crew_vector)
  crew_vector=scalers["crew_scaler"].transform([cr_tens.numpy()])
  crew_tensor=torch.tensor(crew_vector,dtype=torch.float32)
  
  director_encoder=encoders["director_encoder"]
  dir_scaler=scalers["dir_scaler"]
  dir_vector=director_encoder.transform([director])
  dir_vector=dir_scaler.transform(dir_vector.reshape(-1,1))
  director_tensor=torch.tensor(dir_vector,dtype=torch.float32)

  T=[budget_tensor,dayofyear_tensor,month_tensor, dayofmonth_tensor,dayofweek_tensor,year_tensor]
  T+=[title_tensor,director_tensor,language_tensor,genres_tensor,cast_tensor,crew_tensor,keywords_tensor]

  for i in T:
    if (len(i.shape))==1:
      i=i.unsqueeze(1)

  F=torch.cat(T,dim=1)
  print(F.shape)
  return F

def predict(X):
    import ast
    for i in X:
        try:
            X[i]=ast.literal_eval(X)
        except:
            pass
    X["budget"]=float(X["budget"])
    
    



    SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scalers.pkl')
    MODEL_PATH=os.path.join(os.path.dirname(__file__), 'movie_revenue.pth')
    rev_scaler=pickle.load(open(SCALER_PATH,"rb"))["rev_scaler"]
    class NeuralNetwork3(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq=nn.Sequential(
                nn.Linear(in_features=1277,out_features=1600),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1600,out_features=800),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=800,out_features=400),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=400,out_features=100),
                nn.LeakyReLU(),
                nn.Linear(in_features=100,out_features=1)
            )
        def forward(self,x):
            return self.seq(x)
    model=NeuralNetwork3()
    model.load_state_dict(torch.load(MODEL_PATH))
    X=input_to_tensor(X)

    model.eval()
    with torch.inference_mode():
        y_pred=model(X).squeeze()
        return rev_scaler.inverse_transform(y_pred.reshape(-1,1))[0][0]


def handle_movie_form(request):
    import json
    if request.method == 'POST':
        try:
            # Load the JSON data from the POST request
            data = json.loads(request.POST.get('data', '{}'))
            
            # Perform prediction
            predicted_revenue = predict(data)  # Assuming `predict` function returns the predicted revenue
            
            # Extract movie title from the data
            movie_title = data.get('title', '')
            
            # Initialize actual_revenue
            actual_revenue = None
            
            # Read the CSV file to find actual revenue
        
            DAT_PATH = os.path.join(os.path.dirname(__file__), 'movie_data.csv')
            df = pd.read_csv(DAT_PATH)
            
            # Search for the movie in the CSV file
            if not df.empty and movie_title:
                movie_row = df[df['title'].str.contains(movie_title, case=False, na=False)]
                if not movie_row.empty:
                    actual_revenue = float(movie_row.iloc[0]['revenue'])  # Adjust 'revenue' to your column name
        
            # Return the prediction and actual revenue in the response
            return JsonResponse({'status': 'success', 'predicted_revenue': predicted_revenue, 'actual_revenue': actual_revenue})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            # Handle other potential exceptions
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

