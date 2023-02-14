#       Movie Recommender System   

#Import the necessary libraries and load the dataset:
import pandas as pd


movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')


#Exploratory Data Analysis:

movies_df.head()


ratings_df.head()

# Finding unique users and movies
unique_users = ratings_df['userId'].nunique()
unique_movies = ratings_df['movieId'].nunique()

print("Number of Unique Users: %d" % unique_users)
print("Number of Unique Movies: %d" % unique_movies)

# Average rating and total movies at genre level
genre_ratings = ratings_df.groupby(by=['movieId'], as_index=False)['rating'].mean()
genre_ratings = pd.merge(genre_ratings, movies_df, on='movieId')
genre_ratings.head()


# Unique genres considered
genre_list = set()
for genres in genre_ratings['genres']:
    genre_list = genre_list.union(set(genres.split('|')))
print("Unique genres: ", genre_list)

# Popularity-Based Recommendation
def popularity_based_recommender(genre, threshold, num_recommendations):
    # Filter movies by genre and threshold
    filtered_movies = genre_ratings[(genre_ratings['genres'].str.contains(genre)) & (genre_ratings['rating'] > threshold)]
    # Order by rating in descending order
    filtered_movies = filtered_movies.sort_values(by='rating', ascending=False)
    # Return top N recommendations
    return filtered_movies.head(num_recommendations)

#get inputs from user
genre = input("Enter genre: ")
threshold = float(input("Enter minimum rating threshold: "))
num_recommendations = int(input("Enter number of recommendations: "))

#Call the function and assign the return value to a variable
result = popularity_based_recommender(genre, threshold, num_recommendations)
result


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Content Based Recommendation
def content_based_recommender(movie_title, num_recommendations):
    if movie_title not in movies_df['title'].tolist():
        return "Movie title not found in the dataset."
    else:
        # Vectorize movie titles
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['title'])
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # Find index of given movie title
        idx = movies_df[movies_df['title'] == movie_title].index[0]
        # Get top N similar movies
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_df.iloc[movie_indices]
    
movie_title = input("Enter movie title: ")
num_recommendations = int(input("Enter number of recommendations: "))

content_based_recommender(movie_title, num_recommendations)


#Collaborative Filtering Recommendation
def collaborative_filtering_recommendation(user_id, threshold, N):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_rated_movies = ratings_df['movieId'].values
    similar_users = ratings_df[ratings_df['movieId'].isin(user_rated_movies)].groupby('userId')['rating'].count().reset_index().sort_values('rating', ascending=False)
    similar_users = similar_users[similar_users['rating']>=threshold]
    similar_users = similar_users['userId'].values
    movies_by_similar_users = ratings_df[ratings_df['userId'].isin(similar_users)]['movieId'].unique()
    most_rated_movies_by_similar_users = ratings_df[ratings_df['movieId'].isin(movies_by_similar_users)].groupby('movieId')['rating'].count().reset_index().sort_values('rating', ascending=False)
    recommendations = most_rated_movies_by_similar_users['movieId'][:N].tolist()
    
    return recommendations


user_id = int(input("Enter user ID: "))
threshold = int(input("Enter threshold for similar users: "))
N = int(input("Enter number of recommendations: "))

recommendations = collaborative_filtering_recommendation(user_id, threshold, N)

# Use the recommendations to get the movie titles
movie_titles = movies_df[movies_df['movieId'].isin(recommendations)]['title']

# Convert movie_titles to a DataFrame
movie_titles_df = pd.DataFrame(movie_titles)



# Display the first N rows of the DataFrame
movie_titles_df.head(N)

import ipywidgets as widgets
from IPython.display import display

# Create widgets for input
genre_input = widgets.Text(description="Genre:")
threshold_input = widgets.FloatSlider(description="Threshold:", min=0.5, max=5, step=0.1)
num_recommendations_input = widgets.IntSlider(description="Number of Recommendations:", min=1, max=20)

# Create a function to handle the input and call the appropriate recommendation function
def handle_submit(sender):
    genre_input = widgets.Dropdown(description="Genre:", 
    options=genre_ratings['genres'].unique())
    threshold = threshold_input.value
    num_recommendations = num_recommendations_input.value
    result = popularity_based_recommender(genre, threshold, num_recommendations)
    if result is not None:
        display(result)
    else:
        print("No movies found for the given genre and threshold.")

# Create a submit button and link it to the function
submit_button = widgets.Button(description="Submit")
submit_button.on_click(handle_submit)

# Display the widgets
display(genre_input, threshold_input, num_recommendations_input, submit_button)


###  Popularity-Based Recommendation Model

# Popularity-Based Recommendation Model
import ipywidgets as widgets
from IPython.display import display

# Create widgets for input
genre_input = widgets.Dropdown(description="Genre:", options=genre_ratings['genres'].unique())
threshold_input = widgets.FloatSlider(description="Threshold:", min=0.5, max=5, step=0.1)
num_recommendations_input = widgets.IntSlider(description="Number of Recommendations:", min=1, max=20)

# Create a function to handle the input and call the appropriate recommendation function
def handle_submit(sender):
    genre = genre_input.value
    threshold = threshold_input.value
    num_recommendations = num_recommendations_input.value
    result = popularity_based_recommender(genre, threshold, num_recommendations)
    if result is not None:
        display(result)
    else:
        print("No movies found for the given genre and threshold.")

# Create a submit button and link it to the function
submit_button = widgets.Button(description="Submit")
submit_button.on_click(handle_submit)

# Display the widgets
display(genre_input, threshold_input, num_recommendations_input, submit_button)


###  Content Based Recommendation Model

# Content Based Recommendation Model
import ipywidgets as widgets
from IPython.display import display

# Create widgets for input
movie_title_input = widgets.Text(description="Movie Title:")
num_recommendations_input = widgets.IntSlider(description="Number of Recommendations:", min=1, max=20)

# Create a function to handle the input and call the appropriate recommendation function
def handle_submit(sender):
    movie_title = movie_title_input.value
    num_recommendations = num_recommendations_input.value
    result = content_based_recommender(movie_title, num_recommendations)
    if isinstance(result, str):
        print(result)
    else:
        display(result)

# Create a submit button and link it to the function
submit_button = widgets.Button(description="Submit")
submit_button.on_click(handle_submit)

# Display the widgets
display(movie_title_input, num_recommendations_input, submit_button)


### Collaborative Filtering Recommendation Model

#Collaborative Filtering Recommendation Model
import ipywidgets as widgets
from IPython.display import display

user_id_input = widgets.Text(description="User ID:")
threshold_input = widgets.FloatSlider(description="Threshold:", min=1, max=1000, step=0.1)
num_recommendations_input = widgets.IntSlider(description="Number of Recommendations:", min=1, max=20)


def handle_submit(sender):
    user_id = int(user_id_input.value)
    threshold = threshold_input.value
    num_recommendations = num_recommendations_input.value
    recommendations = collaborative_filtering_recommendation(user_id, threshold, num_recommendations)
    movie_titles = movies_df[movies_df['movieId'].isin(recommendations)]['title']
    movie_titles_df = pd.DataFrame(movie_titles)
    display(movie_titles_df.head(num_recommendations))

    
submit_button = widgets.Button(description="Submit")
submit_button.on_click(handle_submit)


display(user_id_input, threshold_input, num_recommendations_input, submit_button)








