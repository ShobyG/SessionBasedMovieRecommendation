import pandas as pd
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectMultipleField

from model import Model
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow as tf


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

movies_df =  pd.read_csv("ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"], encoding='latin-1')


"""
Randomly select 15 movies for the user to make initial selection
"""
def select_movies():
    selected_movies = movies_df.sample(n=15)
    return selected_movies.to_dict(orient='records')

class MovieForm(FlaskForm):
    movie_selection = SelectMultipleField('Select Movies', coerce=int)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MovieForm()

    # Populate the choices for the SelectMultipleField
    selected_movies = select_movies()
    form.movie_selection.choices = [(movie['movie_id'], f"{movie['title']} - {movie['genres']}") for movie in selected_movies]
    if request.method == "POST":
        print("reached")
        selected_movie_ids = form.movie_selection.data
        loaded_model = Model()
        x, movies = loaded_model.get_movies()
        brute_force = tfrs.layers.factorized_top_k.BruteForce(loaded_model.candidate_model)
        brute_force.index_from_dataset(
            movies.batch(128).map(lambda movie_id: (movie_id, loaded_model.candidate_model(movie_id)))
        )
        selected_movie_ids_str = [str(item) for item in selected_movie_ids]
        test_array = tf.constant(selected_movie_ids_str, dtype=tf.string)
        _, titles = brute_force(test_array, k=4)
        choices = []
        for i in titles:
            for j in i:
                numeric_value = tf.strings.to_number(j, out_type=tf.int32).numpy()
                if numeric_value not in selected_movie_ids and numeric_value not in choices:
                    choices.append(numeric_value)
        if len(choices) > 10:
            choices = choices[:10]
        recommended_movies = []
        user_choices = []
        for id in choices:
            recommended_movie_name = str(movies_df.loc[movies_df['movie_id'] == id]['title'].iloc[0])
            recommended_movie_genre = str(movies_df.loc[movies_df['movie_id'] == id]['genres'].iloc[0])
            recommended_movies.append((recommended_movie_name,recommended_movie_genre))
        for id in selected_movie_ids:
            selected_movie_name = str(movies_df.loc[movies_df['movie_id'] == id]['title'].iloc[0])
            selected_movie_genre = str(movies_df.loc[movies_df['movie_id'] == id]['genres'].iloc[0])
            user_choices.append((selected_movie_name, selected_movie_genre))
        print(user_choices)
        print(recommended_movies)
        return render_template('selected_movies.html', selected_movies=user_choices, recommended= recommended_movies)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

