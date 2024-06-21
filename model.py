import tensorflow_recommenders as tfrs
import numpy as np
import tensorflow as tf
import pandas as pd


class Model(tfrs.Model):

    def __init__(self,):
        super().__init__()
        self.embedding_dimension = 32
        self.unique_movie_ids, self.movies = self.get_movies()
        self.query_model = self.query_model()
        self.candidate_model = self.candidate_model()
        self.task = self.get_task()

    @staticmethod
    def get_movies():
        def map_function(movie_id):
            # Convert the input to byte string if it's not already
            return tf.strings.as_string(tf.strings.to_number(movie_id, out_type=tf.int32))

        movies_df = pd.read_csv("ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"], encoding='latin-1')
        dataset = tf.data.Dataset.from_tensor_slices(movies_df['movie_id'].values.astype(np.bytes_))
        mapped_dataset = dataset.map(map_function)
        movie_ids = mapped_dataset.batch(1_000)
        return np.unique(np.concatenate(list(movie_ids))), mapped_dataset

    def query_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_movie_ids) + 1, self.embedding_dimension),
            tf.keras.layers.GRU(self.embedding_dimension),
        ])

    def candidate_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_movie_ids) + 1, self.embedding_dimension)
        ])

    def get_task(self):
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=self.movies.batch(128).map(self.candidate_model)
        )
        return tfrs.tasks.Retrieval(
            metrics=metrics
        )

    def compute_loss(self, features, training=False):
        watch_history = features["synthetic_session_movie_id"]
        watch_next_label = features["label_movie_id"]

        query_embedding = self.query_model(watch_history)
        candidate_embedding = self.candidate_model(watch_next_label)

        return self.task(query_embedding, candidate_embedding, compute_metrics=not training)


if __name__=="__main__":
    loaded_model = Model()
    x, movies = loaded_model.get_movies()
    brute_force = tfrs.layers.factorized_top_k.BruteForce(loaded_model.candidate_model)
    brute_force.index_from_dataset(
        movies.batch(128).map(lambda movie_id: (movie_id, loaded_model.candidate_model(movie_id)))
    )
    # data = np.array([
    #     [b'2402', b'2404', b'2815', b'1031', b'3142', b'1951', b'231', b'265', b'1355', '1363'],
    #     # Add more rows as needed
    # ])
    data = ['2008', '2839']

    test_array = tf.constant(data, dtype=tf.string)
    _, titles = brute_force(test_array, k=10)
    print(titles)


