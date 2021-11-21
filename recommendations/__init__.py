from typing import List, Mapping, Optional
from functools import partial

import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset, SVD

from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

from recommendations.constants import SPACE_REPLACER, RESULT_COLUMNS, DEFAULT_MOVIES_LIMIT


class MovieRecommender:
    def __init__(self, movies_ds_path: str, credits_ds_path: str, ratings_ds_path: str):
        getting_movies_df_start_time = time.time()
        movies_df = pd.read_csv(movies_ds_path)
        movie_credits_df = pd.read_csv(credits_ds_path)
        movie_credits_df.columns = ['id', 'tittle', 'cast', 'crew']
        merged_df = movies_df.merge(movie_credits_df, on='id')

        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            merged_df[feature] = merged_df[feature].apply(literal_eval)

        merged_df['director'] = merged_df['crew'].apply(self.get_director)
        self.__movies_df = merged_df
        self.__svd = self._get_trained_user_svd(ratings_ds_path)
        self.__movie_indices = pd.Series(self.__movies_df.index, index=self.__movies_df['id']).drop_duplicates()

        print("--- Got and formatted movies df in: %s seconds ---" % (time.time() - getting_movies_df_start_time))

    @staticmethod
    def _get_trained_user_svd(ratings_ds_path: str) -> SVD:
        reader = Reader()
        ratings = pd.read_csv(ratings_ds_path)
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

        svd = SVD()
        train_set = data.build_full_trainset()
        svd.fit(train_set)

        return svd

    @staticmethod
    def _make_list_from_names(row) -> List[str]:
        if isinstance(row, list):
            names = [i['name'] for i in row][:3]

            return [str.lower(i.replace(" ", SPACE_REPLACER)) for i in names]

        return []

    @staticmethod
    def get_director(crew: List[Mapping]) -> str:
        for item in crew:
            if item['job'] == 'Director':
                return str.lower(item['name'].replace(" ", SPACE_REPLACER))
        return ''

    @staticmethod
    def _create_string_metadata(x: pd.DataFrame) -> str:
        return ' '.join(MovieRecommender._make_list_from_names(x['keywords'])) + ' ' + ' '.join(
            MovieRecommender._make_list_from_names(x['cast'])) + ' ' + \
               x[
                   'director'] + ' ' + ' '.join(MovieRecommender._make_list_from_names(x['genres']))

    @staticmethod
    def _weighted_rating(minimum_votes_to_be_listed: float,
                         votes_average: float, row: pd.DataFrame) -> float:
        vote_count = row['vote_count']
        vote_average = row['vote_average']

        return (vote_count / (vote_count + minimum_votes_to_be_listed) * vote_average) + (
                minimum_votes_to_be_listed / (minimum_votes_to_be_listed + vote_count) * votes_average)

    @staticmethod
    def _format_recommendation_result(result_df: pd.DataFrame, skip: int, limit: int) -> Mapping:
        return result_df[RESULT_COLUMNS][skip:limit].to_dict('records')

    def get_overall_movies_recommendation(self) -> Mapping:
        vote_average_mean = self.__movies_df['vote_average'].mean()

        # Only movies that have more votes than another 90%
        minimum_votes_for_chart = self.__movies_df['vote_count'].quantile(0.9)

        # Calculating min amount of votes to be in chart
        movies_with_enough_votes_for_chart = self.__movies_df.copy().loc[
            self.__movies_df['vote_count'] >= minimum_votes_for_chart]

        bound_function = partial(self._weighted_rating, minimum_votes_for_chart, vote_average_mean)
        movies_with_enough_votes_for_chart['score'] = movies_with_enough_votes_for_chart.apply(bound_function,
                                                                                               axis=1)
        recommendations = movies_with_enough_votes_for_chart.sort_values('score', ascending=False)
        return recommendations

    def get_similar_movies_recommendation(self, title: str) -> pd.DataFrame:
        self.__movies_df['metadata'] = self.__movies_df.apply(self._create_string_metadata, axis=1)

        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(self.__movies_df['metadata'])
        cosine_sim = cosine_similarity(count_matrix)

        indices = pd.Series(self.__movies_df.index, index=self.__movies_df['title']).drop_duplicates()

        # Get the index of the movie that matches the title
        target_movie_id = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = enumerate(cosine_sim[target_movie_id])

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores[
                                       1:]]  # for not including the searching movie (the biggest sim_score)

        return self.__movies_df.iloc[movie_indices]

    def get_recommendation_for_user(self, user_id: int) -> pd.DataFrame:

        self.__movies_df['est'] = self.__movies_df['id'].apply(
            lambda x: self.__svd.predict(user_id, self.__movie_indices[x]).est)
        return self.__movies_df.sort_values('est', ascending=False)

    def recommend(self, movie_title: str, user_id: Optional[str], skip: Optional[str], limit: Optional[str]) -> Mapping:
        normalised_skip = int(skip) if skip else 0
        normalised_limit = (limit if int(limit) <= 100 else 100) if limit else DEFAULT_MOVIES_LIMIT

        if user_id:
            result = self.get_recommendation_for_user(int(user_id))
        elif movie_title:
            result = self.get_similar_movies_recommendation(movie_title.capitalize())
        else:
            result = self.get_overall_movies_recommendation()

        return self._format_recommendation_result(result, normalised_skip, normalised_limit)
