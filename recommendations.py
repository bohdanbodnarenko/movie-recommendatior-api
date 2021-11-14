from typing import List

import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset, SVD

from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

SPACE_REPLACER = '_'


def make_list_from_names(row) -> List[str]:
    if isinstance(row, list):
        names = [i['name'] for i in row][:3]

        return [str.lower(i.replace(" ", SPACE_REPLACER)) for i in names]

    return []


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(crew):
    for item in crew:
        if item['job'] == 'Director':
            return str.lower(item['name'].replace(" ", SPACE_REPLACER))
    return ''


def create_string_metadata(x) -> str:
    return ' '.join(make_list_from_names(x['keywords'])) + ' ' + ' '.join(make_list_from_names(x['cast'])) + ' ' + x[
        'director'] + ' ' + ' '.join(make_list_from_names(x['genres']))


def get_overall_movies_recommendation(movies_df: pd.DataFrame) -> pd.DataFrame:
    vote_average_mean = movies_df['vote_average'].mean()

    # Only movies that have more votes than another 90%
    minimum_votes_for_chart = movies_df['vote_count'].quantile(0.9)

    # Calculating min amount of votes to be in chart
    movies_with_enough_votes_for_chart = movies_df.copy().loc[movies_df['vote_count'] >= minimum_votes_for_chart]

    def weighted_rating(row: pd.DataFrame, minimum_votes_to_be_listed=minimum_votes_for_chart,
                        votes_average=vote_average_mean):
        vote_count = row['vote_count']
        vote_average = row['vote_average']

        return (vote_count / (vote_count + minimum_votes_to_be_listed) * vote_average) + (
                minimum_votes_to_be_listed / (minimum_votes_to_be_listed + vote_count) * votes_average)

    movies_with_enough_votes_for_chart['score'] = movies_with_enough_votes_for_chart.apply(weighted_rating, axis=1)
    recommendations = movies_with_enough_votes_for_chart.sort_values('score', ascending=False)
    return recommendations


def get_movies_df() -> pd.DataFrame:
    getting_movies_df_start_time = time.time()
    movies_df = pd.read_csv('./datasets/tmdb_5000_movies.csv')
    movie_credits_df = pd.read_csv('./datasets/tmdb_5000_credits.csv')
    movie_credits_df.columns = ['id', 'tittle', 'cast', 'crew']
    merged_df = movies_df.merge(movie_credits_df, on='id')

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        merged_df[feature] = merged_df[feature].apply(literal_eval)

    merged_df['director'] = merged_df['crew'].apply(get_director)

    print("--- Got and formatted movies df in: %s seconds ---" % (time.time() - getting_movies_df_start_time))
    return merged_df


def get_similar_movies_recommendation(movies_df: pd.DataFrame, title: str) -> pd.DataFrame:
    movies_df['metadata'] = movies_df.apply(create_string_metadata, axis=1)

    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(movies_df['metadata'])
    cosine_sim = cosine_similarity(count_matrix)

    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

    # Get the index of the movie that matches the title
    target_movie_id = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = enumerate(cosine_sim[target_movie_id])

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores[
                                   1:]]  # for not including the searching movie (the biggest sim_score)

    return movies_df.iloc[movie_indices]


def get_recommendation_for_user(movies_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    reader = Reader()
    ratings = pd.read_csv('./datasets/ratings_small.csv')
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    svd = SVD()
    train_set = data.build_full_trainset()
    svd.fit(train_set)

    indices = pd.Series(movies_df.index, index=movies_df['id']).drop_duplicates()

    movies_df['est'] = movies_df['id'].apply(lambda x: svd.predict(user_id, indices[x]).est)
    return movies_df.sort_values('est', ascending=False)


if __name__ == '__main__':
    start_time = time.time()
    movies_df = get_movies_df()

    limit: int = 10
    offset: int = 0

    print('Recommendation based only on films data overall: ')
    print(get_overall_movies_recommendation(movies_df)[offset:limit]['title'])
    print()

    movie_title = 'The Dark Knight Rises'
    print('Recommendation for the "%s": ' % movie_title)
    print(get_similar_movies_recommendation(movies_df, movie_title)[offset:limit]['title'])
    print()

    user_id = 1
    print('Recommendation for user with id {%i}: ' % user_id)
    print(get_recommendation_for_user(movies_df, user_id)[offset:limit]['title'])
    print()

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
