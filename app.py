import os

from flask import Flask, jsonify, request, abort
from flask_cors import CORS

from recommendations import MovieRecommender

app = Flask(__name__)
CORS(app)


def get_dataset_abs_path(dataset: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'datasets', dataset)


recommender = MovieRecommender(get_dataset_abs_path('tmdb_5000_movies.csv'),
                               get_dataset_abs_path('tmdb_5000_credits.csv'),
                               get_dataset_abs_path('ratings_small.csv'))


@app.route('/movie', methods=['GET'])
def recommend_movies():
    title = request.args.get('title')
    user_id = request.args.get('user_id')
    skip = request.args.get('skip')
    limit = request.args.get('limit')

    try:
        res = recommender.recommend(movie_title=title, user_id=user_id, skip=skip,
                                    limit=limit)
        return jsonify(res)
    except ValueError:
        abort(400)
    except KeyError:
        abort(404)


@app.route('/')
def base_route():
    return 'Go to /movie for check the app'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
