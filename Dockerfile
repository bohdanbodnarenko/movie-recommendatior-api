FROM python:3.9.1

ADD . /movie-recommender
WORKDIR /movie-recommender

RUN pip install -r requirements.txt

