FROM ubuntu:latest

LABEL maintainer="bodya.bodnarenko@gmail.com"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential vim

COPY . /movie-recommender
WORKDIR /movie-recommender

EXPOSE 8003

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
ENTRYPOINT gunicorn -w 4 app:app -b 0.0.0.0:8003
