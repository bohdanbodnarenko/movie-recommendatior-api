# movie-recommendation-api

A simple api for recommending movies based on votes all the time, movie similarity or by user rates

## Before you start

You **need** to have docker or python (v3.2 or higher) installed on the machine on which app will run

## Run Application

For installing all dependencies simply run next command:

```shell
$ pip install -r requirements.txt
```

For starting the application:

```shell
$ python app.py
```

Running App in Docker For running our app inside the docker container we have to build our image firstly:

```shell
$ make build
```

Now we can run it with the command:

```shell
$ make run
```

You can stop the container by Docker CLI or Docker UI, or just run:

```shell
$ make kill
```

Otherwise, you can run it with `docker-compose` by running:

```shell
$ docker-compose up
```

## Verify that app works correctly:

Once you are done with starting the app by any way described above, you can go to http://0.0.0.0:8003/movie and check
that app works fine.

### Possible query parameters:

* `limit` - the amount of movies to be returned (`100` maximum)
* `skip` - how many movies with such query should app skip in returning (`0` by default), made for pagination
* `title` - returns the list of recommendations based on the film title you liked
* `userId` - id of particular user movies will be recommended for (based on [this dataset](/datasets/ratings_small.csv))

## Data
All datasets used for this app are located [here](datasets/)