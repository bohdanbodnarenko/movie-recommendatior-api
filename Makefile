app_name = movie_recommender

build:
	@docker build -t $(app_name) .

run:
	@docker run --detach -p 8003:8003 $(app_name)
	@echo 'App is running on http://0.0.0.0:8003'

kill:
	@echo 'Killing container...'
	@docker ps | grep $(app_name) | awk '{print $$1}' | xargs docker kill