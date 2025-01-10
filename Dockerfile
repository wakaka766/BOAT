FROM readthedocs/build:latest

RUN apt-get update && apt-get install -y python3.9-dev
