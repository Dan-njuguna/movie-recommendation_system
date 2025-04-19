# MOVIE RECOMMENDATION SYSTEM WITH MATRIX FACTORIZAATION

This documentation will not carry the heavy lifting for why I opted for Matrix Factorization formula, however, it is important to highlight that the recommender system I have created is `collaborative filtering` model. It gets the similarity in user trends and recommends the movies based on the trends. Further, it is important to note that the `IMDB Movie Dataset` is publicly available on Kaggle. Just check it out!

## Quick Guide for Consuming this recommender system
**NOTE:** Ensure you have `conda` or `uv` installed. However, in my local machine, I used  `conda`. That is in event that you wish to run the recommender system locally and not on Docker.

### OPTION A: Using Docker container
1. Clone this repository and navigate to the folder downloaded to your local machine.
```bash
git clone https://github.com/Dan-njuguna/movie-recommendation_system.git && cd movie-recommendation_system
```

2. Build the docker container
```bash
docker compose build
```

3. Run the docker container(either in `detached mode` or otherwise - **one-off** mode)
```bash
docker compose up -d # I prefer detached mode btw😄
```

4. _OPTIONALLY_: You can check the logs for our running image, live logs as the app runs
```bash
docker logs movie-recommender-api:latest -f
```

5. Check the webview of the recommender system here: [webview](http://localhost:7999/docs). Or else, `http://localhost:7999/docs`

#### TO CLOSE DOWN THE CONTAINER IF NO LONGER IN USE
1. Run the following command to stop and remove the container if you no longer need it.
```bash
docker stop movie-recommender-api && docker rm movie-recommender-api && docker rmi movie-recommender-api
```

### Option B: Using your local environment

1. Create a virtual environment and activate it but most importantly ensure you have cloned the repo and navigated to its folder.
```bash
conda create --name rec python=3.11 && conda activate rec # Ensure this runs completely, you should have stable internet connection.
```

2. Install packages required to run the model and the api.
```bash
pip install -r requirements.txt
```

3. Run your now ready API.
```bash
uvicorn --reload --port 8000 main:app
```

4. In this case, visit the path [web-view](http://localhost:8000/docs), or else: `http://localhost:8000/docs`