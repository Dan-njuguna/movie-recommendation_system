#!/usr/bin/env python3

"""
AUTHOR: DAN NJUGUNA
DATE: 2025-03-31

This module predicts the movie which a user will likely love.
"""

import sys
import os
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent.parent))
from steps.training import MatrixFactorization
from torch import nn
import torch
import json
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('logs/predict.log')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Constants
MODEL_ARCHITECTURE = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "models", "architecture", "movie_model_architecture.json")
MODEL_PATH = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "models", "movie_model.pth")
MOVIE_TITLES = os.path.join(Path(os.path.abspath(__file__)).parent.parent, "models", "mappings", "movie_title_map.json")
MOVIE_ID_MAP = os.path.join(Path(os.path.abspath(__file__)).parent.parent, "models", "mappings", "movie_mapping.json")

with open(MODEL_ARCHITECTURE, "r") as f:
    model_architecture: Dict = json.load(f)

class Predictor:
    """The class creates prediction for the movie based on the user id."""
    def __init__(self,
        model: nn.Module,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_recommendations(
            self,
            user_id: int,
            movie_id_maps: str = MOVIE_ID_MAP,
            num_recommendations: Optional[int] = 10
    ) -> List[str]:
        """This function creates movie recommendations for the user.
        -----
        Args:
            user_id (int) - The user id for which to make recommendations.
            num_recommendations (int) - The number of recommendations to make.
        -------
        Returns:
            List[str] - The list of recommended movies.
        """
        self.model.eval()
        # Get the user ID as a tensor
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
        # Get all movies ids
        with open(movie_id_maps, 'r') as f:
            movie_ids = json.load(f)
        movie_tensor = torch.tensor(list(movie_ids.values()), dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Get the embeddings for user and all movies
            user_embedding = self.model.user_embedding(user_tensor)
            movie_embeddings = self.model.movie_embedding(movie_tensor)
            # Calculate the predicted ratings
            predicted_ratings = torch.matmul(user_embedding, movie_embeddings.t())
        
        # Getting the top-rated movies
        top_movies = torch.topk(predicted_ratings, num_recommendations).indices.squeeze().tolist()

        # Convert movie indices back to movie IDs
        recommended_movies = [list(movie_ids.keys())[idx] for idx in top_movies]

        return recommended_movies


class PredictorPipeline:
    """This pipeline predicts the movie recommendtaions required and
    returns the movie names list.
    """
    def __init__(
            self,
            user_id: int,
            top_k: int = 5
    ):
        self.model = MatrixFactorization(**model_architecture)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.recommendations = Predictor(self.model).make_recommendations(user_id=user_id, num_recommendations=top_k)
    

    def movie_titles(self, movie_titles: Dict = MOVIE_TITLES):
        with open(movie_titles, "r") as f:
            movie_title_map = json.load(f)

        recommended_movies = []
        for movie_id in self.recommendations:
            if movie_id in movie_title_map:
                recommended_movies.append(movie_title_map[movie_id])
                logger.info(f"Movie ID {movie_id} corresponds to title: {movie_title_map[movie_id]}")
            else:
                logger.warning(f"Movie ID {movie_id} not found in movie title map.")
        return recommended_movies


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python predict.py <user_id>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    top_k = 5

    predictor_pipeline = PredictorPipeline(user_id=user_id, top_k=top_k)
    recommended_movies = predictor_pipeline.movie_titles()
    
    if recommended_movies:
        logger.info(f"Recommended movies for user {user_id}:\n{recommended_movies}")
    else:
        logger.info(f"No recommendations found for user {user_id}.")