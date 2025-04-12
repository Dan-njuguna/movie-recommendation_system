#!/usr/bin/env python3

"""
AUTHOR: DAN NJUGUNA
DATE: 2025-04-03
"""

import os
import sys
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn as nn
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.abspath(__file__), ".."))

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("logs/training.log")
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = data['userId'].values
        self.movies = data['movieId'].values
        self.ratings = data['rating'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(movie, dtype=torch.long), torch.tensor(rating, dtype=torch.float32)


class MatrixFactorization(nn.Module):
    """Matrix Factorization model for collaborative filtering recommendation systems.

    This class implements a neural network-based matrix factorization model that learns
    latent representations of users and movies for rating prediction.

    Args:
        num_users (int): Total number of unique users in the dataset
        num_movies (int): Total number of unique movies in the dataset  
        embedding_size (int): Size of the embedding vectors for users and movies

    Attributes:
        user_embedding (nn.Embedding): Embedding layer for users
        movie_embedding (nn.Embedding): Embedding layer for movies
        fc (nn.Linear): Fully connected layer for final prediction

    Methods:
        forward(user, movie): Forward pass of the model

            Args:
                user (torch.Tensor): Tensor of user IDs
                movie (torch.Tensor): Tensor of movie IDs
                
            Returns:
                torch.Tensor: Predicted ratings for the user-movie pairs
    """
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users 
        self.num_movies = num_movies
        self.embedding_size = embedding_size

        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    

    def forward(
        self,
        user: torch.Tensor,
        movie: torch.Tensor
    ):
        """Forward pass of the model.

        Args:
            user (torch.Tensor): Tensor of user IDs
            movie (torch.Tensor): Tensor of movie IDs
            
        Returns:
            torch.Tensor: Predicted ratings for the user-movie pairs
        """
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        x = torch.cat([user_embed, movie_embed], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save_model_and_architecture(
        self,
    ) -> bool:
        """This function returns the model architecture.

        Args:
            num_users (int): Total number of unique users in the dataset
            num_movies (int): Total number of unique movies in the dataset  
            embedding_size (int): Size of the embedding vectors for users and movies

        Returns:
            bool - True if the model architecture is saved successfully, False otherwise.
        """
        try:
            logger.info("Saving model and architecture ...")
            MODEL_PATH = os.path.join(
                Path(os.path.dirname(os.path.abspath(__file__))).parent, "models"
            )
            logger.info("Saving model and architecture to %s", MODEL_PATH)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(self.state_dict(), f"{MODEL_PATH}/movie_model.pth")
            model_architecture = {
                "num_users": self.num_users,
                "num_movies": self.num_movies,
                "embedding_size": self.embedding_size,
            }
            # Path to save model architecture in
            path = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "models", "architecture")
            os.makedirs(path, exist_ok=True)

            with open(f"{path}/movie_model_architecture.json", "w") as f:
                json.dump(model_architecture, f)

            return True

        except Exception as e:
            logger.error(f"Error saving model and architecture: {e}")
            return False


class Trainer:
    """This class implements the training of the model.
    -----
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train(self, num_epochs: int = 10):
        """This function trains the model.
        -----
        Args:
            num_epochs (int): Number of epochs to train the model for.
        -------
        Returns:
            nn.Module - The trained model.
        """
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(
                self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}',
                colour="green", mininterval=0.3, miniters=0.3
            )
            for user, movie, rating in pbar:
                self.optimizer.zero_grad()
                outputs = self.model(user, movie)
                loss = self.criterion(outputs, rating.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/len(self.train_loader)})
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}")
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user, movie, rating in self.test_loader:
                outputs = self.model(user, movie)
                loss = self.criterion(outputs, rating.view(-1, 1))
                test_loss += loss.item()
        mse = test_loss/len(self.test_loader)
        logger.info(f"Test Loss (MSE): {mse}")
        logger.info(f"Root Mean Square Error (RMSE): {mse ** 0.5}")
        
        return self.model

def main(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_users: int,
    num_movies: int,
    embedding_size: int = 50,
    epochs: int = 10
):
    """Main function to train the model.
    -----
    Args:
    epochs (int): Number of epochs to train the model for.
    -------
    Returns:
    None
    """
    logger.info("Initializing model with parameters:")
    logger.info(f"Number of users: {num_users}")
    logger.info(f"Number of movies: {num_movies}")
    logger.info(f"Embedding size: {embedding_size}")
    
    # Initialize the model
    model = MatrixFactorization(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=embedding_size
    )
    logger.info("Model initialized successfully")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    logger.info("Using MSELoss and SGD optimizer with lr=0.01, momentum=0.9")
    
    # Create Trainer object
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer)
    logger.info("Trainer object created")

    # Save the model architecture
    if model.save_model_and_architecture():
        logger.info("Model architecture saved successfully")
    else:
        logger.error("Failed to save model architecture")
    
    # Train the model
    logger.info(f"Starting training for {epochs} epochs")
    trained_model = trainer.train(epochs)
    logger.info("Training completed")

    return trained_model


if __name__ == "__main__":
    import argparse
    DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "ratings.csv")
    parser = argparse.ArgumentParser(description="Train a matrix factorization model.")
    parser.add_argument("--data", type=str, help="Path to the dataset.", default=DEFAULT_DATASET)
    parser.add_argument("--embedding_size", type=int, default=50, help="Size of the embedding vectors.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    args = parser.parse_args()
    
    # Check if the dataset exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found at {args.data}")

    # Load the movie dataset
    movies = pd.read_csv(f"{os.path.dirname(DEFAULT_DATASET)}/movies.csv", usecols=['movieId', 'title', 'genres'])

    # Load the dataset
    ratings = pd.read_csv(args.data)

    # Create user and movie indices
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    # Create mappings for user and movie IDs
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_mapping = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
    movie_mapping = {int(movie_id): idx for idx, movie_id in enumerate(movie_ids)}

    # Save movie mappings as JSON
    PATH = os.path.join(Path(os.path.abspath(__file__)).parent.parent, "models", "mappings")
    os.makedirs(PATH, exist_ok=True)
    with open(f"{PATH}/movie_mapping.json", "w") as f:
        json.dump(movie_mapping, f)
    logger.info("Movie ID mapping saved successfully")
    
    # Get the number of unique users and movies
    num_users = len(user_ids)
    num_movies = len(movie_ids)

    # Movie ID, Title map
    movie_title_map = dict(zip(movies['movieId'], movies['title']))

    # Saving movie titles map for future reference in a json file
    with open(f"{PATH}/movie_title_map.json", "w") as f:
        json.dump(movie_title_map, f)
    logger.info("Movie title map saved successfully")

    # Replace user and movie IDs with indices
    ratings['userId'] = ratings['userId'].map(user_mapping)
    ratings['movieId'] = ratings['movieId'].map(movie_mapping)

        # Split the dataset into training and test sets
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Create DataLoader objects for training and test sets
    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train the model
    trained_model = main(
        train_loader=train_loader,
        test_loader=test_loader,
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=args.embedding_size,
        epochs=args.epochs
    )
