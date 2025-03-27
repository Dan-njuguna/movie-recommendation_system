import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os

# Load the MovieLens dataset
DATA_PATH = "data"
movies = pd.read_csv(f"{DATA_PATH}/movies.csv", usecols=['movieId', 'title', 'genres'])
ratings = pd.read_csv(f"{DATA_PATH}/ratings.csv", usecols=['userId', 'movieId', 'rating'])

# Create user and movie indices
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

# Map users and movies to indices
user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Movie ID, Title map
movie_title_map = dict(zip(movies['movieId'], movies['title']))

# Replace user and movie IDs with indices
ratings['userId'] = ratings['userId'].map(user_id_map)
ratings['movieId'] = ratings['movieId'].map(movie_id_map)

# Split the data into training and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

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

train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user, movie):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        x = torch.cat([user_embed, movie_embed], dim=1)
        x = self.fc(x)
        return x

# Define model parameters
num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_size = 50

# Initialize the model
model = MatrixFactorization(num_users, num_movies, embedding_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Set number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Iterate through batches of training data
    for user, movie, rating in train_loader:
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(user, movie)
        # Calculate loss
        loss = criterion(outputs.squeeze(), rating)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        running_loss += loss.item()

    # Print epoch statistics
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Save the trained model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/matrix_factorization.pth')

# Evaluation phase
model.eval()
test_loss = 0.0
with torch.no_grad():
    # Calculate loss on test set
    for user, movie, rating in test_loader:
        outputs = model(user, movie)
        loss = criterion(outputs.squeeze(), rating)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_loader)}')

def recommend_movies(user_id, num_recommendations=5):
    """
    Generate movie recommendations for a specific user
    """
    model.eval()
    # Convert user ID to tensor
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    # Get all movie IDs
    movie_tensor = torch.tensor(list(movie_id_map.values()), dtype=torch.long)
    with torch.no_grad():
        # Get embeddings for user and all movies
        user_embed = model.user_embedding(user_tensor)
        movie_embed = model.movie_embedding(movie_tensor)
        # Calculate predicted ratings
        ratings = torch.matmul(user_embed, movie_embed.t())
    # Get top-rated movies
    top_movies = torch.topk(ratings, num_recommendations).indices.squeeze().tolist()
    # Convert movie indices back to movie IDs
    return [list(movie_id_map.keys())[idx] for idx in top_movies]

# Example usage: Get recommendations for user 1
recommendations = recommend_movies(10)
for movie_id in recommendations:
    print(f'{movie_title_map[movie_id]}', end="\n")
print(f'Recommended Movies: {recommendations}')