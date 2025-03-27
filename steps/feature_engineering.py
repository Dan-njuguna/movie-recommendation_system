#!/usr/bin/env python3

"""
This module is responsible for feature engineering.
"""

from typing import Optional
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class DataCleaner:
    """Handles data cleaning operations."""
    
    def clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove missing values from the data.
        Args:
            data (pd.DataFrame): The data to clean
        Returns:
            pd.DataFrame: Cleaned data
        Raises:
            ValueError: If data is empty or None
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        return data.dropna()

    def clean_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates from the data.
        Args:
            data (pd.DataFrame): The data to deduplicate
        Returns:
            pd.DataFrame: Deduplicated data
        Raises:
            ValueError: If data is empty or None
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        return data.drop_duplicates()

class IdGenerator:
    """Handles creation of unique IDs for features."""
    
    def create_feature_ids(self, data: pd.DataFrame, feature: str) -> pd.Series:
        """Create unique ids for the features.
        Args:
            data (pd.DataFrame): The data containing the feature
            feature (str): The feature column name
        Returns:
            pd.Series: The unique ids for the feature
        Raises:
            ValueError: If feature doesn't exist in data
        """
        if feature not in data.columns:
            raise ValueError(f"Feature {feature} not found in data")
        
        feature_ids = data[feature].unique()
        feature_ids_map = {feature_id: idx for idx, feature_id in enumerate(feature_ids)}
        return data[feature].map(feature_ids_map)

class FeatureEngineer:
    """Orchestrates the feature engineering process."""
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.id_generator = IdGenerator()
        
    def process_data(self, data: pd.DataFrame, id_features: Optional[list[str]] = None) -> pd.DataFrame:
        """Process the data through the feature engineering pipeline.
        Args:
            data (pd.DataFrame): Input data
            id_features (list[str], optional): Features to generate IDs for
        Returns:
            pd.DataFrame: Processed data
        """
        processed_data = self.cleaner.clean_missing_values(data)
        processed_data = self.cleaner.clean_duplicates(processed_data)
        
        if id_features:
            for feature in id_features:
                processed_data[f"{feature}id"] = self.id_generator.create_feature_ids(
                    processed_data, feature
                )
                
        return processed_data

class TorchDataset(Dataset):
    """PyTorch dataset for tabular data."""
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): The input data
        """
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


if __name__ == "__main__":
    # Sample usage
    # Create sample data
    data = pd.DataFrame({
        'userId': [1, 2, 3, 4, 5],
        'movieId': [101, 102, 103, 104, 105],
        'rating': [4.5, 3.0, 5.0, 2.5, 4.0],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
    })

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Process the data
    processed_data = fe.process_data(data, id_features=['title'])
    print("Processed data:")
    print(processed_data)

    # Create PyTorch dataset
    dataset = TorchDataset(processed_data)
    print(f"\nDataset size: {len(dataset)}")
    print(f"First item: {dataset[0]}")
