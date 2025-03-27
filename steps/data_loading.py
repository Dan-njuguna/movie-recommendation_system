#!/usr/bin/env python3

"""
This module is responsible for loading the data from various sources.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Type


class DataLoader(ABC):
    """Abstract base class for data loading operations"""
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load data from a source"""
        pass


class CSVLoader(DataLoader):
    """Concrete implementation for loading CSV files"""
    def load(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV file
        Args:
            path (str): Path to the CSV file
        Returns:
            pd.DataFrame: Loaded data
        """
        return pd.read_csv(path)


class DataLoaderFactory:
    """Factory class for creating data loaders"""
    _loaders: Dict[str, Type[DataLoader]] = {
        'csv': CSVLoader
    }

    @classmethod
    def register_loader(cls, file_type: str, loader: Type[DataLoader]) -> None:
        """Register a new loader type"""
        cls._loaders[file_type.lower()] = loader

    @classmethod
    def get_loader(cls, file_type: str) -> DataLoader:
        """
        Create appropriate loader instance
        Args:
            file_type (str): Type of file to load ('csv', etc)
        Returns:
            DataLoader: Appropriate loader instance
        """
        loader_class = cls._loaders.get(file_type.lower())
        if not loader_class:
            raise ValueError(f"No loader found for file type: {file_type}")
        return loader_class()
    

def main():
    # Create a CSV loader through the factory
    loader = DataLoaderFactory.get_loader('csv')
    
    # Load a sample CSV file
    try:
        df = loader.load('path/to/your/data.csv')
        print("Data loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == '__main__':
    main()