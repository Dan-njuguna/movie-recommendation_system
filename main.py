"""
This module is the API for the application.
It gets the user Id and makes the movie recommendation based on user behaviour learnt by the model.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Movie Recommendation API",
    description="This is a movie recommendation API",
    version="0.1.0"
)