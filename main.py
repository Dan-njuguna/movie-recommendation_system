"""
This module is the API for the application.
It gets the user Id and makes the movie recommendation based on user behaviour learnt by the model.

AUTHOR: Dan Njuguna
DATE: 13-04-2025
"""

import sys
from fastapi import FastAPI
from typing import List
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from steps.predict import PredictorPipeline
from db.models import UserRequest

app = FastAPI(
    title="Movie Recommendation API",
    description="This is a movie recommendation API",
    version="0.1.0"
)


@app.post("/recommendations")
async def recommendations(request: UserRequest):
    """This function receives the user id and the number of recommendations required
    """
    user_id: int = int(request.user_id)
    n_recs: int = int(request.num_recs)

    recommendations: List[str] = PredictorPipeline(user_id=user_id, top_k=n_recs).movie_titles()
    return {
        "status": "success",
        "message": f"Movie recommendations made successfully for user {user_id}",
        "recommendations": recommendations
    }