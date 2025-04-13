from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    user_id: int = Field(
        description="The user id for the user to get recommendations for...",
        examples=[1]
    )
    num_recs: int = Field(
        description="The number of recommendations to be made for the user.",
        examples=[5]
    )