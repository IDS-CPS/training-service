from pydantic import BaseModel

class BaseTrainingParam(BaseModel):
    df_name: str
    split_ratio: float
    epochs: int