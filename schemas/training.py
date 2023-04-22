from pydantic import BaseModel

class BaseTrainingParam(BaseModel):
    df_name: str
    split_ratio: float

class AutoencoderParam(BaseTrainingParam):
    history_size: int
    target_size: int
    epochs: int

class PCAParam(BaseTrainingParam):
    n_components: int