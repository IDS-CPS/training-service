from pydantic import BaseModel

class BaseTrainingParam(BaseModel):
    df_name: str
    split_ratio: float

class AutoencoderParam(BaseTrainingParam):
    history_size: int
    epochs: int

class PCAParam(BaseTrainingParam):
    n_components: int

class OneDCNNParam(BaseTrainingParam):
    history_size: int
    epochs: int
    n_filter: int
    pool_size: int
    kernel_size: int
    dropout_rate: float

class LSTMParam(BaseTrainingParam):
    history_size: int
    epochs: int
    n_units: int

class TaskAck(BaseModel):
    task_id: str

class TaskStatus(BaseModel):
    state: str
    meta: dict