from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from celery.result import AsyncResult

from worker.tasks.autoencoder import train_ae
from worker.tasks.pca import train_pca
from worker.tasks.cnn import train_cnn
from worker.tasks.lstm import train_lstm
from schemas.common import CommonResponse
from schemas.training import AutoencoderParam, PCAParam, OneDCNNParam, LSTMParam, TaskStatus, TaskAck

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=CommonResponse)
def read_root():
    return {"message": "IDS-CPS Training Service"}

@app.post("/train-ae", response_model=TaskAck)
async def train_ae_model(param: AutoencoderParam):
    task = train_ae.delay(jsonable_encoder(param))

    return {"task_id": str(task)}

@app.post("/train-pca", response_model=TaskAck)
async def train_pca_model(param: PCAParam):
    task = train_pca.delay(jsonable_encoder(param))

    return {"task_id": str(task)}

@app.post("/train-cnn", response_model=TaskAck)
async def train_cnn_model(param: OneDCNNParam):
    task = train_cnn.delay(jsonable_encoder(param))

    return {"task_id": str(task)}

@app.post("/train-lstm", response_model=TaskAck)
async def train_lstm_model(param: LSTMParam):
    task = train_lstm.delay(jsonable_encoder(param))
    
    return {"task_id": str(task)}

@app.get('/result/{task_id}', response_model=TaskStatus)
async def fetch_result(task_id):
    task = AsyncResult(task_id)

    return {"state": task.state, "meta": task.info}
