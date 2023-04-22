from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from celery.result import AsyncResult

from worker.tasks.autoencoder import train_ae
from worker.tasks.pca import train_pca
from schemas.common import CommonResponse
from schemas.training import AutoencoderParam, PCAParam

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

@app.post("/train-ae")
async def train_ae_model(param: AutoencoderParam):
    task = train_ae.delay(jsonable_encoder(param))

    return {"task_id": str(task)}

@app.post("/train-pca")
async def train_pca_model(param: PCAParam):
    task = train_pca.delay(jsonable_encoder(param))

    return {"task_id": str(task)}

@app.get('/result/{task_id}')
async def fetch_result(task_id):
    task = AsyncResult(task_id)

    return {"state": task.state, "meta": task.info}
