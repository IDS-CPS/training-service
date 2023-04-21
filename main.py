from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from worker.tasks import train_ae, train_cnn, train_lstm, train_pca
from schemas.common import CommonResponse

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
async def train_ae_model():
    task = train_ae.delay("dataset/swat-minimized.csv", 0.6)

    return {"task_id": str(task)}

@app.get('/result/{task_id}')
async def fetch_result(task_id):
    # Fetch result for task_id
    task = AsyncResult(task_id)
    
    return {'status': task.status}
