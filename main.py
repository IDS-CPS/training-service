from fastapi import FastAPI
from schemas.common import CommonResponse

app = FastAPI()

@app.get("/", response_model=CommonResponse)
def read_root():
    return {"message": "IDS-CPS Training Service"}