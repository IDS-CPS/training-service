from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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