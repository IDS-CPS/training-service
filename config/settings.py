from fastapi import FastAPI
from pydantic import BaseSettings

class Settings(BaseSettings):
    S3_ADDRESS: str
    S3_USER: str
    S3_PASSWORD: str
    S3_BUCKET: str

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_TTL: int

    MANAGEMENT_SERVICE_URL: str
    
    class Config:
        env_file = ".env"

settings = Settings()
