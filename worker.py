from celery import Celery

app = Celery(
    'celery_web',
    broker='redis://localhost:6379/0',
    backend='rpc://',
    include=['tasks']
)
