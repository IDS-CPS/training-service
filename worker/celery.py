from celery import Celery
from config.settings import settings

app = Celery(
    'ircm_training',
    broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0',
    backend=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0',
    include=['worker.tasks.autoencoder', 'worker.tasks.pca', 'worker.tasks.cnn', 'worker.tasks.lstm']
)

app.conf.update(
    result_expires=settings.REDIS_TTL
)

if __name__ == '__main__':
    app.start()