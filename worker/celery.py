from celery import Celery

app = Celery(
    'ircm_training',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['worker.tasks.autoencoder', 'worker.tasks.pca']
)

app.conf.update(
    result_expires=60
)

if __name__ == '__main__':
    app.start()