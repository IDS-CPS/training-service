from worker.celery import app

@app.task(name="train_pca")
def train_pca():
    pass