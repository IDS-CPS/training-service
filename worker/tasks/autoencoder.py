import tensorflow as tf

from worker.celery import app
from service.data_preprocessor import DataPreprocessor
from service.training.autoencoder import Autoencoder
from service.training.callback import UpdateTaskState
from service.training.utils import generate_id, calculate_error
from service.minio import minio_client

@app.task(name="train_autoencoder", bind=True)
def train_ae(self, param):
    preprocessor = DataPreprocessor(param["history_size"])
    x_train, y_train, x_test, y_test = preprocessor.preprocess(param["df_name"], param["split_ratio"])

    train_tensor = preprocessor.process_tensor(x_train, y_train)
    test_tensor = preprocessor.process_tensor(x_test, y_test)

    model = Autoencoder(x_train.shape)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.experimental.AdamW(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        train_tensor, 
        epochs=param["epochs"],
        steps_per_epoch=100,
        validation_data=test_tensor,
        validation_steps=50,
        callbacks=[early_stopping, UpdateTaskState(task=self, total_epoch=param["epochs"])]
    )

    loss, mae = model.evaluate(x_test, y_test)

    scaler = preprocessor.get_scaler()
    e_mean, e_std = calculate_error(model, preprocessor.get_test_data(), param["history_size"])

    model_id = generate_id()
    minio_client.save_file(e_mean, "npy/ae", f"mean-{model_id}")
    minio_client.save_file(e_std, "npy/ae", f"std-{model_id}")
    minio_client.save_file(scaler, "scaler/ae", f"scaler-{model_id}")
    minio_client.save_keras_model(model, f"ae-{model_id}")

    return {'loss': loss, 'mae': mae}