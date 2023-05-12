import tensorflow as tf

from worker.celery import app
from service.data_preprocessor import DataPreprocessor
from service.training.callback import UpdateTaskState
from service.training.utils import generate_id, calculate_error
from service.minio import minio_client
from service.management import management_service
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

@app.task(name="train_lstm", bind=True)
def train_lstm(self, param):
    preprocessor = DataPreprocessor(param["history_size"])
    x_train, y_train, x_test, y_test = preprocessor.preprocess(param["df_name"], param["split_ratio"])

    train_tensor = preprocessor.process_tensor(x_train, y_train)
    test_tensor = preprocessor.process_tensor(x_test, y_test)

    n_units = param["n_units"]

    model = tf.keras.models.Sequential([ 
        Bidirectional(LSTM(n_units, return_sequences=True, input_shape=x_train.shape[1:])),
        Bidirectional(LSTM(n_units//2, return_sequences=True)),
        Bidirectional(LSTM(n_units//4, return_sequences=True)),
        Bidirectional(LSTM(n_units//8, return_sequences=True)),
        Bidirectional(LSTM(n_units//16)),
        Dense(x_train.shape[2]),
    ]) 

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.experimental.AdamW(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

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
    mean_name = minio_client.save_file(e_mean, "npy/lstm", f"mean-{model_id}")
    std_name = minio_client.save_file(e_std, "npy/lstm", f"std-{model_id}")
    scaler_name = minio_client.save_file(scaler, "scaler/lstm", f"scaler-{model_id}")
    model_name = minio_client.save_keras_model(model, f"lstm-{model_id}")

    management_service.notify_train_finished(str(self.request.id), model_name, scaler_name, mean_name, std_name)

    return {'loss': loss, 'mae': mae}