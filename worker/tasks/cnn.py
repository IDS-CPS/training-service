import tensorflow as tf

from worker.celery import app
from service.data_preprocessor import DataPreprocessor
from service.training.callback import UpdateTaskState
from service.training.utils import generate_id, calculate_error
from service.minio import minio_client
from service.management import management_service
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Reshape

@app.task(name="train_cnn", bind=True)
def train_cnn(self, param):
    preprocessor = DataPreprocessor(param["history_size"])
    x_train, y_train, x_test, y_test = preprocessor.preprocess(param["df_name"], param["split_ratio"])

    train_tensor = preprocessor.process_tensor(x_train, y_train)
    test_tensor = preprocessor.process_tensor(x_test, y_test)

    n_filter = param["n_filter"]
    dropout_rate = param["dropout_rate"]
    kernel_size = param["kernel_size"]
    pool_size = param["pool_size"]

    model = tf.keras.models.Sequential()
    model.add(Conv1D(filters=n_filter, kernel_size=kernel_size, activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=pool_size, strides=1))
    model.add(Conv1D(filters=n_filter*2, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, strides=1))
    model.add(Conv1D(filters=n_filter*4, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, strides=1))
    model.add(Conv1D(filters=n_filter*8, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, strides=1))
    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=x_train.shape[2]))

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
    mean_name = minio_client.save_file(e_mean, "npy/cnn", f"mean-{model_id}")
    std_name = minio_client.save_file(e_std, "npy/cnn", f"std-{model_id}")
    scaler_name = minio_client.save_file(scaler, "scaler/cnn", f"scaler-{model_id}")
    model_name = minio_client.save_keras_model(model, f"cnn-{model_id}")

    management_service.notify_train_finished(str(self.request.id), model_name, scaler_name, mean_name, std_name)

    return {'loss': loss, 'mae': mae}