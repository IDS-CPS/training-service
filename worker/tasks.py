import tensorflow as tf

from worker.celery import app
from celery.utils.log import get_task_logger
from service.data_preprocessor import DataPreprocessor
from service.training.autoencoder import Autoencoder

@app.task(name="train_autoencoder")
def train_ae(df_name, split_ratio):
    preprocessor = DataPreprocessor(10, 10)
    scaler, x_train, y_train, x_test, y_test = preprocessor.preprocess(df_name, split_ratio)

    train_tensor = preprocessor.process_tensor(x_train, y_train)
    test_tensor = preprocessor.process_tensor(x_test, y_test)

    model = Autoencoder(x_train.shape)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.experimental.AdamW(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(
        train_tensor, 
        epochs=1,
        steps_per_epoch=100,
        validation_data=test_tensor,
        validation_steps=50,
        callbacks=[early_stopping]
    )

    return 'OK'

@app.task
def train_cnn():
    pass

@app.task
def train_lstm():
    pass

@app.task
def train_pca():
    pass