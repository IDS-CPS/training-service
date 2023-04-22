import tensorflow as tf

from worker.celery import app
from service.data_preprocessor import DataPreprocessor
from service.training.autoencoder import Autoencoder
from service.training.callback import UpdateTaskState

@app.task(name="train_autoencoder", bind=True)
def train_ae(self, param):
    preprocessor = DataPreprocessor(10, 10)
    x_train, y_train, x_test, y_test = preprocessor.preprocess(param["df_name"], param["split_ratio"])

    train_tensor = preprocessor.process_tensor(x_train, y_train)
    test_tensor = preprocessor.process_tensor(x_test, y_test)

    model = Autoencoder(x_train.shape)

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

    return {'curent_epoch': param["epochs"], 'total_epoch': param["epochs"]}