import tensorflow as tf

class UpdateTaskState(tf.keras.callbacks.Callback):
    def __init__(self, task, total_epoch):
        self.task = task
        self.total_epoch = total_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.task.update_state(state='PROGRESS', meta={'curent_epoch': epoch, 'total_epoch': self.total_epoch})