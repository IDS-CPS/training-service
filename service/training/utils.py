import numpy as np
import tensorflow as tf
import nanoid

def generate_id():
    return nanoid.generate(size=6)

def calculate_error(model, data, history_size):
    error_arr = []
    for i in range (len(data)-history_size):
        end_index = i + history_size
        input_window = data[i:end_index]
        target_window = data[end_index]

        prediction = model.predict(np.expand_dims(input_window, axis=0), verbose=0).squeeze()
        error = np.abs(prediction - target_window)

        error_arr.append(error)

    error_arr = np.asarray(error_arr)

    error_mean = np.mean(error_arr, axis=0)
    error_std = np.std(error_arr, axis=0)

    return error_mean, error_std