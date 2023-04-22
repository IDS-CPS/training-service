import numpy as np
import nanoid

def generate_id():
    return nanoid.generate(size=6)

def calculate_error(model, train_data, history_size, target_size):
    error_arr = []
    for i in range (len(train_data)//history_size-1):
        start_index = history_size * i
        end_index = start_index + history_size
        input_window = train_data[start_index:start_index+history_size]
        target_window = train_data[end_index:end_index+target_size]

        prediction = model.predict(input_window.reshape(1, history_size, -1), verbose=0).reshape((target_window.shape[0], target_window.shape[1]))
        error = np.abs(prediction - target_window)

        error_arr.append(error)

    error_arr = np.asarray(error_arr)
    error_arr = error_arr.reshape((-1, error_arr.shape[-1]))

    error_mean = np.mean(error_arr, axis=0)
    error_std = np.std(error_arr, axis=0)

    return error_mean, error_std