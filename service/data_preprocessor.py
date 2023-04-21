import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from tensorflow.data import Dataset

class DataPreprocessor():
    def __init__(self, history_size, target_size):
        self.history_size = history_size
        self.target_size = target_size
        self.scaler = MinMaxScaler()

    def _read_df(self, df_name: str):
        df = pd.read_csv(df_name, delimiter=";", decimal=",")
        df = df.drop("Normal/Attack", axis=1)
        df = df.drop("Timestamp", axis=1)

        return df

    def _train_test_split(self, df, split_ratio):
        n = len(df)
        train_df = df[:int(n*split_ratio)]
        test_df = df[int(n*split_ratio):]

        return train_df, test_df

    def _feature_selection(self, train_df, test_df):
        features_considered = []
        for column in train_df.columns:
          ks_result = ks_2samp(train_df[column],test_df[column])
          if (ks_result.statistic < 0.02):
            features_considered.append(column)

        return train_df[features_considered], test_df[features_considered]

    def _scale_data(self, train_df, test_df):
        self.scaler.fit(train_df)
        train_data = self.scaler.transform(train_df)
        test_data = self.scaler.transform(test_df)

        return train_data, test_data

    def _create_sequences(self, values):
        data = []
        target = []

        for i in range(len(values)//self.history_size-1):
            start_index = self.history_size * i
            end_index = start_index + self.history_size
            data.append(values[start_index:end_index])
            target.append(values[end_index:end_index+self.target_size])
        
        return np.array(data), np.array(target)

    def preprocess(self, df_name, split_ratio):
        df = self._read_df(df_name)
        train_df, test_df = self._train_test_split(df, split_ratio)
        train_df, test_df = self._feature_selection(train_df, test_df)
        train_data, test_data = self._scale_data(train_df, test_df)
        x_train, y_train = self._create_sequences(train_data)
        x_test, y_test = self._create_sequences(test_data)

        print("Training input shape: ", x_train.shape, y_train.shape)

        return self.scaler, x_train, y_train, x_test, y_test

    def process_tensor(self, x_data, y_data):
        tensor = Dataset.from_tensor_slices((x_data, y_data))
        tensor = tensor.cache().shuffle(50000).batch(256).repeat()

        return tensor
