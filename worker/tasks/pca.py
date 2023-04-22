import numpy as np

from worker.celery import app
from service.data_preprocessor import DataPreprocessor
from sklearn.decomposition import PCA

@app.task(name="train_pca", bind=True)
def train_pca(self, param):
    preprocessor = DataPreprocessor()

    df = preprocessor._read_df(param["df_name"])
    train_df, test_df = preprocessor._train_test_split(df, param["split_ratio"])
    train_df, test_df = preprocessor._feature_selection(train_df, test_df)
    train_data, test_data = preprocessor._scale_data(train_df, test_df)

    pca = PCA(n_components=13)
    pca.fit(train_data)
    test_pca = pca.transform(test_data)
    test_pred = pca.inverse_transform(test_pca)
    error = np.abs(test_pred - test_data)

    e_mean = np.mean(error, axis=0)
    e_std = np.std(error, axis=0)

    return "OK"