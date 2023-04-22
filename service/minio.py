import tempfile
import zipfile
import os
import joblib

from minio import Minio
from config.settings import Settings, settings

class MinioClient():
    client: Minio

    def __init__(self, settings: Settings):
        try:
            self.bucket_name = settings.S3_BUCKET
            self.client = Minio(
                endpoint=settings.S3_ADDRESS, 
                access_key=settings.S3_USER, 
                secret_key=settings.S3_PASSWORD,
                secure=False
            )

        except Exception as e:
            raise e

    def zipdir(self, path, ziph):
        # Zipfile hook to zip up model folders
        length = len(path) # Doing this to get rid of parent folders
        for root, dirs, files in os.walk(path):
            folder = root[length:] # We don't need parent folders! Why in the world does zipfile zip the whole tree??
            for file in files:
                ziph.write(os.path.join(root, file), os.path.join(folder, file))

    def save_file(self, object, directory, filename):
        with tempfile.TemporaryDirectory() as tempdir:
            joblib.dump(object, f"{tempdir}/{filename}")
            result = self.client.fput_object(self.bucket_name, f"{directory}/{filename}.gz", f"{tempdir}/{filename}")

            return result.object_name

    def save_keras_model(self, model, filename):
        with tempfile.TemporaryDirectory() as tempdir:
            model.save(f"{tempdir}/{filename}")
            # Zip it up first
            zipf = zipfile.ZipFile(f"{tempdir}/{filename}.zip", "w", zipfile.ZIP_STORED)
            self.zipdir(f"{tempdir}/{filename}", zipf)
            zipf.close()
            result = self.client.fput_object(self.bucket_name, f"model/{filename}.zip", f"{tempdir}/{filename}.zip")

            return result.object_name

minio_client = MinioClient(settings)