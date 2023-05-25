import requests
import json

from typing import List
from config.settings import Settings, settings

class ManagementService():

    def __init__(self, settings: Settings):
        self.base_url = settings.MANAGEMENT_SERVICE_URL

    def notify_train_finished(
        self, 
        task_id: str, 
        model_name: str, 
        scaler_name: str, 
        mean_name: str, 
        std_name: str,
        feature_used: List[str],
        history_size: int
    ):
        payload = {
            'model_name': model_name,
            'scaler_name': scaler_name,
            'mean_name': mean_name,
            'std_name': std_name,
            'feature_used': feature_used,
            'history_size': history_size
        }
        r = requests.post(
            f'{self.base_url}/api/v1/learning-model/{task_id}/confirmation',
            data=json.dumps(payload)
        )

        if r.status_code != requests.codes.ok:
            print("failed notifying")

management_service = ManagementService(settings)