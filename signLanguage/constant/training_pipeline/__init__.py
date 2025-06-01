import os

ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://github.com/Basant19/sign_language_image/raw/main/sign%20language%20detection.v1i.yolov5pytorch.zip"




"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "test", "data.yaml"]



"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov5s.pt"

MODEL_TRAINER_NO_EPOCHS: int = 50

MODEL_TRAINER_BATCH_SIZE: int = 4




"""
MODEL PUSHER related constant start with MODEL_PUSHER var name
"""
BUCKET_NAME = "sign-lang-detection-basant"
S3_MODEL_NAME = "best.pt"


"""
MODEL DOWNLOADER related constant start with MODEL_DOWNLOADER var name
"""
MODEL_DOWNLOADER_DIR_NAME: str = "model_downloader"
MODEL_DOWNLOADER_LOCAL_MODEL_PATH: str = "best.pt"
