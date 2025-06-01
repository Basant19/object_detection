import os
import sys
from signLanguage.configuration.s3_operations import S3Operation
from signLanguage.entity.config_entity import ModelDownloaderConfig
from signLanguage.entity.artifacts_entity import ModelDownloaderArtifact
from signLanguage.exception import SignException
from signLanguage.logger import logging


class ModelDownloader:
    def __init__(self, config: ModelDownloaderConfig):
        self.config = config
        self.s3 = S3Operation()

    def initiate_model_download(self) -> ModelDownloaderArtifact:
        try:
            os.makedirs(self.config.model_downloader_dir, exist_ok=True)

            # Check if file already exists locally
            if os.path.exists(self.config.local_model_path):
                logging.info(f"Model already downloaded at: {self.config.local_model_path}")
                return ModelDownloaderArtifact(
                    model_path=self.config.local_model_path, downloaded=False
                )

            # Check if model exists in S3
            if not self.s3.is_model_present(
                self.config.bucket_name, self.config.s3_model_key
            ):
                raise SignException(f"Model not found in S3: {self.config.s3_model_key}")

            # Download the model
            self.s3.s3_resource.meta.client.download_file(
                Bucket=self.config.bucket_name,
                Key=self.config.s3_model_key,
                Filename=self.config.local_model_path
            )

            logging.info(f"Model downloaded from S3 to: {self.config.local_model_path}")

            return ModelDownloaderArtifact(
                model_path=self.config.local_model_path, downloaded=True
            )

        except Exception as e:
            raise SignException(e, sys) from e
