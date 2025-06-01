import sys, os
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.components.data_ingestion import DataIngestion
from signLanguage.components.data_validation import DataValidation
from signLanguage.components.model_trainer import ModelTrainer
from signLanguage.components.model_pusher import ModelPusher
from signLanguage.configuration.s3_operations import S3Operation
from signLanguage.entity.config_entity import (DataIngestionConfig,DataValidationConfig,ModelTrainerConfig,ModelPusherConfig,ModelDownloaderConfig)
from signLanguage.entity.artifacts_entity import (DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelPusherArtifacts,ModelDownloaderArtifact)
from signLanguage.components.model_downloader import ModelDownloader

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.model_downloader_config = ModelDownloaderConfig()
        self.s3_operations = S3Operation()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact
        except Exception as e:
            raise SignException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:

        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact
        

        except Exception as e:
            raise SignException(e, sys) from e
        
    def start_model_trainer(self,data_validation_artifact: DataValidationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
    

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact, s3: S3Operation):

        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact= model_trainer_artifact,
                s3=s3
                
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise SignException(e, sys)

    
    def start_model_download(self) -> ModelDownloaderArtifact:
        """Downloads the trained model from S3 if not present locally.
        """
        try:
            logging.info("Entered the start_model_download method of TrainPipeline class")

            model_downloader = ModelDownloader(config=self.model_downloader_config)
            model_downloader_artifact = model_downloader.initiate_model_download()

            logging.info("Exited the start_model_download method of TrainPipeline class")

            return model_downloader_artifact

        except Exception as e:
            raise SignException(e, sys)

    def run_pipeline (self) ->None:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            try:
                data_ingestion_artifact = self.start_data_ingestion() 
                data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            
                if data_validation_artifact.validation_status == True:
                    model_trainer_artifact = self.start_model_trainer(data_validation_artifact=data_validation_artifact)
                    model_pusher_artifact = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact,s3=self.s3_operations)
                    model_downloader_artifact = self.start_model_download()
                    logging.info(f"Model downloaded: {model_downloader_artifact.downloaded}, Path: {model_downloader_artifact.model_path}")
                else:
                    raise Exception("Your data is not in correct format")
                
            except Exception as e:
                raise SignException(e, sys)    