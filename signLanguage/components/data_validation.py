import os, sys
import shutil
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import DataValidationConfig
from signLanguage.entity.artifacts_entity import (DataIngestionArtifact,
                                                  DataValidationArtifact)



class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

        except Exception as e:
            raise SignException(e, sys) 
        

    
    def validate_all_files_exist(self) -> bool:
        try:
            all_files = os.listdir(self.data_ingestion_artifact.feature_store_path)
            missing_files = []

            for required_file in self.data_validation_config.required_file_list:
                if required_file not in all_files:
                    missing_files.append(required_file)

            validation_status = len(missing_files) == 0

            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                if validation_status:
                    f.write(f"Validation status: True\nAll required files found.")
                else:
                 f.write(f"Validation status: False\nMissing files: {missing_files}")

            return validation_status

        except Exception as e:
            raise SignException(e, sys)

    
    def initiate_data_validation(self) -> DataValidationArtifact: 
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            status = self.validate_all_files_exist()
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,data_zip_file_path=self.data_ingestion_artifact.data_zip_file_path)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data validation artifact: {data_validation_artifact}")



            return data_validation_artifact

        except Exception as e:
            raise SignException(e, sys)
