import os, sys
import yaml
import zipfile
import shutil
from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifacts_entity import ModelTrainerArtifact
from signLanguage.entity.artifacts_entity import DataValidationArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        self.model_trainer_config = model_trainer_config
        self.data_validation_artifact = data_validation_artifact

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            with zipfile.ZipFile(self.data_validation_artifact.data_zip_file_path, 'r') as zip_ref:
                zip_ref.extractall()

            logging.info("Unzipped data successfully")
            logging.info("Reading data.yaml file")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)

            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            # Run training command
            os.system(
                f"cd yolov5 && python train.py --img 416 --batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} --data ../data.yaml "
                f"--cfg ./models/custom_{model_config_file_name}.yaml --weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results --cache"
            )

            # Get the latest training run directory
            runs_train_path = os.path.join("yolov5", "runs", "train")
            latest_run_dir = sorted(
                [d for d in os.listdir(runs_train_path) if os.path.isdir(os.path.join(runs_train_path, d))],
                key=lambda x: os.path.getmtime(os.path.join(runs_train_path, x))
            )[-1]
            logging.info(f"Latest training run directory: {latest_run_dir}")

            trained_model_source = os.path.join(runs_train_path, latest_run_dir, "weights", "best.pt")
            trained_model_dest = os.path.join("yolov5", "best.pt")
            trainer_model_output = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")

            # Copy best.pt to yolov5/
            shutil.copy(trained_model_source, trained_model_dest)

            # Ensure model output directory exists
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            # Copy best.pt to output dir
            shutil.copy(trained_model_source, trainer_model_output)

            # Cleanup
            shutil.rmtree(os.path.join("yolov5", "runs"), ignore_errors=True)
            shutil.rmtree("train", ignore_errors=True)
            shutil.rmtree("test", ignore_errors=True)

            if os.path.exists("data.yaml"):
                os.remove("data.yaml")
            
            for readme_file in ["README.dataset.txt", "README.roboflow.txt"]:
                if os.path.exists(readme_file):
                    os.remove(readme_file)


            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_dest,
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
