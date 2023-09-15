import os.path
import sys

from pythae.pipelines import TrainingPipeline
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainerConfig
import torchvision

sys.path.append(os.path.dirname("pipelines"))

if __name__ == '__main__':
    # Set up the training configuration
    aeTrainingConfig = BaseTrainerConfig(
        output_dir="ae",
        num_epoches=100,
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_test_batch_size=64,
        steps_saving=20,
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5}
    )

    # Set up the model configuration
    aeConfig = AEConfig(
        input_dim=(1, 218, 178),
        latent_dim=10
    )

    # Build the model
    aeModel = AE(model_config=aeConfig)
    print("Model Architecture:", aeModel)

    # Build the pipeline
    pipeline = TrainingPipeline(
        training_config=aeTrainingConfig,
        model=aeModel
    )

    datasetPath = "E:\TorchVision_Datasets\CelebA"
    celebaTrain = torchvision.datasets.CelebA(root=datasetPath, split="train", download=True)
    celebaEval = torchvision.datasets.CelebA(root=datasetPath, split="valid", download=True)
    celebaTest = torchvision.datasets.CelebA(root=datasetPath, split="test", download=True)

    # Launch the Pipeline
    pipeline(
        train_data=celebaTrain,
        eval_data=celebaTest
    )
