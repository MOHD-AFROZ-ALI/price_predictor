import os

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.integrations.bentoml.steps import bento_builder_step, bentoml_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")



@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()  # No need for is_promoted return value anymore

    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)


# def continuous_deployment_pipeline():
#     """Run a training job and package + deploy with BentoML."""
#     # 1. Train the model
#     trained_model = ml_pipeline()

#     # 2. Package the model into a Bento
#     bento = bento_builder_step(
#         model=trained_model,
#         model_name="prices_predictor",
#         model_type="sklearn",            # or the appropriate flavor
#         service="service.py:PredictionService",  # path to your Bento service
#     )

#     # 3. Deploy the Bento as a local HTTP service
#     deployed_model = bentoml_model_deployer_step(
#         bento=bento,
#         model_name="prices_predictor",
#         port=8000,
#         deployment_type="local",         # use "container" if you prefer Docker
#     )






@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)
