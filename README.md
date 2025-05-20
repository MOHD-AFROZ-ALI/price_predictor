 Price Predictor
A modular end-to-end housing price prediction system built with ZenML and MLflow

Table of Contents
1. Overview
2. Features
3. Architecture
3.1 Directory Structure
3.2 Design Patterns
4. Installation
5. Usage
5.1 Training Pipeline
5.2 Deployment Pipeline
5.3 Making Predictions
6. Technologies
7. Model Details
8. Exploratory Data Analysis
9. Contributing
10. License
1. Overview
The Price Predictor is a production-grade machine learning project designed to predict housing prices based on the Ames Housing dataset. This project showcases a complete end-to-end ML pipeline built with ZenML, from data ingestion through model deployment, with a focus on modularity, extensibility, and software engineering best practices.

The system is structured using advanced design patterns to ensure components are reusable, maintainable, and easily extensible. The entire machine learning workflow is automated, from data preparation to model serving, with comprehensive experiment tracking via MLflow.

2. Features
Modular, extensible architecture using design patterns
Automated data preprocessing pipeline
Comprehensive feature engineering
Model training with hyperparameter optimization
Experiment tracking with MLflow
Model deployment as a REST API service
Detailed exploratory data analysis
Continuous deployment pipeline
Clear separation of concerns between data, model, and deployment code
3. Architecture
The project follows a well-structured architecture focused on modularity, reusability, and extensibility. It implements several design patterns to ensure clean code organization and separation of concerns.

3.1 Directory Structure
└── mohd-afroz-ali-price_predictor/
    ├── README.md
    ├── config.yaml               # Main configuration file
    ├── requirements.txt          # Project dependencies
    ├── run_deployment.py         # Script to run deployment pipeline
    ├── run_pipeline.py           # Script to run training pipeline
    ├── sample_predict.py         # Example prediction script
    ├── analysis/                 # Exploratory data analysis
    │   ├── EDA.ipynb
    │   └── analyze_src/          # Analysis modules
    │       ├── basic_data_inspection.py
    │       ├── bivariate_analysis.py
    │       ├── missing_values_analysis.py
    │       ├── multivariate_analysis.py
    │       └── univariate_analysis.py
    ├── explanations/             # Design pattern explanations
    │   ├── factory_design_patter.py
    │   ├── strategy_design_pattern.py
    │   └── template_design_pattern.py
    ├── extracted_data/           # Directory for extracted data
    ├── pipelines/                # ZenML pipeline definitions
    │   ├── deployment_pipeline.py
    │   └── training_pipeline.py
    ├── src/                      # Core functionality modules
    │   ├── data_splitter.py
    │   ├── feature_engineering.py
    │   ├── handle_missing_values.py
    │   ├── ingest_data.py
    │   ├── model_building.py
    │   ├── model_evaluator.py
    │   └── outlier_detection.py
    └── steps/                    # Pipeline step definitions
        ├── data_ingestion_step.py
        ├── data_splitter_step.py
        ├── feature_engineering_step.py
        ├── handle_missing_values_step.py
        ├── model_building_step.py
        ├── model_evaluator_step.py
        ├── model_loader.py
        ├── outlier_detection_step.py
        ├── prediction_service_loader.py
        └── predictor.py
3.2 Design Patterns
The project implements the following design patterns to ensure clean, maintainable, and extensible code:

Strategy Pattern
Used extensively for various interchangeable algorithms throughout the codebase:

Feature Engineering: Various strategies for transforming data (LogTransformation, StandardScaling, etc.)
Model Building: Interchangeable modeling strategies
Data Splitting: Different strategies for splitting data
Missing Value Handling: Various strategies for handling missing values
Analysis Components: Different analysis strategies
Example implementation:

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: List[str]):
        self.features = features
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for feature in self.features:
            if feature in df_copy.columns:
                # Apply log transformation to positive values
                mask = df_copy[feature] > 0
                df_copy.loc[mask, feature] = np.log(df_copy.loc[mask, feature])
        return df_copy

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy
        
    def apply_feature_engineering(self, df: pd.DataFrame):
        return self._strategy.apply_transformation(df)
Factory Pattern
Used to abstract the creation of different components, particularly in data ingestion:

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")
Template Method Pattern
Used to define the skeleton of an algorithm while allowing subclasses to override specific steps:

class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass
        
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass
Pipeline Pattern
The workflow is organized using ZenML's pipeline framework, which encapsulates the entire ML workflow as a series of connected steps:

@pipeline(model=Model(name="prices_predictor"))
def ml_pipeline():
    raw_data = data_ingestion_step(...)
    filled_data = handle_missing_values_step(raw_data)
    engineered_data = feature_engineering_step(...)
    clean_data = outlier_detection_step(...)
    X_train, X_test, y_train, y_test = data_splitter_step(...)
    model = model_building_step(...)
    evaluation_metrics, mse = model_evaluator_step(...)
    return model
4. Installation
Follow these steps to set up the project locally:

Clone the repository
Create and activate a virtual environment (recommended)
Install the required dependencies
# Clone the repository
git clone https://github.com/username/price-predictor.git
cd price-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
5. Usage
5.1 Training Pipeline
To run the training pipeline, which will ingest data, preprocess it, train a model, and log metrics to MLflow:

python run_pipeline.py
After running the pipeline, you can view the experiment tracking results in the MLflow UI:

mlflow ui --backend-store-uri 'path/to/mlflow/tracking'
5.2 Deployment Pipeline
To train and deploy the model as a REST API service using MLflow:

python run_deployment.py
This will start the MLflow model server locally, making the model available for predictions.

To stop the service:

python run_deployment.py --stop-service
5.3 Making Predictions
Once the model is deployed, you can use the sample prediction script to make predictions:

python sample_predict.py
This sends a sample house data record to the prediction server and returns the predicted price.

Example prediction request:

import json
import requests

# URL of the MLflow prediction server
url = "http://127.0.0.1:8000/invocations"

# Sample input data for prediction
input_data = {
    "dataframe_records": [
        {
            "Order": 1,
            "PID": 5286,
            "MS SubClass": 20,
            "Lot Frontage": 80.0,
            "Lot Area": 9600,
            # ... other features
            "Mo Sold": 5,
            "Yr Sold": 2010,
        }
    ]
}

# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

# Send the POST request to the server
response = requests.post(url, headers=headers, data=json_data)

# Check the response status code
if response.status_code == 200:
    # If successful, print the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    # If there was an error, print the status code and the response
    print(f"Error: {response.status_code}")
    print(response.text)
6. Technologies
ZenML: ML pipeline orchestration framework
MLflow: For experiment tracking and model deployment
scikit-learn: For machine learning algorithms and preprocessing
pandas: For data manipulation
NumPy: For numerical operations
Matplotlib & Seaborn: For data visualization
7. Model Details
The current implementation uses a linear regression model for house price prediction. The model pipeline includes:

Data Preprocessing:
Imputation of missing values (numerical with mean, categorical with most frequent)
Outlier detection and removal
Feature Engineering:
Log transformation of skewed features
Standard scaling of numerical features
One-hot encoding of categorical features
Model: Linear regression implemented via scikit-learn
Evaluation Metrics: Mean Squared Error (MSE), R-squared (R²)
The model pipeline is fully automated and tracked using MLflow, allowing for easy comparison of different model versions and configurations.

8. Exploratory Data Analysis
The project includes comprehensive exploratory data analysis (EDA) in the analysis/ directory. The EDA covers:

Basic data inspection and summary statistics
Missing value analysis and visualization
Univariate analysis of numerical and categorical features
Bivariate analysis to explore relationships between features
Multivariate analysis to understand complex interactions
Key insights from the EDA:

The dataset contains 2930 entries and 82 columns
There are significant missing values in features like Alley, Pool QC, and Misc Feature
SalePrice is positively skewed, suggesting a log transformation
Features like Gr Liv Area and Overall Qual show strong correlations with SalePrice
9. Contributing
Contributions to the project are welcome. To contribute:

Fork the repository
Create a new branch (git checkout -b feature-branch)
Make your changes
Run tests to ensure functionality
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Create a new Pull Request
10. License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

Developed by Mohammad Afroz Ali

© 2023
