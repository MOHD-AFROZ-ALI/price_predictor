


# 🏠 Price Predictor

A modular end-to-end housing price prediction system built with **ZenML** and **MLflow**.

---

## 📑 Table of Contents

1. [Overview](#1-overview)  
2. [Features](#2-features)  
3. [Architecture](#3-architecture)  
   - [3.1 Directory Structure](#31-directory-structure)  
   - [3.2 Design Patterns](#32-design-patterns)  
4. [Installation](#4-installation)  
5. [Usage](#5-usage)  
   - [5.1 Training Pipeline](#51-training-pipeline)  
   - [5.2 Deployment Pipeline](#52-deployment-pipeline)  
   - [5.3 Making Predictions](#53-making-predictions)  
6. [Technologies](#6-technologies)  
7. [Model Details](#7-model-details)  
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)  
9. [Contributing](#9-contributing)  
10. [License](#10-license)  

---

## 1. Overview

The **Price Predictor** is a production-grade machine learning project designed to predict housing prices based on the **Ames Housing dataset**. This project showcases a complete end-to-end ML pipeline using **ZenML**, from data ingestion to model deployment. It focuses on **modularity**, **extensibility**, and software engineering **best practices**.

With a structured architecture and automated ML workflow, this project enables efficient model training, evaluation, and deployment with experiment tracking via **MLflow**.

---

## 2. Features

- Modular, extensible architecture using design patterns  
- Automated data preprocessing pipeline  
- Comprehensive feature engineering  
- Model training with hyperparameter tuning  
- Experiment tracking with MLflow  
- Model deployment as a REST API  
- Detailed EDA and insights  
- Continuous deployment pipeline  
- Clear separation of concerns between data, model, and deployment  

---

## 3. Architecture

A modular system designed with clean separation of components and robust design patterns.

### 3.1 Directory Structure

```

mohd-afroz-ali-price\_predictor/
├── README.md
├── config.yaml
├── requirements.txt
├── run\_deployment.py
├── run\_pipeline.py
├── sample\_predict.py
├── analysis/
│   ├── EDA.ipynb
│   └── analyze\_src/
│       ├── basic\_data\_inspection.py
│       ├── bivariate\_analysis.py
│       ├── missing\_values\_analysis.py
│       ├── multivariate\_analysis.py
│       └── univariate\_analysis.py
├── explanations/
│   ├── factory\_design\_patter.py
│   ├── strategy\_design\_pattern.py
│   └── template\_design\_pattern.py
├── extracted\_data/
├── pipelines/
│   ├── deployment\_pipeline.py
│   └── training\_pipeline.py
├── src/
│   ├── data\_splitter.py
│   ├── feature\_engineering.py
│   ├── handle\_missing\_values.py
│   ├── ingest\_data.py
│   ├── model\_building.py
│   ├── model\_evaluator.py
│   └── outlier\_detection.py
└── steps/
├── data\_ingestion\_step.py
├── data\_splitter\_step.py
├── feature\_engineering\_step.py
├── handle\_missing\_values\_step.py
├── model\_building\_step.py
├── model\_evaluator\_step.py
├── model\_loader.py
├── outlier\_detection\_step.py
├── prediction\_service\_loader.py
└── predictor.py

````

### 3.2 Design Patterns

- **Strategy Pattern** – For interchangeable logic like feature transformations, model building, data splitting, etc.
- **Factory Pattern** – For dynamic object creation (e.g., selecting a data ingestor).
- **Template Method Pattern** – For reusable analysis workflows (e.g., missing values analysis).
- **Pipeline Pattern** – Entire ML lifecycle structured via ZenML pipelines.

Example (Strategy Pattern for Feature Engineering):

```python
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for feature in self.features:
            if feature in df_copy.columns:
                mask = df_copy[feature] > 0
                df_copy.loc[mask, feature] = np.log(df_copy.loc[mask, feature])
        return df_copy
````

---

## 4. Installation

```bash
# Clone the repository
git clone https://github.com/username/price-predictor.git
cd price-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

---

## 5. Usage

### 5.1 Training Pipeline

```bash
python run_pipeline.py
```

View MLflow experiment tracking:

```bash
mlflow ui --backend-store-uri 'path/to/mlflow/tracking'
```

### 5.2 Deployment Pipeline

```bash
python run_deployment.py
```

To stop the deployed service:

```bash
python run_deployment.py --stop-service
```

### 5.3 Making Predictions

```bash
python sample_predict.py
```

Example prediction via REST API:

```python
import json
import requests

url = "http://127.0.0.1:8000/invocations"
input_data = {
    "dataframe_records": [
        {
            "Order": 1,
            "PID": 5286,
            "MS SubClass": 20,
            "Lot Frontage": 80.0,
            "Lot Area": 9600,
            "Mo Sold": 5,
            "Yr Sold": 2010,
            # ... other features
        }
    ]
}
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(input_data))

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Error: {response.status_code}\n{response.text}")
```

---

## 6. Technologies

* 🧪 **ZenML** – ML pipeline orchestration
* 📊 **MLflow** – Experiment tracking & model deployment
* 🧠 **Scikit-learn** – Machine learning algorithms
* 📈 **Pandas**, **NumPy** – Data manipulation
* 🎨 **Matplotlib**, **Seaborn** – Data visualization

---

## 7. Model Details

* **Model**: Linear Regression (`scikit-learn`)
* **Preprocessing**:

  * Missing values: imputation (mean / mode)
  * Outlier detection and removal
  * Log transformation of skewed features
  * Standard scaling and one-hot encoding
* **Evaluation**: Mean Squared Error (MSE), R² Score
* **Tracking**: All runs and parameters tracked using **MLflow**

---

## 8. Exploratory Data Analysis

Located in the `analysis/` folder:

* ✅ Basic inspection, summary stats
* 🔍 Missing value analysis
* 📊 Univariate & bivariate analysis
* 🔄 Multivariate relationships

**Key insights:**

* 2930 entries, 82 features
* Features like `Gr Liv Area` and `Overall Qual` highly correlated with `SalePrice`
* `SalePrice` is skewed — log transformation applied
* Missing values notable in `Alley`, `Pool QC`, `Misc Feature`

---

## 9. Contributing

Contributions welcome!

Steps to contribute:

```bash
# Fork the repo
# Create a new branch
git checkout -b feature-branch

# Make changes and commit
git commit -am "Add new feature"

# Push to your fork
git push origin feature-branch

# Open a Pull Request on GitHub
```

---

## 10. License

This project is licensed under the **Apache 2.0 License**.
See the [LICENSE](LICENSE) file for more details.

---

**Developed by: Mohammad Afroz Ali**
© 2025


