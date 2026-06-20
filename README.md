# Wellness Tourism Package Prediction - MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for "Visit with Us" travel company to predict which customers are likely to purchase the newly introduced Wellness Tourism Package. The pipeline automates data preparation, model training, experimentation tracking, and deployment using Hugging Face and GitHub Actions.

**Last Updated:** June 20, 2026

## Project Overview

This project implements an end-to-end MLOps pipeline for "Visit with Us" travel company to predict which customers are likely to purchase the newly introduced Wellness Tourism Package. The pipeline automates data preparation, model training, experimentation tracking, and deployment using Hugging Face and GitHub Actions.

## Business Context

"Visit with Us" faces challenges in targeting the right customers efficiently for their new Wellness Tourism Package. This automated system integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies.

## Project Structure

```
Project/
├── master/
│   └── data/
│       └── tourism.csv                # Raw customer dataset
├── .github/
│   └── workflows/
│       └── mlflow_ci_cd.yml           # GitHub Actions CI/CD pipeline
├── register_data.py                   # Register dataset to Hugging Face
├── data_preparation.py                # Clean, split, and upload prepared data
├── model_training.py                  # Train, tune, and evaluate models
├── app.py                             # Streamlit application for inference
├── utils.py                           # Utility functions for the app
├── deploy_to_huggingface.py           # Deploy to Hugging Face Space
├── Dockerfile                         # Docker configuration for deployment
├── requirements.txt                   # Python dependencies for training
├── requirements-deployment.txt        # Python dependencies for deployment
└── README.md                          # This file
```

## Dataset Features

The customer dataset includes the following features:
- `age`: Customer age
- `gender`: Customer gender (M/F)
- `income`: Annual income
- `travel_frequency`: Number of trips per year
- `previous_trips`: Total previous trips
- `travel_spend`: Average travel spending
- `interest_adventure`: Interest in adventure tourism (0/1)
- `interest_culture`: Interest in cultural tourism (0/1)
- `interest_relaxation`: Interest in relaxation tourism (0/1)
- `interest_wellness`: Interest in wellness tourism (0/1)
- `loyalty_member`: Loyalty program membership (0/1)
- `marketing_emails_clicked`: Number of marketing emails clicked
- `website_visits`: Number of website visits
- `last_booking_months`: Months since last booking
- `purchased_wellness_package`: Target variable (0/1)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Hugging Face Setup

1. Create a Hugging Face account at https://huggingface.co/
2. Generate an access token: https://huggingface.co/settings/tokens
3. Set the `HF_TOKEN` environment variable or update the token in the scripts

```bash
export HF_TOKEN="your_huggingface_token"
```

### 3. Register Dataset to Hugging Face

```bash
python register_data.py
```

This uploads the raw dataset to: `visit-with-us/wellness-tourism-customers`

### 4. Prepare Data

```bash
python data_preparation.py
```

This:
- Cleans the data (removes customer_id, encodes gender)
- Splits into train/test sets (80-20)
- Uploads prepared data to: `visit-with-us/wellness-tourism-customers-prepared`

### 5. Train Models

```bash
python model_training.py
```

This:
- Trains 4 models: Decision Tree, Random Forest, XGBoost, Gradient Boosting
- Performs hyperparameter tuning using GridSearchCV
- Logs parameters and metrics to MLflow
- Selects the best model based on F1 score
- Registers the best model to Hugging Face: `visit-with-us/wellness-tourism-predictor`

## Models Trained

1. **Decision Tree** - Simple interpretable model
2. **Random Forest** - Ensemble method with bagging
3. **XGBoost** - Gradient boosting optimized for performance
4. **Gradient Boosting** - Sequential ensemble learning

## Experimentation Tracking with MLflow

All model training runs are tracked using MLflow:
- Parameters logged for each model
- Metrics: Accuracy, Precision, Recall, F1 Score
- Models saved and versioned
- View runs: `mlflow ui`

## Model Deployment

The project includes a Dockerized Streamlit application for model inference.

### Docker Configuration

The `Dockerfile` defines:
- Python 3.9 slim base image
- System dependencies installation
- Python dependencies from `requirements-deployment.txt`
- Streamlit server configuration (port 8501)

### Streamlit Application

The `app.py` provides:
- Interactive customer profile input via sidebar
- Real-time prediction using the trained model
- Probability breakdown visualization
- Model information and feature details

### Deployment to Hugging Face Space

To deploy the application to Hugging Face:

```bash
python deploy_to_huggingface.py
```

This script:
- Creates or updates the Hugging Face Space
- Uploads all deployment files (app.py, utils.py, Dockerfile, requirements-deployment.txt)
- Creates a README for the space

The deployed app will be available at: https://huggingface.co/spaces/visit-with-us/wellness-tourism-predictor

## GitHub Actions CI/CD Pipeline

The `.github/workflows/mlflow_ci_cd.yml` file defines an automated pipeline that:

1. **Data Preparation Job**
   - Caches pip packages for faster builds
   - Installs dependencies
   - Logs into Hugging Face
   - Registers raw dataset
   - Prepares and uploads train/test data

2. **Model Training Job**
   - Trains all models with hyperparameter tuning
   - Logs experiments to MLflow
   - Uploads MLflow artifacts with 30-day retention

3. **Model Deployment Job**
   - Installs deployment dependencies
   - Deploys Streamlit app to Hugging Face Space
   - Reports deployment status and URL

4. **Quality Checks Job**
   - Runs black code formatter
   - Runs flake8 linter for syntax errors
   - Ensures code quality standards

### Setting up GitHub Actions

1. Add `HF_TOKEN` as a GitHub Secret in your repository settings
2. Push the code to GitHub
3. The pipeline will automatically run on push/PR to main/master branches
4. Manual trigger is also available via workflow_dispatch

## Hugging Face Repositories

- **Dataset (Raw)**: `visit-with-us/wellness-tourism-customers`
- **Dataset (Prepared)**: `visit-with-us/wellness-tourism-customers-prepared`
- **Model**: `visit-with-us/wellness-tourism-predictor`

## Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate (minimize false positives)
- **Recall**: True positive coverage (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall (primary metric)

## Usage Example

```python
from huggingface_hub import hf_hub_download
import joblib

# Download the trained model
model_path = hf_hub_download(
    repo_id="visit-with-us/wellness-tourism-predictor",
    filename="wellness-tourism-predictor.pkl"
)

# Load the model
model = joblib.load(model_path)

# Make predictions
customer_features = [[45, 1, 85000, 3, 12, 4500, 1, 1, 0, 1, 1, 5, 20, 6]]
prediction = model.predict(customer_features)
print(f"Purchase probability: {prediction[0]}")
```

## Future Enhancements

- Add more advanced models (Neural Networks, LightGBM)
- Implement feature importance analysis
- Add SHAP values for model interpretability
- Create a prediction API endpoint
- Add automated retraining pipeline
- Implement A/B testing framework

## Requirements

- Python 3.9+
- Hugging Face account and access token
- Git and GitHub account
- (Optional) MLflow server for remote tracking

## License

This project is part of the MLOps coursework for "Visit with Us" travel company.
