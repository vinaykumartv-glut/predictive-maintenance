# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000") # Set the tracking server URI
mlflow.set_experiment("mlops-predictive-maintenance-experiment") # Set the active MLflow experiment

# Retrieve Hugging Face API key from Colab user data
#from google.colab import userdata
#access_key = userdata.get('MY_API_KEY')
access_key = os.getenv("HF_TOKEN")

# Initialize Hugging Face API client
api = HfApi(token = access_key)

# Define paths to preprocessed data on Hugging Face Hub
Xtrain_path = "hf://datasets/vinaykumartv/predictive-maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/vinaykumartv/predictive-maintenance/Xtest.csv"
ytrain_path = "hf://datasets/vinaykumartv/predictive-maintenance/ytrain.csv"
ytest_path = "hf://datasets/vinaykumartv/predictive-maintenance/ytest.csv"

# Load the preprocessed data into pandas DataFrames
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# Ensure both float and int numeric columns are included
numerical_columns = Xtrain.select_dtypes(include=["float64","int64"]).columns.tolist()
categorical_columns = Xtrain.select_dtypes(include="category").columns.tolist()

# Calculate class weight to handle class imbalance
# This gives more weight to the minority class during training
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps using ColumnTransformer
# StandardScaler for numerical features, OneHotEncoder for categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown = 'ignore'), categorical_columns)
)

# Define the base XGBoost Classifier model
xgb_model = xgb.XGBClassifier(scale_pos_weight = class_weight, random_state = 42)

# Define the hyperparameter grid for GridSearchCV
# These parameters will be tuned to find the best model configuration
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of trees to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Create a pipeline that first preprocesses the data then applies the XGBoost model
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Validation check
expected = [
    "Engine rpm","Lub oil pressure","Fuel pressure",
    "Coolant pressure","lub oil temp","Coolant temp"
]
missing = [col for col in expected if col not in Xtrain.columns]
if missing:
    raise ValueError(f"Missing features before upload: {missing}")

# Start an MLflow run to track the entire training process
with mlflow.start_run():
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv = 5, n_jobs = -1) # 5-fold cross-validation
    grid_search.fit(Xtrain, ytrain) # Fit the GridSearchCV to the training data

    # Log all parameter combinations and their mean test scores to MLflow
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate nested MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log the best parameters found by GridSearchCV to the main MLflow run
    mlflow.log_params(grid_search.best_params_)

    # Retrieve the best model estimator from the grid search
    best_model = grid_search.best_estimator_

    # Define a classification threshold for converting probabilities to binary predictions
    classification_threshold = 0.45

    # Get predicted probabilities and apply threshold for binary predictions on training data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # Get predicted probabilities and apply threshold for binary predictions on testing data
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Generate classification reports for both training and testing data
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key evaluation metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the best trained model locally using joblib
    model_path = "best_predictive_maintenance_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the saved model as an artifact in MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Define Hugging Face repository details for model registration
    repo_id = "vinaykumartv/predictive-maintenance-model"
    repo_type = "model"

    # Check if the Hugging Face model repository exists, create it if it doesn't
    try:
        api.repo_info(repo_id = repo_id, repo_type = repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id = repo_id, repo_type = repo_type, private = False) # Create a public repository
        print(f"Space '{repo_id}' created.")

    # Upload the best model file to the Hugging Face repository
    api.upload_file(
        path_or_fileobj = "best_predictive_maintenance_model.joblib",
        path_in_repo = "best_predictive_maintenance_model.joblib",
        repo_id = repo_id,
        repo_type = repo_type,
    )
