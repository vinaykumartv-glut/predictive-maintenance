# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

#from google.colab import userdata
#access_key = userdata.get('MY_API_KEY')
access_key = os.getenv("HF_TOKEN")
# Define constants for the dataset and output paths
api = HfApi(token = access_key)

DATASET_PATH = "hf://datasets/vinaykumartv/predictive-maintenance/enginedata.csv"

data = pd.read_csv(DATASET_PATH, index_col = 0)

print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = "Engine Condition"

numerical_columns = data.select_dtypes(exclude = "object").columns.tolist()

# List of categorical features in the dataset
categorical_columns = data.select_dtypes(include = "category").columns.tolist()

# Define predictor matrix (X) using selected numeric and categorical features
X = data[numerical_columns + categorical_columns]

# Define target variable
y = data[target]

X.drop([target], axis = 1, inplace = True)

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size = 0.2,     # 20% of the data is reserved for testing
    random_state = 42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv", index = False)
Xtest.to_csv("Xtest.csv", index = False)
ytrain.to_csv("ytrain.csv", index = False)
ytest.to_csv("ytest.csv", index = False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj = file_path,
        path_in_repo = file_path.split("/")[-1],  # just the filename
        repo_id = "vinaykumartv/predictive-maintenance",
        repo_type = "dataset",
    )
