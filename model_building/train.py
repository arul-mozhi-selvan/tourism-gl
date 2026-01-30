# for data manipulation
import pandas as pd
import os
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# for model serialization
import joblib

# for MLflow logging (local file-based tracking)

# -------------------------
# Configuration / HF dataset paths
# -------------------------
# Replace HF usernames/repos if different
HF_DATASET_REPO = "arulmozhiselvan/superkart"
XTRAIN_FILENAME = "Xtrain.csv"
XTEST_FILENAME  = "Xtest.csv"
YTRAIN_FILENAME = "ytrain.csv"
YTEST_FILENAME  = "ytest.csv"

# -------------------------
# Download dataset splits from Hugging Face dataset repo (requires HF_TOKEN in env)
# -------------------------
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

# hf_hub_download returns local path to the file
Xtrain_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=XTRAIN_FILENAME, repo_type="dataset", token=hf_token)
Xtest_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename=XTEST_FILENAME,  repo_type="dataset", token=hf_token)
ytrain_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=YTRAIN_FILENAME, repo_type="dataset", token=hf_token)
ytest_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename=YTEST_FILENAME,  repo_type="dataset", token=hf_token)

# Read CSVs
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest  = pd.read_csv(ytest_path)

# Convert y DataFrame -> Series if needed
if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1:
    ytrain = ytrain.iloc[:, 0]
if isinstance(ytest, pd.DataFrame) and ytest.shape[1] == 1:
    ytest = ytest.iloc[:, 0]

print("Data loaded:")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain.shape, "ytest:", ytest.shape)

# -------------------------
# Feature selection (adjust to match your dataset)
# -------------------------
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year',
    'Product_Store_Sales_Total'
    ]
numeric_features = [c for c in numeric_features if c in Xtrain.columns]

categorical_features = [c for c in [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
    ] if c in Xtrain.columns]

print("Numeric features used:", numeric_features)
print("Categorical features used:", categorical_features)

# -------------------------
# Class weight to handle imbalance (guarded)
# -------------------------
pos = int(ytrain.value_counts().get(1, 0))
neg = int(ytrain.value_counts().get(0, 0))
if pos == 0 or neg == 0:
    class_weight = 1.0
else:
    class_weight = neg / pos
print("Class weight (neg/pos):", class_weight)

# -------------------------
# Preprocessing pipeline
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    remainder='drop'
)

# -------------------------
# Define XGBoost model
# -------------------------
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Hyperparameter grid (same style as your example)
param_grid = {
    'xgbregressor__max_depth': [3, 5, 7, 10],
    'xgbregressor__n_estimators': [100, 200, 500],
    'xgbregressor__learning_rate': [0.01, 0.1, 0.2]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------
# Configure MLflow to use local file-based tracking (development)


# -------------------------
# Grid search with cross-validation and MLflow logging
# -------------------------

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Save best model locally
model_filename = "best_model_v1.joblib"
joblib.dump(best_model, model_filename)
print("Model saved as", model_filename)


# -------------------------
# Upload saved model to Hugging Face Model Hub (new model repo)
# -------------------------
repo_id = "arulmozhiselvan/superkart-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the model repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{repo_id}' not found. Creating new model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Model repo '{repo_id}' created.")

# Upload the model file
try:
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=repo_id,
        repo_type=repo_type,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Uploaded {model_filename} to {repo_id}")
except HfHubHTTPError as e:
    print("Failed to upload model to Hugging Face:", str(e))

print("Training script finished.")
