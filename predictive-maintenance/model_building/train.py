import pandas as pd
import joblib
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Loading the train and test datasets
train_url = "https://huggingface.co/datasets/ksricheenu/Predictive-Maintenance-Prediction/resolve/main/train.csv"
test_url = "https://huggingface.co/datasets/ksricheenu/Predictive-Maintenance-Prediction/resolve/main/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# Separate features and target
X_train = train_df.drop("Engine Condition", axis=1)
y_train = train_df["Engine Condition"]

X_test = test_df.drop("Engine Condition", axis=1)
y_test = test_df["Engine Condition"]


# Train Model (Final Selected Model: AdaBoost)
model = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.2,
    estimator= DecisionTreeClassifier(max_depth=3),
    random_state=1
)
model.fit(X_train, y_train)
print("Model training completed")


# Evaluating the Model
y_pred = model.predict(X_test)

print("\nModel Performance on Test Data")
print("--------------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))


# Saving the Model
model_file = "vehicle_engine_failure_adaboost_model.joblib"
joblib.dump(model, model_file)
print("\nModel saved locally")


# Upload Model to Hugging Face Model Hub
repo_id = "ksricheenu/vehicle-engine-failure-prediction-adaboost"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=repo_id,
    repo_type="model"
)

print("\nModel uploaded successfully to Hugging Face Hub")
