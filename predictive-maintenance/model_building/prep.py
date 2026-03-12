# for data manipulation

#Importing needed libraries
from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

# 3.1 Load the dataset directly from the Hugging Face data space:
dataset = load_dataset("ksricheenu/Predictive-Maintenance-Prediction", split="train")

# 3.2. Perform data cleaning and remove any unnecessary columns:
There are several variables which contain outliers, we need to retain them because the data represents real engine sensor measurements, where extreme values may correspond to abnormal operating conditions or early failure signals. Removing such data could lead us to miss important predictive information. So, we can preserve the outliers for our model training. According to the above stats, we don't need to perform any cleanup/removal of data.

# converting to pandas
df = dataset.to_pandas()

# 3.3. Split the cleaned dataset into training and testing sets, and save them locally:
X = df.drop("Engine Condition", axis=1)
y = df["Engine Condition"]

# Train test split (Splitting the data for Train/Test with the ratio of 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# As the cleansed dataset has to be trained and tested, and has to be saved locally.
train_df = X_train.copy()
train_df["Engine Condition"] = y_train

test_df = X_test.copy()
test_df["Engine Condition"] = y_test

#Saving the train and test data locally
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train/Test CSV files created")

# 3.4. Uploading the train and test dataset back to the Hugging Face dataset repo.
for file in ["train.csv","test.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="ksricheenu/Predictive-Maintenance-Prediction",
        repo_type="dataset",
    )
