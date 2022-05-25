import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
from termcolor import colored
import joblib
import os

BUCKET_NAME = "wagon-data-867-dk"
BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"
BUCKET_MODEL_PATH = "models/TaxiFareModel"

def get_data():
    url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    df = pd.read_csv(url, nrows=100)
    print(colored(f"data retrieved from AWS URL {url}", "blue"))
    return df


def get_data_using_blob(line_count):

    # get data from my google storage bucket

    data_file = "train_1k.csv"

    client = storage.Client()  # verifies $GOOGLE_APPLICATION_CREDENTIALS

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)
    
    print(colored(f"data retrieved from GCP URL {BUCKET_NAME + '/'+BUCKET_TRAIN_DATA_PATH}", "red"))

    blob.download_to_filename(data_file)

    # load downloaded data to dataframe
    df = pd.read_csv(data_file, nrows=line_count)
    
    os.remove(data_file)

    return df

def save_model_locally(pipe, model_name):
    print(colored("model.joblib saved locally", "green"))
    return joblib.dump(pipe, f"{model_name}.joblib")


def save_model_to_gcp(model_name):

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(f"{BUCKET_MODEL_PATH}/{model_name}.joblib")
    
    print(colored("model.joblib saved GCP", "white"))
    blob.upload_from_filename(f"{model_name}.joblib")
    

def clean_df(df):
    df = df.dropna(how="any", axis="rows")
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 1]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def holdout(df):

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    return X_train, X_test, y_train, y_test
