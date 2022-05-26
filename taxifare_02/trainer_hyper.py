import mlflow
from taxifare_02.data import get_data, holdout, clean_df, get_data_using_blob, save_model_locally, save_model_to_gcp
from taxifare_02.model import get_model
from taxifare_02.pipeline import set_pipeline
from taxifare_02.metrics import compute_rmse
from taxifare_02.mlflo import MLFlowBase

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib
import hypertune

import argparse


class Trainer():
    
    """TRAINER CLASS FOR HYPER PARAM TRAINING ON GCP"""
    
    def __init__(self):
        pass
        
    def get_args(self):
        
        """GET ARGUMENTS FROM CONFIG.YML"""
        
        # Create the argument parser for each parameter plus the job directory
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--job-dir',  # Handled automatically by AI Platform
            help='GCS location to write checkpoints and export models',
            required=True
            )
        parser.add_argument(
            '--alpha',  # Specified in the config file
            help='Constant that multiplies the regularization term',
            default=0.0001,
            type=float
            )
        parser.add_argument(
            '--max_iter',  # Specified in the config file
            help='Max number of iterations.',
            default=1000,
            type=int
            )
        parser.add_argument(
            '--penalty',  # Specified in the config file
            help='The penalty (aka regularization term) to be used',
            default='l2',
            type=str
            )

        args = parser.parse_args()
        
        return args

    def run(self, model):
        
        """RUN MODEL"""

        # get model name
        str_model = str(model).split(".")[-1].replace("'>", "")
        print(str_model)
        
        #get args from yml file
        args = self.get_args()

        # get data and split
        df = clean_df(get_data_using_blob(1000))

        X_train, X_test, y_train, y_test = holdout(df)
        
        #instantiate model with params from args
        mdl = model(
                alpha=args.alpha,
                max_iter=args.max_iter,
                penalty=args.penalty
                )

        # set pipeline with model
        pipe = set_pipeline(mdl)

        # fit pipe
        pipe.fit(X_train, y_train)

        # get y_pred
        y_pred = pipe.predict(X_test)

        # compute rmse
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        
        #instantiate hypertune class
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='rmse',
                metric_value=rmse,
                global_step=1000
        )

        # save model locally
        save_model_locally(pipe, str_model)
        
        #save model on gcp
        save_model_to_gcp(str_model)


if __name__ == "__main__":

    tr = Trainer()
    tr.run(SGDRegressor)
