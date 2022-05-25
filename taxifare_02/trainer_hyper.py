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


class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(
            "[PT] [LISBO] [MDK] TaxiFareRecap + 3", "https://mlflow.lewagon.ai"
        )
        
    def get_args(self):
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

        # get model name
        str_model = str(model).split(".")[-1].replace("'>", "")
        print(str_model)
        
        args = self.get_args()

        # get data and split
        df = clean_df(get_data_using_blob(1000))

        X_train, X_test, y_train, y_test = holdout(df)

        # get model
        #mdl = get_model(model)
        
        mdl = model(
                alpha=args.alpha,
                max_iter=args.max_iter,
                penalty=args.penalty
                )

        # set pipeline with model
        pipe = set_pipeline(mdl)

        # create run and log param
        # self.mlflow_create_run()
        # self.mlflow_log_param("model", str_model)

        # loop over hyperparameters and log
        # for k, v in self.args.items():
        #     self.mlflow_log_param(k, v)

        # fit pipe
        pipe.fit(X_train, y_train)

        # get y_pred
        y_pred = pipe.predict(X_test)

        # compute rmse
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='rmse',
                metric_value=rmse,
                global_step=1000
        )

        # log rmse metric
        # self.mlflow_log_metric("rmse", rmse)

        # save model locally
        save_model_locally(pipe, str_model)
        
        #save model on gcp
        save_model_to_gcp(str_model)


if __name__ == "__main__":

    tr = Trainer()
    tr.run(SGDRegressor)
