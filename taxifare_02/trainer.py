import mlflow
from taxifare_02.data import get_data, holdout, clean_df, get_data_using_blob, save_model_locally, save_model_to_gcp
from taxifare_02.model import get_model
from taxifare_02.pipeline import set_pipeline
from taxifare_02.metrics import compute_rmse
from taxifare_02.mlflo import MLFlowBase

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import joblib



class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(
            "[PT] [LISBO] [MDK] TaxiFareRecap + 3", "https://mlflow.lewagon.ai"
        )
        
        
    def run(self, model, **params):

        # get model name
        str_model = str(model).split(".")[-1].replace("'>", "")
        print(str_model)

        # get data and split
        df = clean_df(get_data_using_blob(1000))

        X_train, X_test, y_train, y_test = holdout(df)

        # get model
        mdl = get_model(model)

        # set pipeline with model
        pipe = set_pipeline(mdl)

        # create run and log param
        self.mlflow_create_run()
        self.mlflow_log_param("model", str_model)

        # loop over hyperparameters and log
        for k, v in params.items():
            self.mlflow_log_param(k, v)

        # fit pipe
        pipe.fit(X_train, y_train)

        # get y_pred
        y_pred = pipe.predict(X_test)

        # compute rmse
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)

        # log rmse metric
        self.mlflow_log_metric("rmse", rmse)

        # save model locally
        save_model_locally(pipe, str_model)
        
        #save model on gcp
        save_model_to_gcp(str_model)


if __name__ == "__main__":

    model_params_knn = dict(n_neighbors=10, leaf_size=10)

    model_params_rf = dict(n_estimators=300, max_depth=2)

    for params, model in zip(
        [model_params_knn, model_params_rf],
        [KNeighborsRegressor, RandomForestRegressor],
    ):
        tr = Trainer()
        tr.run(model, **params)
