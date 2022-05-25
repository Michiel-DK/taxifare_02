from taxifare_02.trainer import Trainer
from taxifare_02.mlflo import MLFlowBase
from sklearn.model_selection import GridSearchCV
from taxifare_02.data import clean_df, get_data, holdout
from taxifare_02.model import get_model
from taxifare_02.pipeline import set_pipeline
from taxifare_02.metrics import compute_rmse
from sklearn.metrics import make_scorer
import joblib


class TaxiFareGrid(MLFlowBase):
    def __init__(self):
        super().__init__(
            "[PT] [LISBO] [MDK] TaxiFareRecap + 3", "https://mlflow.lewagon.ai"
        )

    def grid(self, **params):

        # split **kwargs in model and hyper params
        models = params["model"]

        hypers = params["hyper"]

        # get and split the data
        df = clean_df(get_data())

        X_train, X_test, y_train, y_test = holdout(df)

        # loop over both model/hyper
        for model, param_grid in zip(models, hypers):

            # get string of model
            str_model = str_model = str(model).split(".")[-1].replace("'>", "")
            print(str_model)

            # get model
            mdl = get_model(model)

            # set pipeline
            pipe = set_pipeline(mdl)

            # create run and log model
            self.mlflow_create_run()
            self.mlflow_log_param("model", str_model)

            # instantiate GridSearch and fit
            grid_search = GridSearchCV(
                pipe, param_grid, cv=5, scoring=make_scorer(compute_rmse)
            )

            grid_search.fit(X_train, y_train)

            # get best params
            best_params = grid_search.best_params_
            print(best_params)

            # loop and log params gridsearch
            for k, v in best_params.items():
                self.mlflow_log_param(str(k), str(v))

            # score gridsearch
            grid_search.score(X_test, y_test)

            # get best score
            rmse = grid_search.best_score_

            # log score metric
            self.mlflow_log_metric("rmse", rmse)

            # save model
            joblib.dump(pipe, f"{str_model}.joblib")
