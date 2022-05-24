import mlflow
from taxifare_02.data import get_data, holdout, clean_df
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
            "[PT] [LISBO] [MDK] TaxiFareRecap + 2",
            "https://mlflow.lewagon.ai")
    
    def run(self, model, **params):
        
        #get model name
        str_model = str(model).split('.')[-1].replace("'>", '')
        
        #get data and split

        
        #get model

        
        #set pipeline with model

        
        #create run and log param

        
        #loop over parameters and log

        
        #fit pipe

        
        #get y_pred

        
        #compute rmse

        
        #log rmse metric

        
        #save model

                
        

if __name__ == "__main__":
    
    model_params_knn = dict(n_neighbors=10, leaf_size=10)
    
    model_params_rf = dict(
        n_estimators=300,
        max_depth=2)
    
    for params, model in zip([model_params_knn, model_params_rf], [KNeighborsRegressor, RandomForestRegressor]):
        tr = Trainer()
        tr.run(model, **params)
