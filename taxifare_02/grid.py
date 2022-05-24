from taxifare_02.trainer import Trainer
from taxifare_02.mlflo import MLFlowBase
from sklearn.model_selection import GridSearchCV
from taxifare_02.data import clean_df, get_data, holdout
from taxifare_02.model import get_model
from taxifare_02.pipeline import set_pipeline
from taxifare_02.metrics import compute_rmse
from sklearn.metrics import make_scorer

class TaxiFareGrid(MLFlowBase):
    def __init__(self):
        super().__init__(
            "[PT] [LISBO] [MDK] TaxiFareRecap + 2",
            "https://mlflow.lewagon.ai")
    
    
    def grid(self, **params):
        
        #split **kwargs in model and hyper params

        
        #get and split the data

        
        # loop over both model/hyper
        for model, param_grid in zip(model, hypers):
            
            #get string of model
            str_model = str_model = str(model).split('.')[-1].replace("'>", '')
            print(str_model)

            
            #get model
            
            #set pipeline
            
            #create run and log model

            
            #instantiate GridSearch and fit

            
            #get best params

            
            #loop and log params gridsearch

            
            #score gridsearch

            
            #get best score

            
            #log score metric

            
            #save model
            