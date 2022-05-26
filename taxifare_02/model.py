from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def get_model(model, **params):
    
    """INSTANTIATE MODEL WITH PARAMS IF NEEDED"""

    if params:
        return model(**params)

    else:
        return model()
