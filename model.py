import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor

#Model training and metrics
#decide depth after
def xgboost_model(X,y,depth=6)->tuple:

    #Split test 80% and train 20%
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

    #xg boost tree generation, 
    tree_model = XGBRegressor(n_estimators=300,max_depth=depth,learning_rate=0.1,random_state=1,n_jobs=-1)
    #training on the train data
    tree_model.fit(X_train, y_train)

    #test on the test fraction
    y_predict = tree_model.predict(X_test)

    # regression metrics calucation
    R2 = r2_score(y_test, y_predict)
    MSE = mean_squared_error(y_test, y_predict)
    MAE = mean_absolute_error(y_test, y_predict)

    return (R2, MSE, MAE)


def best_model(X, y) -> tuple:
    max_R2 = -np.inf
    best_R2_depth = 0
    min_MSE = np.inf
    best_MSE_depth = 0
    min_MAE = np.inf
    best_MAE_depth = 0

    for new_depth in range(2, 10):
        R2, MSE, MAE = xgboost_model(X, y, new_depth)

        if R2 > max_R2:
            max_R2 = R2
            best_R2_depth = new_depth
        if MSE < min_MSE:
            min_MSE = MSE
            best_MSE_depth = new_depth
        if MAE < min_MAE:
            min_MAE = MAE
            best_MAE_depth = new_depth

    
    return best_R2_depth, best_MSE_depth, best_MAE_depth
#what is cross validation
#what is hyperparameter tuning
#what is the demention reduction




