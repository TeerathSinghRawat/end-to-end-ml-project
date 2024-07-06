import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb')as file_ob:
            dill.dump(obj,file_ob)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        for i in range (len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,Y_train)
            Y_pred=model.predict(X_test)
            model_score=r2_score(Y_test,Y_pred)
            report[list(models.keys())[i]] = model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)

