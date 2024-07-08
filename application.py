from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)
app=application
if not app.debug:
    import logging
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('error.log', maxBytes=1024 * 1024 * 100, backupCount=20)
    file_handler.setLevel(logging.ERROR)
    app.logger.addHandler(file_handler)
##route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))

        except Exception as e:
            app.logger.error(f"Exception occurred: {e} Error 1")
            return str(e),500
        pred_df=data.get_data_as_frame()
        pred_df['race/ethnicity']=pred_df['race_ethnicity']
        pred_df['parental level of education']=pred_df['parental_level_of_education']
        pred_df['test preparation course']=pred_df['test_preparation_course']
        pred_df['writing score']=pred_df['writing_score']
        pred_df['reading score']=pred_df['reading_score']
        pred_df.drop(columns=['race_ethnicity','parental_level_of_education','test_preparation_course','writing_score','reading_score'],inplace=True)
        print(pred_df)

        try:
            predict_pipeline =PredictPipeline()
            result=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=result[0])
        except Exception as e:
            app.logger.error(f"Exception occurred: {e} Error 2")
            return str(e),500

if __name__=="__main__":
    app.debug=True
    app.run()
