'''
Date: 9/4/2020
Author: Devendra Dahal
Update:
Description: Develop xgboost classifier model reading a large datafile. Write model accuracy results 
	in a text file and also save the model as pkl file.
USAGE: 
'''
import os,sys,traceback, datetime,time, itertools,random, math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import xgboost as xgb
import pickle

print (sys.version)
# Start Timer
t1 = datetime.datetime.now()

def runPro(data):
	try:
		
		test = data.replace('.data','.test')
		OutModel = data.replace('.data','_xgbst.pkl')

		if not os.path.exists(OutModel):
			print('\nBuilding XGBoost model ' + os.path.basename(OutModel) +'...')

			outfile = data.replace('.data','_xgboost.out')
			outTxt = open(outfile, 'w')
			outTxt.write("---------Model accuracy---------\nDatafile: {} \n\n".format(data))
			
			#Open CSV using numpy
			Indata = pd.read_csv(data, header = None)
			
			# Set up data correctly..e.g. skip first three and last columns for x
			# use only last column for y
			X = Indata.iloc[:,3:-1].values
			Y = Indata.iloc[:,-1:].values.ravel()
			
			if os.path.basename(test):
				Intest = pd.read_csv(test, header = None)
				
				X_test = Intest.iloc[:,3:-1].values
				Y_test = Intest.iloc[:,-1:].values.ravel()
			else:
				X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

			xgb_model = xgb.XGBRegressor(objective='reg:squarederror',nthread = -1)
			
			## opitmize the model using gridsearch 
			clf = GridSearchCV(xgb_model,
				   {'max_depth': [2,3,4,5],
					'n_estimators': [50,75,100,150,175,200]}, verbose=1)
			results = clf.fit(X,Y)
			
			## Check best parameters for this dataset after model optimization 
			print("Best: %f using %s" % (results.best_score_, results.best_params_))
			outTxt.write("Best Score: {} using {} \n-------------\n".format(round(results.best_score_,3), results.best_params_))
			
			## write model accuracy results to the out text file
			outTxt.write("\n\n------------\n")
			outTxt.write("Train MeanAE: {} \n".format(round(metrics.mean_absolute_error(Y, results.predict(X)),3)))
			outTxt.write("Test MeanAE: {} \n".format(round(metrics.mean_absolute_error(Y_test, results.predict(X_test)),3)))
			
			outTxt.write("Train MedAE: {} \n".format(round(metrics.median_absolute_error(Y, results.predict(X)),3)))
			outTxt.write("Test MedAE: {} \n".format(round(metrics.median_absolute_error(Y_test, results.predict(X_test)),3)))
			
			outTxt.write("Train r: {} \n".format(round(math.sqrt((metrics.r2_score(Y, results.predict(X)))),3)))
			outTxt.write("Test r: {} \n".format(round(math.sqrt(metrics.r2_score(Y_test, results.predict(X_test))),3)))
			
			## save model as pkl file.
			joblib.dump(results, OutModel, compress=9)
			
		else:
			print('\n' + os.path.basename(OutModel) + ' already exists...')
			
	except:
		print ("Processed halted on the way.")
		print (traceback.format_exc())


data = sys.argv[1]

runPro(data)

t2 = datetime.datetime.now()
print (t2.strftime("%Y-%m-%d %H:%M:%S"))
tt = t2 - t1
print ("\nProcessing time: " + str(tt))