#!/usr/bin/env python
###################################
#Date: July 8, 2016
#Purpose: Practice Random Forest
#Data: titanic
#source: https://www.youtube.com/watch?v=0GrciaGYzV0
#source(Bokeh): https://www.youtube.com/watch?v=Mz1AXUE0nR4
###################################

###################################
#Imports
###################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import preprocessing

import bokeh.charts
import bokeh.charts.utils
import bokeh.io 
import bokeh.models
import bokeh.palettes
import bokeh.plotting
from bokeh.charts import color, marker
from bokeh.charts import Line, Bar, output_file, show
from bokeh.sampledata.autompg import autompg as df

#Import Data
X = pd.read_csv('nature.csv')
X2 = pd.read_csv('nature.csv')
results = pd.DataFrame()
#X.drop(['name'], 1, inplace=True)
# X.convert_objects(convert_numeric=True)
le = preprocessing.LabelEncoder()

X2 = X2.drop(['individual'], axis = 1)
X2 = X2.drop(['chromosome'], axis = 1)
X2 = X2.drop(['sv_type'], axis = 1)

#to convert into numbers
X2['individual'] = le.fit_transform(X.individual)
# print (X.individual)
X2['chromosome'] = le.fit_transform(X.chromosome)
X2['sv_type'] = le.fit_transform(X.sv_type)


# #to convert back
# X.score = le.inverse_transform(X.score)

# X['sv_type'] = X['sv_type'].convert_objects(convert_numeric=True)

y = X2.pop("sv_type")

print (X2.describe())
 

###################################
#Data Munging
###################################

#Select all of the numerical variables and ignore the categorical variables
numeric_variables = list(X2.dtypes[X2.dtypes != "object"].index)
X[numeric_variables].head()

model = RandomForestRegressor(10, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X2,y)
print ("C-stat: ", roc_auc_score(y, model.oob_prediction_))
results['numeric_variables'] = roc_auc_score(y, model.oob_prediction_)


###################################
#Determine variable importance
###################################
model.feature_importances_

#Bokeh
feature_importances = pd.Series(model.feature_importances_, index=X2.columns)
print feature_importances
# # future = feature_importances.sort_values(inplace=True)
# p = Bar(feature_importances, ylabel="Feature Importance", legend=None, tools='pan,resize', logo=None)
# output_file("feature.html")
# show(p)
# feature_importances.plot(kind ='bar')


def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
	'''
	Author: Mike Bernico
	Purpose:
	Graphs the feature importances of a random decision forest using a horizontal bar chart.

	Parameters
	----------
	ensemble = Name of the ensemble whose features you would like graphed.
	feature_names = A list of the names of those features, displayed onthe Y axis.
	autoscale = True (Automatically adjust the X axis size to the largest feature +.headroom) / False = scale from 0 to 1
	headroom = used with auroscale, 0.05 default
	width=figure width in inches
	summarized_columns = a list of column prefixes to summarize on, for dummy variables (e.g. ["day_"] would summarize all )
	
	'''
	if autoscale:
		x_scale = model.feature_importances_.max()+ headroom
	else:
		x_scale = 1
	feature_dict=dict(zip(feature_names, model.feature_importances_))


	if summarized_columns:
		#some dummy columns to be summarized
		for col_name in summarized_columns:
			#sum all of the features that contain col_name, store in temp sum_value
			sum_value = sum(x for i, x in feature_dict.items() if col_name in i)
			#now remove all keys that are part of col_name
			keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
			for i in keys_to_remove:
				feature_dict.pop(i)
			#lastly, read the summarized field
			feature_dict[col_name] = sum_value
	#Create a graph		
	results = pd.Series(feature_dict.values(), index=feature_dict.keys())
	p = Bar(results, ylabel="Feature Importance", legend=None, logo=None)
	output_file("featureImportance.html")
	show(p)

	print feature_dict
	feature_dict = pd.DataFrame(feature_dict)

	# reults.sort(axis=1)
	# results.plot(kind="barh", figsize=(width, len(results)/4), xlim=(0, x_scale))
# categorical_variables = ['chromosome', 'start', 'end', 'width', 'score', 'individual', 'sv_type']
# graph_feature_importances(model, X2.columns, summarized_columns=categorical_variables)

###################################
#Parameter Adjustment
###################################
'''
Parameters that will make your model better
-------------------------------------------
- n_estimators: number of trees in the forest; choose as high of a number as the computer can count
- max_features: number of features to consider when looking for the best split; 
- min_samples_leaf: minimum number of samplesin newly created leaves

Parameters that will make it easier to train your model
-------------------------------------------------------
n_jobs: determines if multiple processors should be used to trian and test the model; always set to: -1
amd %%timeit vs. if it is set to 1. it should be much faster (especially when many trees are trained)
'''

#n_jobs
model = RandomForestRegressor(10, oob_score=True, n_jobs=1, random_state=42)
model.fit(X2,y)


model = RandomForestRegressor(10, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X2,y)

#n_estimators
results = []
n_estimator_options = [3, 5, 10, 20,25,30,50]

for trees in n_estimator_options:
	model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
	model.fit(X2,y)
	print trees, "trees"
	roc = roc_auc_score(y, model.oob_prediction_)
	print "ROC", roc
	results.append(roc)
	print " "
tree_results = pd.DataFrame()
tree_results["tree_count"] = n_estimator_options
tree_results["ROC"] = results
print tree_results
tree_results.to_csv('tree_results.csv')

rocgraph = pd.Series(results, n_estimator_options)
p = Line(rocgraph, ylabel="Tree ROC Score", legend=None, logo=None)
output_file("trees.html")
show(p)

#max_features: number of variables considered at each split point
'''
Outcome is binary (0,1), but this is treated as a numerical Outcome
"auto", None -> use all variables at every single split
0.9 -> use 90% of variables
0.2 -> use 20% of variables
'''
results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
	model = RandomForestRegressor(n_estimators=10, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
	model.fit(X2,y)
	print (max_features, "option")
	roc = roc_auc_score(y, model.oob_prediction_)
	print ("ROC: ", roc)
	results.append(roc)
	print (" ")

maxfeat_results = pd.DataFrame()
maxfeat_results["max_features_options"] = max_features_options
maxfeat_results["ROC"] = results
print maxfeat_results
maxfeat_results.to_csv('maxfeat_results.csv')

maxfeat = pd.Series(results, max_features_options)
p = Bar(maxfeat, ylabel="Tree ROC Score", legend=None, logo=None)
output_file("max_feat.html")
show(p)

#min_samples_leaf
results = []
min_samples_leaf_options = [1,2,3,4,5,6,7,8,9,10,15,20]
for min_samples in min_samples_leaf_options:
	model = RandomForestRegressor(n_estimators=10,
							      oob_score=True, 
							      n_jobs=-1, 
							      random_state=42, 
							      max_features="auto",
							      min_samples_leaf=min_samples)
	model.fit(X2,y)
	print (min_samples, "min samples")
	roc = roc_auc_score(y, model.oob_prediction_)
	print ("ROC: ", roc)
	results.append(roc)
	print (" ")

minLeaf_results = pd.DataFrame()
minLeaf_results["min_samples_leaf_options"] = min_samples_leaf_options
minLeaf_results["ROC"] = results
print minLeaf_results
minLeaf_results.to_csv('minLeaf_results.csv')

minfeat = pd.Series(results, min_samples_leaf_options)
p = Line(minfeat, ylabel="Tree ROC Score", legend=None, logo=None)
output_file("leaf.html")
show(p)


#Final Model: parameters based on above adjustments
model = RandomForestRegressor(n_estimators=10, 
							  oob_score=True, 
							  n_jobs=-1, 
							  random_state=42, 
							  max_features="auto", 
							  min_samples_leaf=15)

model.fit(X2,y)
roc = roc_auc_score(y, model.oob_prediction_)
print ("Final model ROC: ", roc)