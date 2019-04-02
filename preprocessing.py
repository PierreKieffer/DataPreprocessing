'''
Preprocessing library to process database before machine learning 

1. Data loaders
2. Features selection 
	- identify_missing : Find features with missing values 
	- identify_collinear : Find highly correlated features
	- identify_features_importance : Find most important features for classification
	- identify_single_value_features : Find single value features
3. Data transformation
	- encode_data : transform object values to numericals
	- reverse_encode_data : back to object values
	- scale : scaling data
	- principal_components_analysis : transform features to (n) uncorrelated features




Author : Pierre Kieffer
'''

####
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

import seaborn as sns 
import matplotlib.pyplot as plt

import argparse 	
import pickle
import time
import os
####




####################
""" Data loaders """ 
####################

def load_data(path): 
	data = pd.read_csv(path, index_col = 0)
	return data


##########################
""" Features selection """
##########################

def identify_missing(data, threshold): 
	''' Identify the features with threshold % missing values 
	Return list of features with to threshold % missing values'''
 
	# Get the % of missing values for each feature
	missing = data.isnull().sum()/data.shape[0]
	missing_result = pd.DataFrame(missing).reset_index().rename(columns = {'index' : 'features', 0 :'missing_frac'})
	print(missing_result)
	missing_result = missing_result.sort_values(by ='missing_frac', ascending = False)

	# Features with threshold missing values
	missing_thres = missing_result[missing_result.missing_frac > threshold]
	features_to_drop = missing_thres.columns

	return features_to_drop




def identify_collinear(data, corr_threshold):
	''' Identify the most correlated features
	Return dataframe with correlated features '''

	# Correlation matrix
	corr = data.corr()

	# Extraction of the upper triangle of the correlation matrix 
	upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))

	# Features with correlation > corr_threshold
	features_to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_threshold)]
	print(features_to_drop)

	# Datafame with pairs of correlated features 
	collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
 
	for col in features_to_drop:
		# In upper, we get for each features_to_drop, the correlated features and we save the pair in collinear dataframe 

		# correlated features to features_to_drop  
		corr_features = list(upper.index[upper[col].abs() > corr_threshold])

		# Values 
		corr_values = list(upper[col][upper[col].abs() > corr_threshold])
		
		# Get drop_features
		drop_features = [col for _ in range(len(corr_features))]

		# extraction 
		temp_df = pd.DataFrame.from_dict({'drop_feature' : drop_features,'corr_feature' : corr_features, 'corr_value': corr_values})

		collinear = collinear.append(temp_df, ignore_index = True)

	return collinear




def identify_features_importance(data, label_column, threshold): 
	''' Identify most important features during a classification 
	return list of the most important features '''

	features = data.drop([label_column], axis = 1)
	labels = data[label_column]

	model = ExtraTreesClassifier()
	model.fit(features, labels)

	# Get features importances in %
	feature_importances = pd.DataFrame( columns = features.columns)
	feature_importances_values = model.feature_importances_ *100
	feature_importances.loc[len(feature_importances)] = feature_importances_values

	# Extaction of the most important features according to threshold : 
	for col in feature_importances:
		if feature_importances.loc[0, col] < threshold:
			feature_importances = feature_importances.drop([col], axis =1)

	return feature_importances.columns




def identify_single_value_features(data): 
	''' Identify features with single value 
	return a list of features '''  

	# Count unique values for each feature
	unique = data.nunique()
	unique = pd.DataFrame(unique).reset_index().rename(columns = {'index' : 'features', 0 :'nb_unique_values'})

	single_value_feature = []
	for i in range(len(unique)): 
		if unique.loc[i, 'nb_unique_values'] == 1 : 
			single_value_feature.append(unique.loc[i, features])

	return single_value_feature


###########################
""" Data transformation """
###########################

def encode_data(data, path): 
	''' Encode objects values into numerical values '''
	
	for col in data.columns : 
		if data[col].dtype == 'object' : 

			encoder = LabelEncoder()
			encoder.fit(data[col])
			data[col] = encoder.transform(data[col]) 

			# Save encoder
			encoder_file = path + '/' + col +'_encoder.sav'
			pickle.dump(encoder, open(encoder_file, 'wb'))
	return data 


def reverse_encode_data(data, path):
	''' Reverse function of encode_data 
	Return dataframe with objects values'''

	saved_encoders_path = path

	for encoder_file in os.listdir(saved_encoders_path): 

		# load encoder 
		encoder = pickle.load(open( saved_encoders_path + '/' + encoder_file, 'rb'))
		encoder_name = encoder_file[:-12]

		# Apply inverse encoding on associated feature 
		for col in data.columns: 
			if col == encoder_name:
				data[col] = encoder.inverse_transform(data[col])

	return data



def scale(data): 
	''' Scaling 
	Return scaled dataframe''' 

	scaler = StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	# Save the scaler 
	scalerfile = 'scaler.sav'
	pickle.dump(scaler, open(scalerfile, 'wb'))
	return data



def principal_comp_analysis(data, nb_comp, label_column): 
	''' Principal components analysis transformation 
	Return transformed dataframe with nb_comp features'''

	features_col = []
	for col in data.columns: 
		if col != label_column:
			features_col.append(col)

	features = data.drop([label_column], axis = 1)
	label_data = data[[label_column]]

	pca = PCA(n_components = nb_comp)
	pca.fit(features)
	data_pc = pca.transform(features)

	columns = []
	for i in range(nb_comp): 
		columns.append('pc{}'.format(i+1))

	df = pd.DataFrame(data = data_pc, columns = columns)
	df = pd.concat([df, label_data], axis =1)

	return df 



##########################
""" Data visualization """
##########################

def std_pairplot(data, hue=''):
	sns.pairplot(data, hue = hue)
	plt.show()



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("path")
	args = parser.parse_args()
	data = load_data(args.path)


