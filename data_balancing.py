'''
How to deal with unbalanced data in binary classification 

Unbalanced data is one of the most important problem during data preprocessing before machine learning training. 
'''


#####
import pandas as pd 
import numpy as np 
import sys 
import argparse
#####

def load_data(path): 
	data = pd.read_csv(path)

	return data 

def data_balance_test(data, labels_column):
	
	#count_classes = pd.value_counts(data[labels_column], sort = True).sort_index()
	# count_classes.plot(kind = 'bar')
	# plt.title("Fraud class histogram")
	# plt.xlabel("Class")
	# plt.ylabel("Frequency")

	count_classes = pd.DataFrame({'Total' : data.groupby([labels_column]).size()}).reset_index()

	# Find the minor class : 
	minor_class = count_classes.loc[count_classes.Total == min(count_classes.Total)][[labels_column]].reset_index(drop = True)
	minor_class = minor_class[labels_column].values[0]

	# Find the major class 
	major_class = count_classes.loc[count_classes.Total == max(count_classes.Total)][[labels_column]].reset_index(drop = True)
	major_class = major_class[labels_column].values[0]

	print("The major class is {} with {} data points".format(major_class, max(count_classes.Total)))
	print(10*"--")
	print("The minor class is {} with {} data points".format(minor_class, min(count_classes.Total)))
	print(10*"--")
	


	return count_classes, minor_class, major_class

def balance_data(data, labels_column, minor_class, major_class): 

	# Number of data points in the minority class
	number_record_minor = len(data[data[labels_column] == minor_class])
	record_minor_indices = np.array(data[data[labels_column] == minor_class].index)

	# Pick the indices of the major class 
	record_major_indices = data[data[labels_column] == major_class].index

	# Among major class we randomly select a dataset with same size as minor class 

	random_major_indices = np.random.choice(record_major_indices, number_record_minor, replace = False)
	random_major_indices = np.array(random_major_indices)

	# Appending the two indices 
	under_sample_indices = np.concatenate([record_minor_indices, random_major_indices])

	# Get the asociated data to under_sample_indices 
	under_sample_data = data.iloc[under_sample_indices,:]

	# Check the sizes of the classes : 
	print(pd.DataFrame({'Total' : under_sample_data.groupby([labels_column]).size()}).reset_index())

	return under_sample_data


class Parser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

if __name__ == "__main__": 

	parser = Parser()
	parser.add_argument("Path", help="data path")
	parser.add_argument("Class", help ="Name of the labels column")
	args = parser.parse_args()


	data = load_data(args.Path)
	count_classes, minor_class, major_class = data_balance_test(data,args.Class)

	under_sample_data = balance_data(data,args.Class, minor_class, major_class)
	under_sample_data.to_csv("credit_card_balance.csv")









