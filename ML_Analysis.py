from collections import Counter, defaultdict
from Acquisition import get_values_data
from Correlational_Analysis import pool_join_dirty_csvs, join_clean_csvs
from matplotlib import style
from sklearn import svm, neighbors, model_selection
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

import bs4 as bs
import Correlational_Analysis
import codecs
import csv
import datetime as dt
import functools
import numpy as np
import os
import pandas as pd
import pickle
import pprint as pp
import requests
import warnings

columns = ['security',
'actual_buys',
'actual_sales',
'best_algorithm',
'best_leaf_size',
'best_#_of_n_neighbors',
'accuracy',
'precision',
'recall',
'latest_prediction']

analysis_results = []

joined_close_scaled = pd.read_csv('Data_Acquisition/joined_close_scaled.csv')

def clean_data(ticker,df):

	'''
	Here all columns, except for Adj Close and Date columns are stripped 
	away. Then the Adj Close columns is renamed with the ticker's name, and 
	finally date is set as index.
	'''

	df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
	df.set_index('Date', inplace=True)
	df.rename(columns={"Adj Close":ticker}, inplace=True)
 
	return df


def setup_dataframes(ticker):

	'''
	Df is acquired from csv and then proceeds to be cleaned by the clean_data
	function.
	'''
	
	df=pd.read_csv('Data_Acquisition/securities_dfs/{}.csv'.format(ticker))
	df=clean_data(ticker,df)	

	return df


def process_data(ticker, correlated_security, print_tail=False):

	columns=['Date',ticker,correlated_security]
	# correlated_security.insert(0,ticker)
	# correlated_security.insert(0,'Date')

	security_of_int_df = joined_close_scaled.loc[:,columns].copy(deep=True)
	# print('Number of sorted_correlations: ',len(columns)-2)

	return security_of_int_df


def buy_sell_hold(*args):

	'''
	Function labeling changes in percentage values of:
		over +1% as 1 (buy)
		under -1% as -1 (sell)
		> than -1% & < than +1% as 0 (hold)
	'''

	cols=[c for c in args]
	requirement=2

	for col in cols:
		if col > requirement:
			return 1
		else:
			return 0



def make_labels(ticker, correlated_security):

	'''
	Here we map the buy_sell_hold_functions to all rolling percentage df values
	of our ticker in order to make labels under the column: target.
	'''

	security_of_int_df=process_data(ticker, correlated_security)

	security_of_int_df['{}_target'.format(ticker)]=\
		list(map(buy_sell_hold, security_of_int_df['{}'.format(ticker)]))
	
	security_of_int_df.drop(ticker, axis=1, inplace=True)

	return security_of_int_df


def extract_featuresets(ticker, correlated_security):

	'''
	The features and labels are separated here to be processed by the algorithm
	'''

	security_of_int_df=make_labels(ticker, correlated_security)

	security_of_int_df=security_of_int_df.replace([np.inf, -np.inf], np.nan)

	vals=security_of_int_df['{}_target'.format(ticker)].values.tolist()
	str_vals=[str(i) for i in vals]

	counter=Counter(str_vals)

	# print('Data spread:', counter)
	# print(security_of_int_df.shape)

	security_of_int_df_vals=security_of_int_df.replace([np.inf, -np.inf], np.nan)

	security_of_int_df_vals.fillna(0, inplace=True)

	return security_of_int_df, counter


def do_grid_search(ticker, correlated_security):

	'''
	GridSearch tests neighbors' Kneighbors algorithm on the divided securitites's
	data with different parameter values to pick the best possible combination
	of parameters to perform machine learning on the entire dataset.
	'''

	security_of_int_df, counter=extract_featuresets(ticker, correlated_security)
	security_of_int_df.dropna(inplace=True)
	security_of_int_df_copy=security_of_int_df.copy(deep=True)
	security_of_int_df_copy.reset_index(inplace=True)
	security_of_int_df_copy.drop(labels='Date', axis=1, inplace=True)
	
	labels=security_of_int_df_copy["{}_target".format(ticker)]	
	features=security_of_int_df_copy.reset_index()\
							.drop("{}_target".format(ticker), axis=1).values

	sss=StratifiedShuffleSplit(n_splits=3, test_size=0.33)

	nnbs=[]
	lsbs=[]
	absc=[]

	best_params={'algorithm': 'ball_tree',
	'leaf_size': 10,
	'n_neighbors': 6}
    

	return best_params, security_of_int_df, counter


def doMLalgoAnalysis(ticker, correlated_security):

	'''
	Perform machine Learning algorithm Analysis
	'''

	precisions=[]
	recalls=[]

	best_params, security_of_int_df, counter= do_grid_search(ticker, correlated_security)
	security_of_int_df.drop(labels='Date', axis=1, inplace=True)
	latest_date=security_of_int_df.iloc[[-1]]
	X=security_of_int_df.values
	y=security_of_int_df['{}_target'.format(ticker)].values

	X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,
																test_size=0.25)

	clf =\
    neighbors.KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                                    algorithm=best_params['algorithm'],
                                    leaf_size=best_params['leaf_size'],
                                    n_jobs=-1)

	clf.fit(X_train, y_train)

	confidence=clf.score(X_test, y_test)

	with open('Classifiers/{}.pkl'.format(ticker),'wb') as f:
		pickle.dump(clf,f)


	print('accuracy:',confidence)

	predictions=clf.predict(X_test)
	predicted_actions=Counter(predictions)

	precisions.append(precision_score(y_test.tolist(), predictions.tolist()))
	recalls.append(recall_score(y_test.tolist(), predictions.tolist()))

	precision = sum(precisions)/len(precisions)
	recall = sum(recalls)/len(recalls)
	latest_prediction=clf.predict(latest_date.reset_index().drop("{}_target".format(ticker), axis=1).values)

	print('precision ',precision)
	print('recall ',recall)
	print('latest prediction:',latest_prediction)
	print()

	row =[ticker,
	counter['1'],
	counter['0'],
	best_params['algorithm'],
	best_params['leaf_size'],
	best_params['n_neighbors'],
	round(confidence,2),
	round(precision,2),
	round(recall,2),
	latest_prediction[0]]

	analysis_results.append(row)
	print()


def do_ml(ticker, correlated_security):

	'''
	Perform machine Learning
	'''

	security_of_int_df, counter=extract_featuresets(ticker, correlated_security)
	security_of_int_df.dropna(inplace=True)
	security_of_int_df_copy=security_of_int_df.copy(deep=True)
	security_of_int_df_copy.reset_index(inplace=True)
	security_of_int_df_copy.drop(labels='Date', axis=1, inplace=True)
	security_of_int_df.drop(labels='Date', axis=1, inplace=True)

	latest_date=security_of_int_df.iloc[[-1]]
	X=security_of_int_df.values
	y=security_of_int_df['{}_target'.format(ticker)].values

	X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,
																test_size=0.25)

	clf =\
    neighbors.KNeighborsClassifier(n_neighbors=6,
                                    algorithm='ball_tree',
                                    leaf_size=10,
                                    n_jobs=-1)

	clf.fit(X, y)

	confidence=clf.score(X_test, y_test)

	with open('Classifiers/{}.pkl'.format(ticker),'wb') as f:
		pickle.dump(clf,f)

	latest_prediction=clf.predict(latest_date.reset_index().drop("{}_target".format(ticker), axis=1).values)

	return [ticker,correlated_security,confidence,latest_prediction[0]]


def try_all(sorted_correlations,performMLalgoAnalysis):
	
	'''
	Try all tickers
	'''

	results = []
	analysis_results = []

	key_number = 0
	len_sorted_correlations = len(sorted_correlations)

	if performMLalgoAnalysis:
		for key in sorted_correlations:
			try:
				print(key)
				print()
				doMLalgoAnalysis(key,sorted_correlations[key])

			except Exception as e:
				print(e)
				pass
		analysis_results=pd.DataFrame(analysis_results, columns=columns)
		analysis_results.to_csv('Data_Acquisition/ML_Analysis_analysis_results.csv')

	for key in sorted_correlations:
		key_number += 1
		try:
			print('\n'*100+str(round(key_number/len_sorted_correlations,2)*100)+'%')
			print()
			results.append(do_ml(key,sorted_correlations[key]))

		except Exception as e:
			print(e)
			pass

	results=pd.DataFrame(results, columns = ['ticker','correlated_security','confidence','latest_prediction'])
	results.to_csv('Data_Acquisition/Results.csv')


def find_and_sort_corr_securities():

	'''load dictionary of securities with their respective, highest correlated
	other securities.'''

	securities={}
	joined_close_corr=pd.read_csv('Data_Acquisition/joined_close_corr.csv', index_col=0)

	for security in joined_close_corr.index:
		securities[security]=joined_close_corr[security].drop(security).idxmax()

	return securities


def string_boolean_convert(userInput):

	if userInput == "T":
		return True

	elif userInput =="F":
		return False


if __name__ == '__main__':

	"""
	Prompt user for inputon wether security acronyms and data should be 
	reaqcuired, and also any new or current data should be cleaned, reformatted
	and corelated. These inputs are for user interface only and are set inplace
	to save time.
	"""

	analysis_results = []
	results = []

	acquire_acronyms = input(

		"""
		Acquire Acronyms of Securities again?[T/F]\n
		\tIf True a crawling bot will once again crawl wikipedia for security
		\tacronyms of mora than 5000 cmpanies with a few crypto currency
		\tin acronyms as well. This is not necessary if names were already 
		\tacquired within the past week.\n
		"""
		)

	acquire_data = input(
		"""
		Acquire data for all securities?[T/F]
		\tif True a series of requests for csv data (5 years worth) will be made
		\t, indexed and saved onto files. This is not necessary if models have 
		\tbeen trained within the past week.\n
		"""
		)

	s = os.path.getmtime("Data_Acquisition/joined_close_scaled.csv")
	s = dt.datetime.fromtimestamp(s).strftime('%c')
	

	cleanJoinAndCorr = input(
		"""
		Clean Join And Correlated security data?[T/F]\n
		\tIf True all security matrices will be cleaned, scaled, joined with one
		\tanother and then correlated (Computationally expensive). Otherwise the
		\tscript will follow on to used previously outputed correlation tables
		\tto train the models.\n
		\tLast revised on:\n
		\t%s\n
		""" % s)

	performMLalgoAnalysis = input(
		"""
		Perform Best Algo and param Tests?[T/F]\n
		\tIf True Grid Search will be performed on every data set to find
		\tbest suiting algorithm and it's parameters. This may be redundant work
		\tseeing as most securities are best predicted using kneighbors
		\tand certain parameters found in previous studies.\n
		""")
	
	acquire_acronyms = string_boolean_convert(acquire_acronyms)
	acquire_data = string_boolean_convert(acquire_data)
	cleanJoinAndCorr = string_boolean_convert(cleanJoinAndCorr)
	performMLalgoAnalysis =  string_boolean_convert(performMLalgoAnalysis)

	get_values_data(acquire_acronyms=acquire_acronyms, acquire_data=acquire_data)

	with open(os.path.join(os.getcwd(),'Data_Acquisition/pickles/companylist.pkl'),'rb') as f:
		securitylist=pickle.load(f)

	with open(os.path.join(os.getcwd(),'Data_Acquisition/pickles/cryptolist.pkl'),'rb') as f:
		securitylist+=pickle.load(f)

	if cleanJoinAndCorr:
	
		pool_join_dirty_csvs(os.listdir('Data_Acquisition/securities_dfs'))
		join_clean_csvs(os.listdir('Data_Acquisition/Join_Close_dfs'))

	else: pass
	
	sorted_correlations = find_and_sort_corr_securities()

	try_all(sorted_correlations,performMLalgoAnalysis)