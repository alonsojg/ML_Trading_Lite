'''
Alonso Gutierrez

Correlational_Analysis will output a correlation matrix for all securities and
their percentage movements on a everyday basis for the past 5 years.
'''


from shutil import rmtree
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pickle


def chunk(xs, n):

    '''Split the list, xs, into n chunks'''

    L = len(xs)
    assert 0 < n <= L
    s = L//n
    return [xs[p:p+s] for p in range(0, L, s)]


def open_n_read_csvs(ticker):

	return pd.read_csv('Data_Acquisition/securities_dfs/{}'.format(ticker),sep=',')


def clean_up_dirty_csvs(ticker,df):

	'''
	Drop all other columns other than "Adj Close" for each security
	''' 
	
	df.drop(['Open','Low','High','Volume','Close'],axis=1,inplace=True)
	df.set_index('Date',inplace=True)
	df.rename(columns={'Adj Close':ticker.split('.')[0]},inplace=True)
	df=pd.DataFrame(df) 
	df=df.round(3)

	return df


def join_dirty_csvs(security_list):

	main_df=pd.DataFrame()
	count=0
	len_security_list=len(security_list)

	for ticker in security_list:
		try:
			df=open_n_read_csvs(ticker)
			df=clean_up_dirty_csvs(ticker,df)
			if main_df.empty:
				main_df=df
				count+=1
			else:
				main_df=main_df.join(df,how='outer')
				main_df=main_df.reset_index().drop_duplicates(subset='Date').set_index('Date')
				main_df.replace(["NaN", 'NaT'], np.nan, inplace = True)
				main_df.dropna(thresh=int((0.75*main_df.shape[1])),inplace=True)
				main_df.dropna(thresh=500,inplace=True,axis='columns')
				print(str(round((count/len_security_list)*100))+'% Loaded')
				count+=1
				print(main_df.shape)
				print(ticker, count)

		except Exception as e:
			raise e


	print(os.getpid())	

	main_df.to_csv('Data_Acquisition/Join_Close_dfs/joined_close_{}.csv'.format(os.getpid()))
	

def open_n_read_subjoins(ticker):

    df=pd.read_csv('Data_Acquisition/Join_Close_dfs/{}'.format(ticker),sep=',')
    df.set_index('Date',inplace=True)
    df=df.round(3)
    return df


def join_clean_csvs(list_of_resulting_dfs):

    main_df=pd.DataFrame
    count=0
    len_list_of_resulting_dfs=len(list_of_resulting_dfs)


    for ticker in list_of_resulting_dfs:
        df=open_n_read_subjoins(ticker)
        try:
            if main_df.empty:
                main_df=df
                count+=1
            else:
                main_df=main_df.join(df,how='outer')
                main_df=main_df.reset_index()\
                               .drop_duplicates(subset='Date')\
                               .set_index('Date')
                main_df.replace(["NaN", 'NaT'], np.nan, inplace = True)
                main_df.dropna(thresh=int((0.75*main_df.shape[1])),inplace=True)
                main_df.dropna(thresh=500,inplace=True,axis='columns')
                print(str(round((count/len_list_of_resulting_dfs)*100))+
                    '% Loaded')
                count+=1
                print(main_df.shape)
                print(ticker, count)

        except:
            pass

    main_df.to_csv('Data_Acquisition/joined_close.csv')
    
    # Find the cross correlation of all securities
    corr=main_df.corr()
    corr.to_csv('Data_Acquisition/joined_close_corr_dirty.csv')
    
    # Scale correlations; corre will be 1 if they are => 0.9, and will go to
    # zero otherwise
    corr[abs(corr[:])<0.9]=0
    corr=corr.round(3)
    
    # Narrow down to those securities that have a high correlation with more
    # secuirites other than themselves    
    #f = lambda i: corr[corr.columns[i]].to_numpy().nonzero()
    #securities_of_interest = [corr.columns[i] for i in range(corr.shape[0]) if len(f(i)[0])>1]
    #corr = corr.loc[securities_of_interest,securities_of_interest]
    
    # Let's save this new dataframe
    corr.to_csv('Data_Acquisition/joined_close_corr.csv')
    main_df=main_df.round(3)
    scaled=main_df.pct_change().dropna()*100
    scaled.to_csv('Data_Acquisition/joined_close_scaled.csv')


def pool_join_dirty_csvs(security_list):
	
	pool=Pool(processes=6)
	rmtree('Data_Acquisition/Join_Close_dfs')
	os.makedirs('Data_Acquisition/Join_Close_dfs')
	results=pool.map(join_dirty_csvs,chunk(security_list,6))
	pool.close()


if __name__ == '__main__':

	pool_join_dirty_csvs(os.listdir('Data_Acquisition/securities_dfs'))
	join_clean_csvs(os.listdir('Data_Acquisition/Join_Close_dfs'))