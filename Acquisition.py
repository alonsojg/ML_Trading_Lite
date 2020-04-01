from collections import Counter
from collections import defaultdict
from multiprocessing import Process, Pool, cpu_count
from pathlib import Path

import bs4 as bs 
import csv
import datetime as dt
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
import time

end_date=int(time.mktime(dt.datetime.now().timetuple()))
start_date=end_date-78840000

################################################################################

### Obtain cryptocurrency names from wikipedia and clean_up Data

def clean_crypto_names(crypto):

	pattern=re.compile('\w+')
	match=pattern.search(crypto)

	if match:
		return match.group()
	else:
		pass

def save_crypto_names():

	dirty_names=[]

	if not os.path.exists('Data_Acquisition/pickles'):
		os.makedirs('Data_Acquisition/pickles')
	
	resp=requests.get('https://en.wikipedia.org/wiki/List_of_cryptocurrencies')
	soup=bs.BeautifulSoup(resp.text,'lxml')
	table=soup.find('table', {'class': 'wikitable sortable'})

	for row in table.findAll('tr')[1:]:
		ticker=row.findAll('td')[3].text
		dirty_names.append(ticker)

	cryptolist=[clean_crypto_names(crypto) for crypto in dirty_names]
	cryptolist=list(filter(None, cryptolist))
	cryptolist=[crypto+'-crypto' for crypto in cryptolist]
	# cryptolist.insert(0,cryptolist[-1])
	# cryptolist=cryptolist[:-1]

	with open("Data_Acquisition/pickles/cryptolist.pkl","wb") as f:
		pickle.dump(cryptolist,f)

	return(cryptolist)


################################################################################

### Obtain company acronyms from yahoo

def request_tickers(tickers, csv_url):

	with requests.Session() as s:

		download=s.get(csv_url)
		decoded_content=download.content.decode('utf-8')
		cr=csv.reader(decoded_content.splitlines(), delimiter=',')

		for row in cr.__iter__():
			if row[0] != "Symbol":
				tickers.append(row[0])


def save_tickers():

	tickers=[]

	if not os.path.exists('Data_Acquisition/pickles'):
		os.makedirs('Data_Acquisition/pickles')
	
	request_tickers(tickers, "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download")
	request_tickers(tickers, "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download")
	request_tickers(tickers, "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download")
	
	with open('Data_Acquisition/pickles/companylist.pkl','wb') as f:
		pickle.dump(tickers, f)

	return tickers

################################################################################

''' Obtain data pertaining to each class of security (crypto or stock) from yahoo
via HTML request '''

def get_cookie_value(r):

    return {'B': r.cookies['B']}


def get_page_data(crypto):

    url="https://finance.yahoo.com/quote/%s/?p=%s" % (crypto, crypto)
    r=requests.get(url)
    cookie=get_cookie_value(r)
    lines=r.content.decode('unicode-escape').strip().replace('}', '\n')
    return cookie, lines.split('\n')


def find_crumb_store(lines):

    # Looking for
    # ,"CrumbStore":{"crumb":"9q.A4D1c.b9
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")


def split_crumb_store(v):

    return v.split(':')[2].strip('"')


def get_cookie_crumb(crypto):

    cookie, lines=get_page_data(crypto)
    crumb=split_crumb_store(find_crumb_store(lines))
    return cookie, crumb


def setup_folderpath():
	
	if 'securities_dfs' not in os.listdir('Data_Acquisition/'):
		os.makedirs('Data_Acquisition/securities_dfs')
	else: pass


def setup_filepath(security, kind):

	if kind=='crypto':		
		filename='Data_Acquisition/securities_dfs/%s-crypto.csv' % (security)
		return filename

	filename='Data_Acquisition/securities_dfs/%s.csv' % (security)
	return filename
	

def setup_query(security, kind, start_date, end_date, crumb):

	if kind=='crypto':
		security='%s-USD' % (security)

	url="https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s"\
		 % (security, start_date, end_date, crumb)

	return url


def get_data(security, kind, start_date, end_date, cookie, crumb, append=False):


	print('Acquiring data for {}'.format(security))

	filename=setup_filepath(security, kind)
	url=setup_query(security, kind, start_date, end_date, crumb)
	response=requests.get(url, cookies=cookie)

	if response.text[0] == 'D':
		
		if append:
		    with open (filename, 'ab') as handle:

		        for block in response.iter_content(1024):

		        	block=block.splitlines()[1:]
		        	block=b'\n'+b'\n'.join(block)
		        	handle.write(block)
		else:
			with open (filename, 'wb') as handle:
				for block in response.iter_content(1024):

					handle.write(block)

	            
	else:

		print('\nSymbol delisted\nData not Acquired\n')
		pass


def df_present(security, kind):


	if kind=='crypto':
		return os.path.exists('Data_Acquisition/securities_dfs/'+security+'-crypto.csv')

	return os.path.exists('Data_Acquisition/securities_dfs/'+security+'.csv')


def not_up_to_date(security, kind):

	if kind=='crypto':
		security=security+'-crypto'

	with open('Data_Acquisition/securities_dfs/'+security+'.csv', 'r') as f:

		last_date = None

		for last_date in csv.reader(f):
			pass

		try:


			last_date=int(time.mktime(dt.datetime.strptime(last_date[0], "%Y-%m-%d").timetuple()))

			if last_date<end_date-86400:
				return last_date

			return False

		except: 

			pass


def millis():

  return int(round(time.time() * 1000))


def try_to_get_data(security, kind, start_date, end_date, append=False):
	
	try:
		cookie, crumb=get_cookie_crumb(security)
		get_data(security, kind, start_date, end_date, cookie, crumb, append)
		
	except: pass


def download_quotes(security, kind):

	# if kind=='crypto':
	# 	security_list=[crypto.split('-')[0] for crypto in security_list]
	
	# for security in security_list[::-1]:

	if df_present(security,kind):
		
		last_date=not_up_to_date(security, kind)

		if last_date:

			append=True 

			try_to_get_data(security, kind, last_date+172800, end_date, append)

		else: pass

	else:

		try_to_get_data(security, kind, start_date, end_date)


def generate_data_aquisition_processes():

	security_list = []

	with open('Data_Acquisition/pickles/cryptolist.pkl','rb') as f:
		security_list=pickle.load(f)
		security_list=[tuple(i.split('-')) for i in security_list]

	with open('Data_Acquisition/pickles/companylist.pkl','rb') as f:
		companylist=pickle.load(f)
		companylist=[(i,'stock') for i in companylist]
		security_list+=companylist

	pool=Pool(processes=cpu_count())
	start_time=millis()
	results=pool.starmap(download_quotes,security_list)
	print("\nTotal took " + str(millis() - start_time) + " ms\n")



def get_values_data(acquire_acronyms=False, acquire_data=False):

	setup_folderpath()

	if acquire_acronyms:
		
		cryptolist=save_crypto_names()
		companylist=save_tickers()
		
		if acquire_data:

			generate_data_aquisition_processes()

		else:
			pass

	else:

		if acquire_data:

			generate_data_aquisition_processes()

		else:
			pass



if __name__ == '__main__':

	get_values_data(acquire_acronyms=False,acquire_data=True)
