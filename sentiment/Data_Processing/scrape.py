from googlesearch import search
import requests as req
import re
from lxml import html
from bs4 import BeautifulSoup

import pytrends
from pytrends.request import TrendReq


def google_trends(query):

    # Set up the trend fetching object
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [search]
    
    # Create the search object
    pytrends.build_payload(kw_list, cat=0, geo='', gprop='news')
    
    # Get the interest over time
    interest = pytrends.interest_over_time()
    print(interest.head())

    # Get related searches
    related_queries = pytrends.related_queries()
    print(related_queries)

#       # Get Google Top Charts
	# top_charts_df = pytrend.top_charts(cid='actors', date=201611)
	# print(top_charts_df.head())
    
    return interest, related_queries


def fetch(query):
	results = []
	for result in search(query, tld="co.in", num=10, stop=1, pause=2): 
		results.append(result)

	resp = req.get(results[0])
	content = resp.text 
	stripped = re.sub('<[^<]+?>', '', content)

	file = open('test.txt','w') 
	file.write(stripped) 
	file.close() 
	return results

# Make date part of the query string
def fetch2(query):
	results = []
	file = open('test.txt','w') 
	for result in search(query, tld="co.in", num=1, stop=1, pause=2): 
		results.append(result)
		resp = req.get(result)
		content = BeautifulSoup(resp.content, 'html.parser')
		for i, p in enumerate(content.select('p')):
			file.write(p.text) 

	file.close() 
	return results


def verify(url):
	# Determines if source is credible and recent. Discards those that do not meet criteria
	return

# Get date from query string, look at how stock changed around date of publication
def label(query):
	# Determines if source is credible and recent. Discards those that do not meet criteria
	return

# Loop that iterates through past dates. For each date, fetches news article and labels it
def main():
	return

if __name__ == '__main__':
	query = "NIO Stock Rating"
	fetch2(query)



# from googlesearch.googlesearch import GoogleSearch
# response = GoogleSearch().search("NIO Stock Rating")
# for result in response.results:
#     print("Title: " + result.title)
#     print("Content: " + result.getText())