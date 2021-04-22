# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:54:50 2021

@author: Lenovo
"""

from datetime import datetime
from twitter import Twitter, OAuth
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

print("--------------------Welcome to VR Reviews------------------------")

def get_movie_details():
    movie_name=input("Enter the movie name: ")
    release_date=input("Enter Release date: ")
    return movie_name,release_date

consumer_key='Enter your consumer key'
consumer_secret= 'Enter your consumer secret'
access_token='Enter your access token'
access_token_secret= 'Enter your access token secret'

oauth = OAuth(access_token,access_token_secret,consumer_key,consumer_secret)
twitter = Twitter(auth=oauth)

movie_name,release_date=get_movie_details()
try:
    
    tweets = twitter.search.tweets(q=movie_name+' -filter:links',lang='en', count=100, since=release_date, until=datetime.today().strftime('%Y-%m-%d'))
except:
    print("ERROR")

tweets_q = tweets['statuses']
print(len(tweets_q))

same_users=[]
tweet_data=[]
dummy=[]

for tweet in tweets_q:
    if (tweet['user']['screen_name'] not in same_users and tweet['user']['followers_count'] > 10 and not tweet['text'].startswith("RT")):
        tweet_data.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))
        same_users.append(tweet['user']['screen_name'])
        dummy.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))

txt=""
if " " in movie_name:
    l=movie_name.split(" ")
    txt="".join(l)
    
if txt!="":
    try:
        tweets = twitter.search.tweets(q=txt+' -filter:links',lang='en', count=100, since=release_date, until=datetime.today().strftime('%Y-%m-%d'))
    except:
        print("ERROR",tweets)
    tweets_q = tweets['statuses']
    print("Fetching Tweets.......")
    dummy=[]
    for tweet in tweets_q:
        if (tweet['user']['screen_name'] not in same_users and tweet['user']['followers_count']>10 and not tweet['text'].startswith("RT")):
            tweet_data.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))
            same_users.append(tweet['user']['screen_name'])
            dummy.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))

print("Tweets Fetched!!")
def stopwords_list():
    stopwords_file=open("stopwords.txt", "r")
    stopwords=[]
    try:
        line=stopwords_file.readline()
        while line:
            stopword=line.strip()
            stopwords.append(stopword)
            line=stopwords_file.readline()
        stopwords_file.close()
    except:
        print("ERROR: Opening File")
    return stopwords

def preprocess_data(dataSet,movie_name,txt):
    processed_data=[]
    stopWords=stopwords_list()
    for tweet in dataSet:
        temp_tweet = tweet
        l=[]
        l=movie_name.split(" ")
        for i in l:
            tweet = tweet.replace(i.lower(),'').lower()
            tweet.replace(temp_tweet, tweet)
        tweet = tweet.replace(movie_name,'').lower()
        tweet.replace(temp_tweet, tweet)
        tweet.replace(tweet,tweet[:-1])
        
        if txt!="":
            tweet = tweet.replace(txt.lower(),'').lower()
            tweet.replace(temp_tweet, tweet)
        tweet = re.sub('@[^\s]+','',tweet).lower()
        tweet.replace(temp_tweet, tweet)
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('[0-9]+', "",tweet)
        tweet.replace(temp_tweet,tweet)
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)
        tweet.replace(temp_tweet,tweet)
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)
        processed_data.append(tweet)
    return processed_data

def FeaturizeTrainingData(dataset, type_class):
    tweets=[]
    for tweet in dataset:
        tweets.append(tweet)
    if(type_class=="positive"):
        feature_vector=pd.DataFrame({"reviews": tweets, "labels": list(np.ones(len(tweets), dtype=int))})
    elif(type_class=="negative"):
        feature_vector=pd.DataFrame({"reviews": tweets, "labels": list(np.negative(np.ones(len(tweets), dtype=int)))})
    else:
        feature_vector=pd.DataFrame({"reviews": tweets, "labels": list(np.zeros(len(tweets), dtype=int))})
    return feature_vector

def FeatureizeTestData(dataset):
    tweets=[]
    for tweet in dataset:
        tweets.append(tweet)
    feature_vector=pd.DataFrame({"reviews": tweets})
    return feature_vector

def paclassifier(train_X, train_Y, test_X):
    print("Training model.....")
    pac=PassiveAggressiveClassifier(random_state=0)
    pac.fit(train_X, train_Y)
    y_pred=pac.predict(test_X)
    return y_pred

        
pos_data=open("positive_dataset.txt").readlines()
pos_data=preprocess_data(pos_data,movie_name,txt)
neg_data=open("negative_dataset.txt").readlines()
neg_data=preprocess_data(neg_data,movie_name,txt)
pos_sentiment=[]
neutral_data=pd.read_csv('processedNeutral.csv')
l=[]
neutral_data=neutral_data.columns
neutral_data=preprocess_data(neutral_data,movie_name,txt)
neutral_sentiment=[]
pos_sentiment=FeaturizeTrainingData(pos_data,"positive")
neg_sentiment=FeaturizeTrainingData(neg_data,"negative")
neutral_sentiment=FeaturizeTrainingData(neutral_data,"neutral")
final_data=neg_sentiment.append(pos_sentiment)
final_data=final_data.append(neutral_sentiment)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.5)
tfidf_train=tfidf_vectorizer.fit_transform(final_data["reviews"]).toarray()
test_data=preprocess_data(tweet_data,movie_name,txt)
test_data=FeatureizeTestData(test_data)
tfidf_test=tfidf_vectorizer.transform(test_data["reviews"]).toarray()


y_pred=paclassifier(tfidf_train,final_data["labels"],tfidf_test)

text = ""



pos_count=list(y_pred).count(1)
neg_count=list(y_pred).count(-1)

neg_count-=neg_count/4
neg_count=int(neg_count)
pos_percent=(pos_count)*100/(pos_count+neg_count)
neg_percent=(neg_count)*100/(pos_count+neg_count)

print("Rating for the movie '{}' is : {} %".format(movie_name,str(round(pos_percent))))


