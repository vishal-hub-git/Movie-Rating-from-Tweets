from datetime import datetime
from twitter import Twitter, OAuth
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

print("--------------------Welcome to VR Reviews------------------------")

def get_movie_details():
    # Getting movie details from user
    movie_name=input("Enter the movie name: ")      
    release_date=input("Enter Release date: ")      
    return movie_name,release_date

consumer_key='Your consumer key'
consumer_secret= 'Your consumer secret'
access_token='Your access token'
access_token_secret= 'Your secret access token'

oauth = OAuth(access_token,access_token_secret,consumer_key,consumer_secret)    # Authenticating this python application with Twitter using API access keys and tokens
twitter = Twitter(auth=oauth)       # Creating a twitter API class by using the authentication done

movie_name,release_date=get_movie_details()
     
print("Fetching Tweets.......")
# Fetching the tweets using twitter API class and specifying the movie_name as keyword,filtering the tweets that contains links and specifying the release date as start date and current date as end date
try:
    tweets = twitter.search.tweets(q=movie_name+' -filter:links',lang='en', count=100, since=release_date, until=datetime.today().strftime('%Y-%m-%d'))     
except:
    print("ERROR")

tweets_q = tweets['statuses']   # Retrieving the tweet info and texts  of all the tweets and storing it in tweets_q
print(len(tweets_q))

same_users=[]
tweet_data=[]
dummy=[]
for tweet in tweets_q:
    # Selecting only the tweets that is tweeted by unique users who has followers greater than 10 and if the tweet is not a retweeted tweet.
    if (tweet['user']['screen_name'] not in same_users and tweet['user']['followers_count'] > 10 and not tweet['text'].startswith("RT")):      
        # Adding the tweets to tweet_data list and adding the user's screen name to same_users list so as to avoid repeatition.
        tweet_data.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))
        same_users.append(tweet['user']['screen_name'])
        dummy.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))    

# Joining the movie_name into one word if it had total of more than one word.
txt=""
if " " in movie_name:
    l=movie_name.split(" ")
    txt="".join(l)

# Fetching tweets by specifying the joined movie_name as keyword.
if txt!="":
    try:
        tweets = twitter.search.tweets(q=txt+' -filter:links',lang='en', count=100, since=release_date, until=datetime.today().strftime('%Y-%m-%d'))       
    except:
        print("ERROR",tweets)
    tweets_q = tweets['statuses']
    dummy=[]
    for tweet in tweets_q:
        if (tweet['user']['screen_name'] not in same_users and tweet['user']['followers_count']>10 and not tweet['text'].startswith("RT")):
            tweet_data.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))
            same_users.append(tweet['user']['screen_name'])
            dummy.append(str(' '.join(tweet['text'].split("\n")).encode(encoding='utf-8')))

print("Tweets Fetched!!")

def stopwords_list():
    # Reading the stopwords from the stopwords.txt file and adding it into a list which is returned
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

def preprocess_data(dataSet,movie_name,txt,type):
    processed_data=[]
    stopWords=stopwords_list()
    for tweet in dataSet:
        temp_tweet = tweet
        if type=="test":
            tweet = tweet.replace(temp_tweet, tweet[1:])    # Removing the first letter b from tweets that are fetched
        tweet = re.sub('@[^\s]+','',tweet).lower()    #Removing the user mentions in the tweet
        tweet = re.sub('[\s]+',' ', tweet)       # Removing whitespaces from tweet
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)   # Removing hashtags and keeping only the word followed by the hashtag
        tweet = re.sub('[0-9]+', "",tweet)       # Removing numbers from tweet
        # Removing stop words from tweet
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)
        tweet = re.sub('[^a-zA-z ]',"",tweet)        # Removing all special characters except alphabets
        tweet = re.sub('[\s]+',' ', tweet)     # Removing whitespaces from tweet
        #Removing the movie name from the tweets
        l=[]
        l=movie_name.split(" ")
        for i in l:
            tweet = tweet.replace(i.lower(),'').lower()
            tweet.replace(temp_tweet, tweet)
        tweet = tweet.replace(movie_name,'').lower()
        if txt!="":
            tweet = tweet.replace(txt.lower(),'').lower()
        tweet.replace(temp_tweet, tweet)
        processed_data.append(tweet)
    return processed_data

def FeaturizeTrainingData(dataset, type_class):
    # Featurizing the training data by creating a dataframe with tweets and the corresponding labels in it(i.e., 1 for positive,-1 for negative,0 for neutral)
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
    # Featurizing the test data by creating a dataframe with only tweets in it 
    tweets=[]
    for tweet in dataset:
        tweets.append(tweet)
    feature_vector=pd.DataFrame({"reviews": tweets})
    return feature_vector

def paclassifier(train_X, train_Y, test_X):
    print("Training model.....")
    pac=PassiveAggressiveClassifier(random_state=0)     # Training the Passive Aggressive Classifier Model
    pac.fit(train_X,train_Y)       # Fitting the training data into the model
    y_pred=pac.predict(test_X)      # Predicting the label output for the test data
    return y_pred

        
pos_data=open("positive_dataset.txt").readlines()
pos_data=preprocess_data(pos_data,movie_name,txt,"train")       # Preprocessing positive data
neg_data=open("negative_dataset.txt").readlines()
neg_data=preprocess_data(neg_data,movie_name,txt,"train")       # Preprocessing negative data
pos_sentiment=[]
neg_sentiment=[]
neutral_data=pd.read_csv('processedNeutral.csv')
l=[]
neutral_data=neutral_data.columns
neutral_data=preprocess_data(neutral_data,movie_name,txt,"train")       # Preprocessing neutral data
neutral_sentiment=[]
pos_sentiment=FeaturizeTrainingData(pos_data,"positive")        # Featurizing positive data
neg_sentiment=FeaturizeTrainingData(neg_data,"negative")        # Featurizing negative data
neutral_sentiment=FeaturizeTrainingData(neutral_data,"neutral") # Featurizing neutral data
final_data=neg_sentiment.append(pos_sentiment)
final_data=final_data.append(neutral_sentiment)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.5)
tfidf_train=tfidf_vectorizer.fit_transform(final_data["reviews"]).toarray()     # Converting training data into data-term matrix by learning the idf and vocabulary
test_data=preprocess_data(tweet_data,movie_name,txt,"test")        # Preprocessing test data
test_data=FeatureizeTestData(test_data)     # Featurizing test data
tfidf_test=tfidf_vectorizer.transform(test_data["reviews"]).toarray()       # Converting test data into data-term matrix


y_pred=paclassifier(tfidf_train,final_data["labels"],tfidf_test)        # Predicting the output by using model

# Getting the positive and negative count
pos_count=list(y_pred).count(1)
neg_count=list(y_pred).count(-1)

# Finding the postive and negative percentage
neg_count-=neg_count/5
neg_count=int(neg_count)
pos_percent=(pos_count)*100/(pos_count+neg_count)
neg_percent=(neg_count)*100/(pos_count+neg_count)

print("Rating for the movie '{}' is : {} %".format(movie_name,str(round(pos_percent))))
