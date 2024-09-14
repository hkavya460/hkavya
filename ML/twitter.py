import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from  nltk.corpus import stopwords
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.stem.porter import  PorterStemmer
from nltk.stem import  WordNetLemmatizer
import  seaborn as sns

#preprocessing the tweeter sentiment data

twitter_data = pd.read_csv("Tweets.csv")
print(twitter_data.shape)
print(twitter_data.columns)
print(twitter_data.head())

e =  twitter_data.isnull().sum()
# print(e)
print(twitter_data['airline_sentiment'].value_counts())  #3 classes of target
# number of tweets
tweet_id = twitter_data.tweet_id.count()
# print(tweet_id)

port_stem = PorterStemmer()
# def stemming(content):
#     stemming_content = re.sub('[^a-zA-Z]',' ',content)
#     stemming_content = stemming_content.lower()
#     stemming_content = stemming_content.split()
#     stemming_content = [port_stem.stem(word) for word in stemming_content if  not  word  in stopwords.words('english')]
#
#     stemming_content = ' '.join(stemming_content)
#     return stemming_content

# twitter_data['stemming_content'] = twitter_data['text'].apply(stemming)
print(twitter_data.head())
stop_words = set(stopwords.words('english'))
wordnet_lemmitizer = WordNetLemmatizer()
def normalize(tweet):
    normalize_text =  re.sub('[^a-zA-Z]',' ',tweet)
    normalize_text = nltk.word_tokenize(normalize_text)[2:]
    normalize_text = [l.lower() for l in normalize_text ]
    normalize_text = list(filter (lambda l:l  not in stop_words,normalize_text))
    lemmas = [wordnet_lemmitizer.lemmatize(t) for t  in normalize_text ]
    return lemmas
twitter_data['normalize_tweet'] = twitter_data['text'].apply(normalize)
print(twitter_data.head())

# ngrams
from nltk import  ngrams
def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list,input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list,input_list[1:],input_list[2:]))]
    return bigrams+trigrams
twitter_data['grams'] = twitter_data['normalize_tweet'].apply(ngrams)
print(twitter_data['grams'].head())

#count collections means common words in the text for different targets
import collections
def count_words(input):
    cunt = collections.Counter()
    for row in input :
        for word in row:
            cunt[word]  +=1
    return cunt

twitter_data[(twitter_data['airline_sentiment']=='negative')][['grams']].apply(count_words)


#####

#vectorization - converting the text into matrix form
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import hstack
vectorizer =  CountVectorizer(ngram_range=(1,2))
vectorized_data = vectorizer.fit_transform(twitter_data['text'])



####
vector_scale = TfidfVectorizer()
vector_x =vector_scale.fit_transform(twitter_data['text'])

#target is replaced Negative -0 ,neutral -1, postive-2
twitter_data.replace({'airline_sentiment':{'negative':0 ,'neutral':1,'positive':2}},inplace=True)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

train_data ,test_data,y_train,y_test = train_test_split(indexed_data,twitter_data['airline_sentiment'],test_size=0.20,shuffle=True,random_state=45)
train_data_index = train_data[:,0]
train_data = train_data[:,1:]
test_data_index = test_data[:,0]
test_data = test_data[:,1:]

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import OneHotEncoder ,OrdinalEncoder ,StandardScaler
model = OneVsRestClassifier(LogisticRegression()
                            )
model_output = model.fit(train_data,y_train)
y_p = model.score(test_data,y_test)
print(y_p)


