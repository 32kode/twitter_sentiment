from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
import string
import warnings

from wordcloud import WordCloud
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display


# Authentication
access_token = "***"
access_token_secret = "***"
consumer_key = "***"
consumer_key_secret = "***"
bearer_token = "***"
client_id = "***"
client_secret = "***"

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

warnings.filterwarnings("ignore", category=DeprecationWarning) # depricationwarning on piechart


def percentage(part, whole):
    return 100 * float(part)/float(whole)


keyword = input("please enter keyword or hashtag to search: ")
noOfTweet = int(input("please enter how many tweets to analyze: "))

tweets = tweepy.Cursor(api.search_tweets, lang='en', q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in tweets:
    tweet_list.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.text)
        negative += 1

    elif pos > neg:
        positive_list.append(tweet.text)
        positive += 1

    elif pos == neg:
        neutral_list.append(tweet.text)
        neutral += 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
# print('total number: ', len(tweet_list))
# print('posive number: ', len(positive_list))
# print('negative number: ', len(negative_list))
# print('neutral number: ', len(neutral_list))

# create pie chart
labels = ['Positive [' + str(positive)+ '%',
          'Neutral [' + str(neutral) + '%',
          'Negative [' + str(negative) + '%']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment analysis for: "+keyword)
plt.axis('equal')
# plt.show()

tweet_list.drop_duplicates(inplace=True)

# clean text

tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]


# remove punct and other
remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^A-Za-z0-9 \t])|(\w+:\/\/\S+)", " ", x)
tw_list['text'] = tw_list.text.map(remove_rt).map(rt)
tw_list['text'] = tw_list.text.str.lower()
# print(tw_list.head(10))

tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = 'negative'
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = 'positive'
    else:
        tw_list.loc[index, 'sentiment'] = 'neutral'
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp

# print(tw_list.head(10))

tw_list_negative = tw_list[tw_list['sentiment'] == 'negative']
tw_list_positive = tw_list[tw_list['sentiment'] == 'positive']
tw_list_neutral = tw_list[tw_list['sentiment'] == 'neutral']

# count values in single columns


def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True)*100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


print(count_values_in_column(tw_list, 'sentiment'))


def create_wordcloud(text):
    mask = np.array(Image.open("cloud.png"))
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('spotify')
    wc = WordCloud(background_color='white',
                   mask=mask,
                   max_words=3000,
                   stopwords=stopwords,
                   repeat=True,)
    wc.generate_from_text(str(text))
    wc.to_file("wc_clean_base.png")
    print('wordcloud saved successfully')
    path = 'wc_clean_base.png'
    display(Image.open(path))


def remove_punctuation(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punctuation(x))


# apply tokenization and lower because this is the first time we access individual words
def tokenization(text):
    text = re.split('\W+', text)
    return text


tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))

# remove stopwords
stopword = nltk.corpus.stopwords.words('english')
stopword.append(['spotify', 'th'])


def remove_stopword(text):
    text = [word for word in text if word not in stopword]
    return text


tw_list['nostop'] = tw_list['tokenized'].apply(lambda x: remove_stopword(x))
ps = nltk.PorterStemmer()  # apply porter


def stemming(text):
    text = [ps.stem(word) for word in text]
    return text


tw_list['stemmed'] = tw_list['nostop'].apply(lambda x: stemming(x))


def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])
    text_rc = re.sub('[0-9]+', '', text_lc)
    text = re.sub('(\\b[A_Za-z] \\b|\\b [A_Za-z]\\b)', ' ', text)
    tokens = re.split('\W+', text_rc)  # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # removes stopwords and stemming
    return text

# something wrong with this


countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0, ascending=False).head(20)
print(countdf[1:11])


create_wordcloud(tw_list['text'])

