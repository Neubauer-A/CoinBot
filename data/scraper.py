import re
import twint
import nest_asyncio
import pandas as pd
from CoinBot.data.utils import is_ascii

class TwitterScraper:

    def __init__(self):
        # Avoid runtime compatibility issue
        nest_asyncio.apply()
        # Avoid unnecessary warnings
        pd.options.mode.chained_assignment = None

    def get_tweets(self, user=None, term=None, limit=None, since=None, popular=False):
        # Configure twint search
        c = twint.Config()
        c.Lang = 'en'
        c.Popular_tweets = popular
        c.Hide_output = True
        c.Pandas = True
        c.Filter_retweets = True
        if user:
            c.Username = user
        if term:
            c.Search = term
        if limit:
            c.Limit = limit
        if since:
            c.Since = since
        twint.run.Search(c)
        tweets = twint.storage.panda.Tweets_df
        # Remove hyperlinks
        tweets['tweet'] = tweets['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        # Remove tweets that are too short
        tweets['length'] = tweets.tweet.str.len()
        tweets = tweets[tweets.length > 4]
        # Remove tweets that aren't ascii (e.g. tweets in Chinese or Arabic)
        tweets['ascii'] = tweets['tweet'].apply(lambda x: is_ascii(x))
        return tweets['tweet'][tweets.ascii]