import re
import twint
import nest_asyncio
import pandas as pd

class TwitterScraper:

    def __init__(self):
        # Avoid runtime compatibility issue.
        nest_asyncio.apply()

    # For live use inputs to the sentiment model.
    # Default parameters significantly reduce spammy tweets. 
    def get_tweets(self, term='bitcoin', limit=100, popular=True):
        c = twint.Config()
        c.Search = term
        c.Lang = 'en'
        c.Limit = limit
        c.Popular_tweets = popular
        c.Hide_output = True
        c.Pandas = True
        twint.run.Search(c)
        tweets = twint.storage.panda.Tweets_df['tweet']
        tweets = tweets.apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        return tweets

    def save_csvs(self, 
                  users='default', 
                  terms='default',
                  save_dir='CoinBot/data/csvs/', 
                  since='2016-05-09'):
      
        if users == 'default':
            users = ['coindesk','Cointelegraph','BitcoinMagazine']
        if terms == 'default':
            terms = ['bitcoin','btc']

        if not type(users) is list:
            users = [users]
        if not type(terms) is list:
            terms = [terms]

        # Iterate through twitter search parameters and save each as a csv file.
        for user in users:
            for term in terms:
                filename = save_dir+'/'+user+'_'+term+'_tweets.csv'
                c = twint.Config()
                c.Username = user
                c.Search = term
                c.Since = since
                c.Hide_output = True
                c.Store_csv = True
                c.Output = filename
                twint.run.Search(c)
                # Remove unnecessary columns. 
                df = pd.read_csv(filename, index_col=3)
                df = df[['time','username','tweet','replies_count',
                        'retweets_count','likes_count','hashtags','cashtags']]
                # Remove hyperlinks from tweets.
                df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
                df.to_csv(filename)