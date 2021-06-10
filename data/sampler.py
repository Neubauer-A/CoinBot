import numpy as np
import tensorflow_text
from time import sleep
from datetime import datetime
from multiprocessing import Pool
from tensorflow.keras.models import load_model
from CoinBot.data.scraper import TwitterScraper
from CoinBot.client.bot2kraken import KrakenClient

SENTIMENT_MODEL = load_model('CoinBot/sentiment/btcBERT', compile=False)

class DataSampler:
    
    def __init__(self, sleep_interval=6.5):
        self.sleep_interval = sleep_interval
        self.client = KrakenClient()
        self.scraper = TwitterScraper()

    def get_sentiment(self):
        # scrape tweets
        tweets = self.scraper.get_tweets(term='btc', limit=100)
        # get model predictions
        sentiment = np.argmax(SENTIMENT_MODEL.predict(tweets), axis=1)
        # get the average prediction and scale it
        sentiment = np.array([np.mean(sentiment) / 2]).astype('float32')
        return sentiment

    def sample_worker(self, query):
        return self.client.public_query(query)

    def get_data(self, num_samples):
        # map queries to processes
        pool = Pool(processes=4)
        queries = ['ohlc','order_book','trades','spread']
        results = []
        for i in range(num_samples):
            p = pool.map_async(self.sample_worker, queries)
            results.append(p.get())
        pool.close()
        pool.join()
        # check if the data has been successfully retrieved
        try:
            if results[0][0]['result']['XXBTZUSD'] and results[0][1]['result']['XXBTZUSD'] and \
                results[0][2]['result']['XXBTZUSD'] and results[0][3]['result']['XXBTZUSD']:
                return results
        # cool down if number of requests exceed Kraken's limits
        except:
            sleep(10)
            return self.get_data(num_samples)

    def data_arrays(self, data, transform=True):
        # get separate arrays for each result
        ohlc_arr = np.array([i[0]['result']['XXBTZUSD'] for i in data]).astype('float32')
        ob_ask_arr = np.array([i[1]['result']['XXBTZUSD']['asks'] for i in data]).astype('float32')
        ob_bid_arr = np.array([i[1]['result']['XXBTZUSD']['bids'] for i in data]).astype('float32')
        trades_arr = np.array([i[2]['result']['XXBTZUSD'] for i in data])
        # encode 'buy', 'sell', 'limit', and 'market'
        trades_arr[:, :, 3][np.where(trades_arr[:, :, 3] == 'b')] = 1
        trades_arr[:, :, 3][np.where(trades_arr[:, :, 3] == 's')] = 0
        trades_arr[:, :, 4][np.where(trades_arr[:, :, 4] == 'l')] = 1
        trades_arr[:, :, 4][np.where(trades_arr[:, :, 4] == 'm')] = 0
        spread_arr = np.array([i[3]['result']['XXBTZUSD'] for i in data]).astype('float32')
        # list of 100 values from each array
        arrays = [ohlc_arr[:,-100:,:][0], ob_ask_arr[0], ob_bid_arr[0], \
                  trades_arr[:,-100:,:-1].astype('float32')[0], spread_arr[:,-100:,:][0]]
        # add feature dimension and return
        return [np.reshape(i, (i.shape[0], i.shape[1], 1)) for i in arrays]

    def get_sample(self):
        sleep(self.sleep_interval)
        sentiment = self.get_sentiment()
        data = self.get_data(1)
        arrays = self.data_arrays(data)
        arrays.append(sentiment)
        return arrays

    def save_samples(self, num_samples, 
                     save_path='CoinBot/data/datasets/rl_samples', 
                     verbose=0):
        # get all the samples we want (unless peer resets connection)             
        samples = []
        for i in range(num_samples):
            try:
                sample = self.get_sample()
                samples.append(sample)
                if verbose:
                    print(f'{i}/{num_samples}')
            except:
                break
        # sort the arrays
        arrays = []
        for i in range(len(samples[0])):
            arr = np.array([j[i] for j in samples])
            arrays.append(arr)
        # save the arrays
        name = datetime.now().strftime("%m%d%y_%H%M")
        np.savez_compressed(name+'.npz', \
            arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5])