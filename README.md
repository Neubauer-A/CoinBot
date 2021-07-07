# CoinBot
A program for deep learning models to day trade cryptocurrencies.

The ultimate purpose of this is to easily deploy a reinforcement agent for bitcoin day trading. 

So far, this includes some pieces that may be of interest on a standalone basis as well:
- A data set of tweets about bitcoin that have been hand-labeled for sentiment.
- A script to finetune a BERT model for sentiment analysis of bitcoin tweets.
- A Twitter scraper that uses twint to get recent tweets about bitcoin.
- An interface for bots to interact with Kraken's REST API.
- A working TF-Agents environment for reinforcement learning that ties in all of the above.
- A growing data set for the TF-Agents environment that can be used to bootstrap new reinforcement agents.

My final goals for this project are to make a pretrained Q-learning agent publicly available and add detailed documentation on the use of each of the above. In the future, especially if there's interest from others, I'll add options for different exchanges and trading strategies.
