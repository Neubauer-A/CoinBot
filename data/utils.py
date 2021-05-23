import os

def unique_tweets(dir):
    tweet_set = set()
    for i in os.listdir(dir):
        df = pd.read_csv(dir+'/'+i)
        tweet_set |= set(df['tweet'].values)
    return tweet_set

def write_raw_set(filename, max_lines):
    count = 0
    f = open(f'{filename}_{count}.py', 'w')
    f.write('d = {\n')
    for i in tweet_set:
      count += 1
      f.write('"'+i.strip()+'": 0,'+'\n')
      if count % max_lines == 0:
        f.write('}')
        f.close()
        f = open(f'{filename}_{count}.py', 'w')
        f.write('d = {\n')
    f.write('}')
    f.close()