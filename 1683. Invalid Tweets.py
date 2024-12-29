import pandas as pd


def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    # SELECT tweet_id FROM tweets WHERE LENGTH(content) > 15 ORDER BY tweet_id;
    return tweets[tweets["content"].apply(lambda data: len(data) > 15)]["tweet_id"].to_frame()


if __name__ == '__main__':
    tweets = pd.read_csv('pandas dataset/1683. Invalid Tweets.csv')
    print(invalid_tweets(tweets))

# Done ✅
