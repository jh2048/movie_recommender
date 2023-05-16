import pandas as pd

from sentiment import SentimentAnalyser

df = pd.read_csv('test_set.csv')


SA = SentimentAnalyser()
print('trying sentiment analyser... ')
df = df.applymap(str)
print(df.head())

df['sentiment'], df['sent_score']  = df['user_review'].apply(lambda x: SA.sentiment_analyser(x)).str
print(df['sentiment'])

df.to_csv('data/test_set_withsent.csv', index=False)

