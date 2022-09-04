from transformers import pipeline
import pandas as pd

class SentimentAnalyser:

    def __init__(self) -> None:
        self.sentiment_pipe = pipeline('sentiment-analysis')
    

    def reduce_size(self, sentence):
        if len(sentence) > 512:
            return sentence[:512]
        else:
            return sentence


    def sentiment_analyser(self, sentence):
        sentiment = self.sentiment_pipe(self.reduce_size(sentence))
        label = sentiment[0]['label']
        score = sentiment[0]['score']
        return label, score
