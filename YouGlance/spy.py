# All Imports
import pandas as pd
import numpy as np
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system('cmd /k "python -m spacy download en"')
    nlp = spacy.load("en_core_web_sm")
import nltk

nltk.download("vader_lexicon")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.decomposition import NMF
from collections import Counter


class spy:
    def __init__(self, video_id):
        self.video_id = video_id
        self.unwanted = [
            "DATE",
            "TIME",
            "MONEY",
            "QUANTITY",
            "ORDINAL",
            "CARDINAL",
            "LANGUAGE",
            "LOC",
        ]
        self.k = YouTubeTranscriptApi.get_transcript(self.video_id)
        self.topic = 0
        self.unique_ents = []

    def get_unwanted(self):
        return self.unwanted

    def tweak_unwanted(self, l):
        self.unwanted = l

    # Creating DataFrame

    def generate_df(self):
        print("Creating DataFrame.....")
        l = []
        t = []
        for x in self.k:
            l.append(x["text"])
            t.append(x["start"])
        d = {"Text": l, "Start": t}

        self.df = pd.DataFrame(d)

        print("Cleaning Text....")
        # Stripping Punctuation in the Text and creating a new Column named cleaned_text
        self.df["cleaned_text"] = self.df["Text"].apply(self.clean_text)

        print("Identifying Entities....")
        # Adding entities of texts to DataFrame
        self.df["label"] = self.df["cleaned_text"].apply(self.show_ents)

        # Creating a List of Unique Entities

        w = " ".join(list(self.df["cleaned_text"]))
        doc = nlp(w)
        for x in doc.ents:
            if x.label_ not in self.unwanted and x.text not in self.unique_ents:
                self.unique_ents.append(x.text)

        return self.df

    # Function to add entities to dataframe
    def show_ents(self, x):
        unq = []
        doc = nlp(x)
        for x in doc.ents:
            if x.label_ not in self.unwanted and x.text not in unq:
                unq.append(x.text)
        return unq

    # Function to clean Text
    def clean_text(self, x):

        k = re.findall(r"[^\-\n]+", x)
        k = " ".join(k)
        return k

    # Return Final DataFrame
    def return_df(self):
        return self.df

    # Return List of Unique Entities
    def get_unique_ents(self):
        return self.unique_ents

    # Return Number of Times an Entity Label Occurs
    def show_label_stats(self):
        self.topic = 0
        unique = []
        for x in list(self.df["cleaned_text"]):
            y = nlp(x)
            for w in y.ents:
                if w.label_ not in self.unwanted:
                    unique.append(w.label_)

        self.counter = Counter(unique)
        for x in self.counter.values():
            if x >= 7:
                self.topic = self.topic + 1

        return self.counter

    # Filter Dataframe by entities ,Parameter p is a List
    def search_by_ents(self, p):
        allot = []
        result = []

        for i in range(0, len(self.df)):
            for x in p:
                if x in self.df.iloc[i]["label"] and i not in allot:
                    allot.append(i)

        for x in allot:
            d = {}
            d["text"] = self.df.iloc[x]["cleaned_text"]
            d["start"] = self.df.iloc[x]["Start"]
            d["ent"] = self.df.iloc[x]["label"]
            result.append(d)

        return result

    # Wildcard Search
    # X is the string to search By
    def wildcard_search(self, x):
        vectorizer = TfidfVectorizer()
        result = []
        corpus = list(self.df["cleaned_text"])
        corpus.append(x)
        X = vectorizer.fit_transform(corpus)
        vals = cosine_similarity(X[-1], X)
        vals = vals.flatten()
        idx = vals.argsort()

        m = idx[-5:-1]

        for y in m:
            d = {}
            d["text"] = self.df.iloc[y]["cleaned_text"]
            d["start"] = self.df.iloc[y]["Start"]
            d["ent"] = self.df.iloc[y]["label"]
            result.append(d)
        corpus = []
        return result

    # Auto Segregarte Topics      Topic Modeling
    def segregate_topic(self, thresh=None):
        if thresh is None:
            d = {}
            tfidf = TfidfVectorizer(max_df=0.96, min_df=2, stop_words="english")
            x = tfidf.fit_transform(self.df["cleaned_text"])
            nmf_model = NMF(n_components=self.topic, random_state=21)
            nmf_model.fit(x)
            for index, topic in enumerate(nmf_model.components_):

                d[index] = [tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]]

            result = nmf_model.transform(x)

            y = result.argmax(axis=1)
            self.df["topic_label"] = y

            return (self.df, d)
        else:
            d = {}
            tfidf = TfidfVectorizer(max_df=0.96, min_df=2, stop_words="english")
            x = tfidf.fit_transform(self.df["cleaned_text"])
            nmf_model = NMF(n_components=thresh, random_state=21)
            nmf_model.fit(x)
            for index, topic in enumerate(nmf_model.components_):

                d[index] = [tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]]

            result = nmf_model.transform(x)

            y = result.argmax(axis=1)
            self.df["topic_label"] = y
            return (self.df, d)

    # Vader Sentiment Analysis   #Thresh will be a tuple for specifying Range of Score for neutral
    def sentiment_analysis(self, thresh=None):
        sid = SentimentIntensityAnalyzer()
        if thresh is None:
            l = []
            r = []

            for x in range(0, len(self.df)):
                d = sid.polarity_scores(self.df.iloc[x]["cleaned_text"])
                l.append(d)
                if d["compound"] <= 0.4 and d["compound"] > -0.3:

                    r.append("Neutral")
                elif d["compound"] > 0.4:
                    r.append("Positive")

                else:
                    r.append("Negative")

            self.df["sentiment_dict"] = pd.Series(l)
            self.df["sentiment_label"] = pd.Series(r)

        else:
            l = []
            r = []

            for x in range(0, len(self.df)):
                d = sid.polarity_scores(self.df.iloc[x]["cleaned_text"])
                l.append(d)
                if d["compound"] <= max(thresh) and d["compound"] > min(thresh):

                    r.append("Neutral")
                elif d["compound"] > max(thresh):
                    r.append("Positive")

                else:
                    r.append("Negative")

            self.df["sentiment_dict"] = pd.Series(l)
            self.df["sentiment_label"] = pd.Series(r)

        neutral = len(self.df[self.df["sentiment_label"] == "Neutral"])
        positive = len(self.df[self.df["sentiment_label"] == "Positive"])
        negative = len(self.df[self.df["sentiment_label"] == "Negative"])

        d = {
            "DataFrame": self.df,
            "Neutral": neutral,
            "Positive": positive,
            "Negative": negative,
        }
        return d
