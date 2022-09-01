import string, csv, os, sys, nltk
import pandas as pd
import numpy as np
import re, pickle, contractions, nltk
import preprocessor as p
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
from collections import Counter
import ast

#nltk.download('wordnet')
#nltk.download('stopwords')


class ai_model:
    def __init__(self):
        pass   

    def lemmatize_text(self,text):  
        lemmatizer = nltk.stem.WordNetLemmatizer()
        w_tokenizer = TweetTokenizer()
        return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]

    def clean_tweets(self,comments):
        comments['comment_clean'] = np.nan
        #add hashtags into a column

        for i,v in enumerate(comments['text']):
            #basic cleaning (urls, mentions, emojis, smileys, hashtags, reserved words(rt,fav
            clean = p.clean(v)

            #remove links
            clean = re.sub(r'http\S+', '', clean)

            #remove digits
            clean = re.sub('[0-9]+', '', clean)

            #make string lowercase
            clean = clean.lower()

            #expand contractions
            clean = contractions.fix(clean)

            #remove punctuations
            clean = re.sub(r'[^\w\s]', '', clean)

            #remove underscores
            clean = clean.replace("_"," ")

            #remove words: hagupit, ruby, rubyph
            remove_words = ['cong', 'congtv', 'team payaman']
            clean_list= clean.split()
            clean_list = [word for word in clean_list if word.lower() not in remove_words]
            clean = ' '.join(clean_list)

            #lemmatize and get ngrams
            clean = self.lemmatize_text(clean)

            #set tweet_clean column data type as object
            comments['comment_clean'] = comments['comment_clean'].astype('object')

            #insert preprocessed texts into column
            comments.at[i, 'comment_clean'] = clean
        
        #remove stopwords
        stop_words = set(stopwords.words('english'))
        comments['comment_clean'] = comments['comment_clean'].apply(lambda x: [item for item in x if item not in stop_words])
        
        #combine unigrams into string
        for i,v in enumerate(comments['comment_clean']):
            unigram_string = ' '.join(v)
            comments.at[i, 'comment_clean'] = unigram_string

        return comments

    def evaluate_model(self,comments):
        # X = feature set, Y = target
        data_x = comments['comment_clean']
        data_y = comments['label']

        #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # original
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        score_array=[]
        ascore_array=[]
        for train_index, test_index in skf.split(data_x, data_y):
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            vect = TfidfVectorizer(use_idf=True, min_df=5, max_df=0.85, ngram_range=(1,2))


            x_train = vect.fit_transform(x_train)


            x_test = vect.transform(x_test)

            clf = SVC(kernel='rbf')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            ascore_array.append(accuracy_score(y_test, y_pred))
            score_array.append(precision_recall_fscore_support(y_test,y_pred, average=None))
        

        
        avg_score = np.mean(score_array, axis=0)
        print("Average scores: \n", avg_score)
        avg_score = np.array(avg_score).transpose().tolist()

        avg_ascore = np.mean(ascore_array, axis=0)
        print("Accuracy score: ", avg_ascore)

        data = {
            "avg_ascore"     : avg_score,
            "accuracy_score" : avg_ascore,
        }

        return data

    def train_model(self):
       
        file = "datasets\\all.csv"
        comments = pd.read_csv(file, encoding='latin1')
        #tweets.tweet_text = tweets.tweet_text.astype(str)

        comments = self.clean_tweets(comments)
        print("tweets cleaned")
        data = self.evaluate_model(comments)
        print("models evaluated")
        
        x_train = comments['comment_clean']
        y_train = comments['label']
        vect = TfidfVectorizer(use_idf=True, min_df=5, max_df=0.85, ngram_range=(1,2))
        vect.fit(x_train)
        dirfile = os.path.dirname(os.path.realpath(__file__))
        #save tfidf features
        with open(os.path.join(dirfile,'tfidf_features.pkl'),'wb') as f:
            pickle.dump(vect, f)
        x_train = vect.fit_transform(x_train)
        clf = SVC(kernel='rbf')
        clf.fit(x_train, y_train)

        #export classification model
        with open(os.path.join(dirfile,'clfmodel.pkl'),'wb') as f:
            pickle.dump(clf, f)

        print("models saved")

        

        return data

    def use_model(self,text):
        dirfile = os.path.dirname(os.path.realpath(__file__))
        dirtfidf = os.path.join(dirfile,'tfidf_features.pkl')
        tfidf = pickle.load(open(dirtfidf, 'rb'))
        text = [text,]
        features = tfidf.transform(text)

        dirclf = os.path.join(dirfile,'clfmodel.pkl')
        loaded_model = pickle.load(open(dirclf, 'rb'))
        y_pred = loaded_model.predict(features)

        #1 Announcements (CA)
        #2 Casualty and Damage (CD)
        #3 Call for Help (CH)
        #4 Others (O)

        return y_pred
   
    





if __name__ == '__main__':
    
    aimodel = ai_model()
    #aimodel.train_model()

    x = aimodel.use_model("ahahahahahahahahaha")
    print(x)

