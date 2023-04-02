import pandas as pd
import numpy as np
import re
df=pd.read_csv(r"D:\Downloads\movie_metadata.csv")
df.drop(columns=['movie_imdb_link','color','aspect_ratio'],inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='movie_title')

tfidf_mat = tfidf.fit_transform(df['director_name'])
tfidf_mat
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_mat,tfidf_mat)
cosine_sim
data = pd.Series(df['director_name'],index = df.index)
data = pd.DataFrame(data)

class ItemRecommender:
    def __init__(self):
        self.data = data
        self.cosine_sim = cosine_sim
        
    def recommendation(self, keyword):
        index = self.data[self.data['director_name'].str.contains(keyword, flags=re.IGNORECASE, regex=True)].index[0]
        sim_score = list(enumerate(self.cosine_sim[index]))    
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

        sim_score = sim_score[1:8]
        final_index = [i[0] for i in sim_score]
        return final_index
    
    def predict(self,ram):
        idx = self.recommendation(ram)
        b=pd.DataFrame()
        b['title_year']=df['title_year'].iloc[idx]
        b['imdb_score']=df['Score'].astype('int64').iloc[idx]
        b.reset_index(drop=True,inplace=True)
        return b
    
rec=ItemRecommender()

import pickle
pickle.dump(rec,open('model.pkl','wb'))