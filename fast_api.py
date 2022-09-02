import string
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from model import ai_model
from ast import literal_eval

#uvicorn fast_api:app --reload
app = FastAPI()

@app.get("/get-comments/{id}")
def get_comments(id: str):
  
  os.system(f"youtube-comment-downloader --youtubeid {id} --output {id}.json --limit 100")
  file_path = id+'.json'
  df = pd.read_json(file_path, lines=True)
  nlp = ai_model()
 
  df['class_label'] = df.apply (lambda row: nlp.use_model(row['text'] )[0], axis=1)
  data =  df.to_dict('records') 
  

  return data

@app.get("/")
def home():
  return "hello world"




