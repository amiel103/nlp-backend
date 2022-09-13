
import os
from fastapi import FastAPI
import json
import pickle
from pydantic import BaseModel
from typing import List
#uvicorn fast_api:app --reload

class comment(BaseModel):
  cid: str
  text: str
  time: str
  author: str
  channel: str
  votes: int
  photo: str
  heart: bool
  time_parsed: float
  class_label: str
  signature:str

class receive(BaseModel):
  data:List
        

def use_model(text):
  dirfile = os.path.dirname(os.path.realpath(__file__))
  dirtfidf = os.path.join(dirfile,'tfidf_features.pkl')
  tfidf = pickle.load(open(dirtfidf, 'rb'))
  text = [text,]
  features = tfidf.transform(text)
  dirclf = os.path.join(dirfile,'clfmodel.pkl')
  loaded_model = pickle.load(open(dirclf, 'rb'))
  y_pred = loaded_model.predict(features)

  return y_pred


#uvicorn fast_api:app --reload
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:3000",
#     "https://youtubecommentlabeler.vercel.app/",
#     "http://youtubecommentlabeler.vercel.app/",
# ]

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get-comments/{id}")
def get_comments(id: str):
  
  os.system(f"youtube-comment-downloader --youtubeid {id} --output {id}.json --limit 100")
  file_path = id+'.json'


  with open(file_path, 'r', encoding="utf8") as f:

    data = f.read()
    data = data.replace('\n','\n,') 
    data = "[" + data[0:-1] + "]"
    print( )
    final_data = json.loads( data )

    last = []

    for x in range( 0 , len(final_data) ):
      #print(use_model( final_data[x]['text'] ) )
      dict0 = dict(final_data[x])
      dict0['class_label'] =   use_model( final_data[x]['text'] )[0]
      
      final_data[x] = dict0

  print(2)
  # df = pd.read_json(file_path, lines=True)
  # df['class_label'] = df.apply (lambda row: use_model(row['text'] )[0], axis=1)
  # data =  df.to_dict('records') 
  
  os.remove(file_path)

  return final_data

@app.get("/")
def home():
  return "hello world"

@app.post("/add-correction/")
async def create_item(item: receive):

  import json
  with open('data.json', 'a', encoding='utf-8') as f:
    for x in item.data:
      json.dump(dict(x), f, ensure_ascii=False, indent=4)
  return item


