
import os
from fastapi import FastAPI
import json
import pickle

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
      dict0['class'] =   use_model( final_data[x]['text'] )[0]
      
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




