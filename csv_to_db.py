from sqlalchemy import create_engine
import pandas as pd

u = 'postgres'
pw = 'b4sT4rd4lyf!'
host = 'localhost'
port = '5432'
db = 'trim2'


db_url = (
    f'postgresql://{u}:{pw}@{host}:{port}/{db}'
)

engine = create_engine(db_url)

data = pd.read_csv('phone_specs_raw.csv')

data.to_sql(
    'phone_spec', 
    engine, 
    schema='landing', 
    if_exists='replace', 
    index=False
)

pd.read_csv('gpu_score.csv').to_sql(
    'gpu_score', 
    engine, 
    schema='landing', 
    if_exists='replace', 
    index=False
)