import pandas as pd
import pickle as pkl

test = pd.read_parquet('data/test.parquet')

with open('', 'rb') as f:
    model = pkl.load(f)
    

