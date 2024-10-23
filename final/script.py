import pandas as pd
import pickle as pkl


if __name__ == '__main__':  
    # читаем датасет с открытыми данными и идентификаторы объектов
    data = pd.read_csv('data.csv')
    ids = pd.read_parquet('test.parquet')['id']
    
    # расконсервируем модель
    with open('model.pkl', 'rb') as model_f:
        model = pkl.load(model_f)
    
    # вычисляем вероятности и создаем submission
    score = model.predict_proba(data)[:, 1]
    
    submission = pd.DataFrame({
        'id': ids, 
        'score': score
    })
    
    submission.to_csv('submission.csv', index=False)
    
    