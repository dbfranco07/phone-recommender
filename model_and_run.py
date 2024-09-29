import os

os.environ['OMP_NUM_THREADS'] = '5'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


df = pd.read_csv('phone_specs_refined_for_ml.csv')
ref_date = datetime(2014, 1, 1)
df['launch_date'] = (pd.to_datetime(df['launch_date']) - ref_date).dt.days

# columns
# Index(['name', 'stars_ave', 'total_votes', 'likes', 'screen_diag_in',
#        'screen_display', 'screen_reso_1', 'screen_reso_2', 'screen_density',
#        'os', 'chipset', 'cpu_score', 'gpu_score', 'ram', 'rear_camera_count',
#        'rear_camera_main_mp', 'front_camera_count', 'front_camera_main_mp',
#        'storage', 'cellular', 'batt_mah', 'batt_fast_charging', 'weight',
#        'launch_date', 'price'],
#       dtype='object')

col_num = []
col_obj = []

for col in df.columns:
    if df[col].dtype.__str__() == 'object':
        col_obj.append(col)
    else:
        col_num.append(col)

#19 total col_num

mas_scaler = MaxAbsScaler()
ohe = OneHotEncoder()

for_ohe = col_obj[1:-1]

df_ohe = ohe.fit_transform(
    df[for_ohe]
)
df_obj = pd.DataFrame(
    df_ohe.toarray(), 
    columns=ohe.get_feature_names_out()
)

df[col_num] = mas_scaler.fit_transform(
    df[col_num]
)

df_pp = pd.concat([df[col_num], df_obj], axis=1)

tsvd = TruncatedSVD(n_components=2)
components = tsvd.fit_transform(df_pp)
plt.scatter(components[:, 0], components[:, 1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')

kmeans = KMeans(
    n_clusters=20,
    n_init='auto'
)

transformed = kmeans.fit(components)

transformed.labels_

plt.scatter(
    components[:, 0], 
    components[:, 1], 
    c=transformed.labels_
)
cluster_labels = [f'cluster {i}' for i in np.unique(transformed.labels_)]
plt.xlabel('Component 1')
plt.ylabel('Component 2')

df_raw = pd.read_csv('phone_specs_raw.csv')
df_ref = pd.read_csv('phone_specs_refined.csv')
df_working = pd.DataFrame()
df_working[['name', 'price', 'launch_date']] = df_ref[['Name', 'Price', 'Launch Date']]
df_working['link'] = df_raw['Link']
df_working['class'] = transformed.labels_

def get_related_phones (df, phone_name, min_price=0, max_price=10000, show_all=False):

    df_working = df

    try:
        min_price = float(min_price)
        max_price = float(max_price)
    except:
        print(f'{min_price} and/or {max_price} are incorrect data types.')

    try:
        phone_class = df_working[df_working['name']==phone_name]['class'].values[0]
        df_class = df_working[df_working['class']==phone_class]
        
        if show_all:
            return df_class
        
        else:
            return df_class[
                (df_class['price'] >= min_price) & (df_class['price'] <= max_price)
            ]

    except:
        print(f'Phone name {phone_name} does not exist. Choose another')

        return

