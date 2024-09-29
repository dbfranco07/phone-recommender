import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('phone_specs_processed.csv')

y = df['price']
df_old = df.copy()
col_to_drop = [
    'name',
    'brand',
    'price'
]

df.drop(columns=col_to_drop, inplace=True)
all_col = df.columns
cat_col = [
    'screen_display_tech',
    'os',
    'cellular',
]
num_col = [col for col in all_col if col not in cat_col]

ohe = OneHotEncoder(sparse_output=False)
ohe.set_output(transform='pandas')
ohe.fit_transform(df[cat_col])

standardscaler = StandardScaler()
standardscaler.set_output(transform='pandas')
standardscaler.fit_transform(df[num_col])


df_ml = pd.merge(
    left=standardscaler.fit_transform(df[num_col]), 
    right=ohe.fit_transform(df[cat_col]),
    how='left',
    left_index=True,
    right_index=True
)

X = df_ml

model  = LinearRegression()

model.fit(X[df.release_date_year <= 2022], y[df.release_date_year <= 2022])
y_pred = model.predict(X)
model.score(X, y)
model.coef_

col_weights = pd.DataFrame({'col_name': X.columns, 'weights': model.coef_})
col_weights.sort_values('weights')
model.intercept_
df_old['score'] = y / y_pred
view = df_old[
    ['name', 'brand', 'release_date_month', 'release_date_year', 'price', 'pred_price', 'score']
][df.release_date_year>=2024]

model.predict(X[X.index==454])
X[X.index==454]
df_old.name.iloc[454]

model2 = RidgeCV(alphas=np.arange(0.1, 10, 0.1), store_cv_results=True)
model2.fit(X, y)
model2.predict(X[X.index==454])

y_pred2 = model2.predict(X)
df_old['score2'] = y/y_pred2

view = df_old[['name', 'brand', 'release_date_year', 'price', 'score', 'score2']]

model2.score(X, y)


model2.alpha_

sns.histplot(df['stars'], kde=True)
help(sns.histplot)

test = np.log(df['stars'] + 1)
sns.histplot(test, kde=True)
sns.histplot(df[['stars']], kde=True)

powertransformer = PowerTransformer()
powertransformer.set_output(transform='pandas')
test = powertransformer.fit_transform(df[['stars']])
test2 = standardscaler.fit_transform(df[['stars']])

sns.scatterplot(x=X['stars'], y=y)
sns.scatterplot(x=test['stars'], y=y)

df_old[['stars', 'price']].corr()

(y-y.mean())/y.std()
X.columns
sns.scatterplot(x=powertransformer.fit_transform(df[['like_share']])['like_share'], y=((y-y.mean())/y.std()))

sns.histplot(powertransformer.fit_transform(df[['like_share']])['like_share'], kde=True)

df_old['like_share'].min()
sns.scatterplot(x=powertransformer.fit_transform(df[['like_share']])['like_share'], y=y)
sns.histplot(powertransformer.fit_transform(y.to_frame()))

sns.scatterplot(
    x=powertransformer.fit_transform(df[['screen_diag_in']])['screen_diag_in'],
    y=powertransformer.fit_transform(y.to_frame())['price']
)


sns.scatterplot(x=df['refreshrate'], y=y)



ols_model = LinearRegression()
ols_model.fit(X, y)
ols_model.coef_

U, S, Vt = np.linalg.svd(X)
# U, S, Vt = np.linalg.svd(np.c_[X, np.ones((X.shape[0], 1))])
sigma = np.zeros((U.shape[0], Vt.shape[1]))
sigma[:min(U.shape[0], Vt.shape[1]), :min(U.shape[0], Vt.shape[1])] = np.diag(S)
# np.linalg.pinv(sigma)
Vt.T @ np.linalg.pinv(sigma) @ U.T @ y
np.linalg.pinv(X) @ y - ols_model.coef_

# np.allclose(np.linalg.pinv(X), Vt.T @ np.linalg.pinv(sigma) @ U.T)
Vt.T @ np.linalg.pinv(sigma) @ U.T @ y

ols_model.intercept_
X.columns
ols_model.score(X, y)


mlp_regressor = MLPRegressor(
    verbose=True,
    learning_rate_init=1,
    warm_start=True,
    learning_rate='adaptive',
    max_iter=50000,
    tol=1e-10,
    n_iter_no_change=40
)

mlp_regressor.fit(X[df.release_date_year <= 2022], y[df.release_date_year <= 2022])

mlp_regressor.loss_curve_

y.shape
samples = list(np.random.choice(np.arange(0, y.shape[0]), int(0.75*y.shape[0]), replace=False))
non_samples = []
for i in np.arange(0, y.shape[0]):
    if i not in samples:
        non_samples.append(i)

len(non_samples) + len(samples)
sorted(samples)


X.to_numpy()[samples]

mlp_regressor.fit(X.to_numpy()[samples], y[samples])
mlp_regressor.fit(X[df.release_date_year <= 2023], y[df.release_date_year <= 2023])
mlp_regressor.score(X[df.release_date_year == 2024], y[df.release_date_year == 2024])
mlp_regressor.score(X.to_numpy()[samples], y[samples])
mlp_regressor.score(X.to_numpy()[non_samples], y[non_samples])






y_pred = mlp_regressor.predict(X)
df_old['score'] = y_pred/y
df_old['pred_price'] = y_pred

view = df_old[['name', 'brand', 'release_date_year', 'price', 'pred_price','score']]

mlp_regressor.score(X, y)
df_old['gpu_score'].describe()
new_data_tested = {
    'stars': 4.5,
    'stars_count': 7,
    'like_share': 5,
    'screen_diag_in': 6.68,
    'screen_reso_width': 720,
    'screen_reso_length': 1600,
    'screen_ppi': 264,
    'refreshrate': 90,
    'rear_camera_count': 2,
    'rear_camera_main_mp': 50,
    'front_camera_count': 1,
    'front_camera_main_mp': 8,
    'cpu_score': 14.8,
    'gpu_score': 30,
    'ram': 8,
    'storage': 128,
    'batt_mah': 6000,
    'batt_fast_charging': 1,
    'release_date_month': 6,
    'release_date_year': 2024,
    'screen_display_tech_lcd': 1,
    'screen_display_tech_ltps': 0,
    'screen_display_tech_oled': 0,
    'os_android': 1,
    'os_emui': 0,
    'os_harmonyos': 0,
    'os_ios': 0,
    'cellular_3g': 0,
    'cellular_4g': 1,
    'cellular_5g': 0
}

new_data = {
    'stars': 4.5,
    'stars_count': 7,
    'like_share': 5,
    'screen_diag_in': 6.68,
    'screen_reso_width': 720,
    'screen_reso_length': 1600,
    'screen_ppi': 264,
    'refreshrate': 90,
    'rear_camera_count': 2,
    'rear_camera_main_mp': 50,
    'front_camera_count': 1,
    'front_camera_main_mp': 8,
    'cpu_score': 14.8,
    'gpu_score': 30,
    'ram': 8,
    'storage': 128,
    'batt_mah': 6000,
    'batt_fast_charging': 1,
    'release_date_month': 6,
    'release_date_year': 2024,
    'screen_display_tech_lcd': 1,
    'screen_display_tech_ltps': 0,
    'screen_display_tech_oled': 0,
    'os_android': 1,
    'os_emui': 0,
    'os_harmonyos': 0,
    'os_ios': 0,
    'cellular_3g': 0,
    'cellular_4g': 1,
    'cellular_5g': 0
}

X.columns.to_list()

X_new = pd.DataFrame({col:[new_data[col]] for col in new_data})

for col in new_data:
    print(col, new_data[col])


right_col = []
left_col = []

for col in X.columns:
    for col2 in cat_col:
        if col2 in col:
            right_col.append(col)

for col in X.columns:
    if col not in right_col:
        left_col.append(col)


X_new_ml = pd.merge(
    left=standardscaler.transform(X_new[left_col]), 
    right=X_new[right_col],
    how='left',
    left_index=True,
    right_index=True
)

mlp_regressor.predict(X_new_ml)
mlp_regressor.predict(X[X.index==1])
df_old
#14906459.93126511

from scrape_phone_specs import get_full_specs_for_phone
from clean_raw_to_processed import preprocess

new_data = get_full_specs_for_phone('https://www.pinoytechnoguide.com/smartphones/vivo-y28')
new_data['name'] = 'vivo Y28'
new_data['link'] = 'https://www.pinoytechnoguide.com/smartphones/vivo-y28'
new_data = pd.DataFrame({col:[new_data[col]] for col in new_data})
preprocess(new_data_df)

new_data.to_csv('test.csv', index=False)
new_data_df = pd.read_csv('test.csv')
new_data_df.info()

new_data_df_ml = pd.merge(
    left=standardscaler.transform(new_data_df[num_col]), 
    right=ohe.transform(new_data_df[cat_col]),
    how='left',
    left_index=True,
    right_index=True
)


mlp_regressor.predict(new_data_df_ml)

for idx, i in enumerate((new_data_df_ml - X_new_ml).to_numpy()[0]):
    print(i, new_data_df_ml.columns[idx], X_new_ml.columns[idx])

new_data_df_ml.columns[19]
new_data_df['cpu_score']

new_data_df.release_date_year
X_new.release_date_year


X_train = X[10:]
y_train = y[10:]

mlp_regressor.fit(X_train, y_train)
mlp_regressor.score(X_train, y_train)

k=5
print(mlp_regressor.predict(X[k:k+1]), y[k])

mlp_regressor.predict(new_data_df_ml)
mlp_regressor.predict(X_new_ml)

X['stars'].where(X['stars']<=0).value_counts()