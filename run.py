import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#Initialize the class object
kmeans = KMeans(n_clusters= 500)
pca = PCA(2)

columns = ['age','address','reason','travel_time','study_time',\
            'gpa','eca','rs','free_time','health','absence',\
            'frs','friends','alcohol','family_size']

df = (pd.read_csv('data2.csv').iloc[1:,2:])
df = pd.DataFrame(pca.fit_transform(df))

#predict the labels of clusters.
label = kmeans.fit_predict(df)
df['label'] = label

# creating bias in data
df2['score']=(np.sum(df2[['rs','free_time','health','absence','frs','friends','alcohol']],axis=1))
df_bad = (df2[df2['score'] >= 28.0])

choices = list(range(1,4))
df_bad['feedback'] = (np.random.choice(choices, size = len(df_bad)))

choices = list(range(5,11))
df_good = (df2[df2['score'] < 28.0])
df_good['feedback'] = (np.random.choice(choices, size = len(df_good)))

df = (pd.concat([df_good, df_bad],axis=0))
df = df.sample(frac=1).reset_index(drop=True)

# random shuffling of groups
df2 = (df.sample(frac=0.5))
df3 = (df[~df.index.isin(df2.index)])

# Obtaining differences in users
X = (df2.iloc[:,:-2].values-df3.iloc[:,:-2].values)
Y_ave = (df2.iloc[:,-1].values + df3.iloc[:,-1].values)/2.0

# Create train and test data
train_x = X[:-100]
train_y = Y_ave[:-100]

test_x = X[-10:]
test_y = Y_ave[-10:]
val = {}
user = []
for idx in np.arange(50,60):
    for i in np.arange(1,11):
        val[i] = (df2.iloc[-1*idx,:-2].values-df3.iloc[(-1*idx)-i,:-2].values)
    user.append(val)
    val = {}

# Create and train linear regression model
lr = LinearRegression()
lr.fit(train_x, train_y)

val = {}
val_array = []
all_val_array = []
for idx in np.arange(0,5):
    for x in np.arange(1,11):
        x_val = (np.array(user[idx][x]).reshape(1,-1))
        val_array.append(lr.predict(x_val)[0])
    all_val_array.append(val_array)
    val_array = []

# Prediction output to match users
for i in np.arange(0,len(all_val_array)):
    print (f'student{i} matches to {(np.argmax(all_val_array[i]))}')
