import numpy as np
import scipy.stats as stt
import sklearn.linear_model as li
import sklearn.neural_network as nn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.decomposition as dec
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import cv2
import scipy.io as sio
from sklearn.metrics import mean_squared_error




df = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')


description = df.describe()
print(description)



""" Correlation """

correlation = df.corr()
plt.figure()
sb.heatmap(correlation, vmin=-1, vmax=+1)
plt.savefig('corr1.jpg',dpi=300)







""" After Normalization """

Y = df['SM'].to_numpy().reshape((-1, 1))
df.drop(['SM'], inplace=True, axis=1)
X = df.to_numpy()


scalerX = pp.MinMaxScaler()
X2 = scalerX.fit_transform(X)


All_Data = np.concatenate((X2,Y),axis=1)

C = np.zeros((np.shape(All_Data)[1], np.shape(All_Data)[1]))
for i in range(np.shape(All_Data)[1]):
    for j in range(np.shape(All_Data)[1]):
        C[i, j] = stt.pearsonr(All_Data[:, i], All_Data[:, j])[0]
plt.figure()
sb.heatmap(C, vmin=-1, vmax=+1)
plt.savefig('corr2.jpg',dpi=300)



""" After PCA """

pca = dec.PCA(n_components=0.95)
X3 = pca.fit_transform(X2)


All_Data2 = np.concatenate((X3,Y),axis=1)

C2 = np.zeros((np.shape(All_Data2)[1], np.shape(All_Data2)[1]))
for i in range(np.shape(All_Data2)[1]):
    for j in range(np.shape(All_Data2)[1]):
        C2[i, j] = stt.pearsonr(All_Data2[:, i], All_Data2[:, j])[0]
plt.figure()
sb.heatmap(C2, vmin=-1, vmax=+1)
plt.savefig('corr3.jpg',dpi=300)




















