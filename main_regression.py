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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from plot_functions import result_lr, result_mlp, result_deep, real_map, est_mlp_map 


def result(model, trX, teX, trY, teY):
    
    r2tr = model.score(trX, trY)
    r2te = model.score(teX, teY)
    
    print(f'Train R2: {round(r2tr, 6)}')
    print(f'Test  R2: {round(r2te, 6)}')
    
    trpred = model.predict(trX)
    tepred = model.predict(teX)
      
    rmse_train = ((np.sqrt(mean_squared_error(trY, trpred))))*100    
    rmse_test = ((np.sqrt(mean_squared_error(teY, tepred))))*100    

    print(f'Train RMSE: {round(rmse_train, 6)}')
    print(f'Test  RMSE: {round(rmse_test, 6)}')

#    a = min([np.min(trpred), np.min(tepred), 0])
#    b = max([np.max(trpred), np.max(tepred), 1])
    
    a = 0
    b = 0.3

    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.scatter(trY, trpred, s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x , ' f'R2 = {round(r2tr, 2)}')
    plt.title(f'Train [RMSE = {round(rmse_train, 3)}%]')
    plt.xlabel('Targte Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(teY, tepred, s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x , ' f'R2 = {round(r2te, 2)}')
    plt.title(f'Test [RMSE = {round(rmse_test, 3)}%]')
    plt.xlabel('Targte Values')
#    plt.ylabel('Predicted Values')
    plt.legend()
#    plt.show()
    plt.savefig('Regression.jpg',dpi=300)
    
    
    
    
    




df = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')


### Finding Outliers ###
#First Method:
# z_scores = stt.zscore(df)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# df = df[filtered_entries]



Y = df['SM'].to_numpy().reshape((-1, 1))
df.drop(['SM'], inplace=True, axis=1)
# df.drop(['Irrigation'], inplace=True, axis=1)
X = df.to_numpy()
# X = df['TVDI 3'].to_numpy().reshape((-1, 1))
 

scalerX = pp.MinMaxScaler(feature_range=(0,1))
X = scalerX.fit_transform(X)
pca = dec.PCA(n_components=0.95)
X = pca.fit_transform(X)



trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.8, random_state=0)
## Fix random_state = 0



mlp = nn.MLPRegressor(hidden_layer_sizes=(300,300), activation='relu', 
                      batch_size=4, tol=1e-9, max_iter=500, solver='adam', random_state=0)
# Fix random_state = 0 or 6.
# Some good neurons: (200,50), (300,300), (50,30), (10)
mlp.fit(trX, trY)
result(mlp, trX, teX, trY, teY)


# lr = li.LinearRegression()
# lr.fit(trX, trY)
# result(lr, trX, teX, trY, teY)

""" MLP Features """
# mlp.n_iter_

# #epochs1 = range(1,mlp.n_iter_+1)
# #a = mlp.loss_curve_
# #plt.figure()
# #plt.plot(epochs1,mlp.loss_curve_)


# weights = mlp.coefs_
# biases = mlp.intercepts_




# """ MLP Features """ 

trpred = mlp.predict(trX)
tepred = mlp.predict(teX)

# trpred = lr.predict(trX)
# tepred = lr.predict(teX)


# n1 = 300
# n2 = 300
# deep = Sequential()
# deep.add(Dense(n1, input_dim=4, activation= "relu")) ## First Hidden Layer has n1 neurons 
# deep.add(Dense(n2, activation= "relu")) ## Second Hidden Layer has n2 neurons 
# deep.add(Dense(1, activation='sigmoid')) ## neurons in last layer is the same as output Dim
# deep.summary() #Print model Summary
# deep.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
# deep.fit(trX, trY, epochs=50)
# result_deep(deep, trX, teX, trY, teY)


# ## Evaluate the Accuracy of Deep Neural Model
# [loss, accuracy_training] = deep.evaluate(trX, trY)
# print('Accuracy for training data: %.2f' % (accuracy_training*100))

# _, accuracy_testing = deep.evaluate(teX, teY)
# print('Accuracy for testing data: %.2f' % (accuracy_testing*100))


# plt.rcParams["figure.figsize"] = (10,6)
# plt.figure()
plt.figure()
plt.plot(tepred, lw=1.5, label='Prediction' )
plt.plot(teY, lw=1.5, label='Real')
plt.title('Prediction on Testing Data')
plt.xlabel('Samples')
plt.ylabel('Soil Moisture')
plt.legend()
plt.savefig('Testing.jpg',dpi=300)


plt.figure()
plt.plot(trpred, lw=1.2, label='Prediction' )
plt.plot(trY, lw=1.2, label='Real')
plt.title('Prediction on Training Data')
plt.xlabel('Samples')
plt.ylabel('Soil Moisture')
plt.legend()
plt.savefig('Training.jpg',dpi=300)













