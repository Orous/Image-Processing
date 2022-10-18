import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error




def result_lr(model, trX, teX, trY, teY):
    
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
    b = 1

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
    plt.savefig('Linear Reg.jpg',dpi=300)








def result_mlp(model, trX, teX, trY, teY):
    
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
    b = 1

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
    plt.savefig('MLP Reg.jpg',dpi=300)








def result_deep(model, trX, teX, trY, teY):
    
 
    trpred = model.predict(trX)
    tepred = model.predict(teX)
      
    rmse_train = ((np.sqrt(mean_squared_error(trY, trpred))))*100    
    rmse_test = ((np.sqrt(mean_squared_error(teY, tepred))))*100    

    print(f'Train RMSE: {round(rmse_train, 6)}')
    print(f'Test  RMSE: {round(rmse_test, 6)}')

#    a = min([np.min(trpred), np.min(tepred), 0])
#    b = max([np.max(trpred), np.max(tepred), 1])
    
    a = 0
    b = 1

    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.scatter(trY, trpred, s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x')
    plt.title(f'Train [RMSE = {round(rmse_train, 3)}%]')
    plt.xlabel('Targte Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(teY, tepred, s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x')
    plt.title(f'Test [RMSE = {round(rmse_test, 3)}%]')
    plt.xlabel('Targte Values')
#    plt.ylabel('Predicted Values')
    plt.legend()
#    plt.show()
    plt.savefig('Deep Reg.jpg',dpi=300)










def real_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('Real Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('Real Temprature Map.jpg',dpi=300)
    

def est_mlp_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('MLP Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('MLP Estimated Temprature Map.jpg',dpi=300)

    

def est_lr_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('LR Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('LR Estimated Temprature Map.jpg',dpi=300)



def est_deep_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('Deep Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('Deep Estimated Temprature Map.jpg',dpi=300)



def error_mlp_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('MLP Error Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('MLP Error Temprature Map.jpg',dpi=300)


def error_lr_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('LR Error Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('LR Error Temprature Map.jpg',dpi=300)
    

def error_deep_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('Deep Error Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('Deep Error Temprature Map.jpg',dpi=300)





def est_new_mlp_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('New MLP Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('New MLP Estimated Temprature Map.jpg',dpi=300)

    

def est_new_lr_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('New LR Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('New LR Estimated Temprature Map.jpg',dpi=300)



def est_new_deep_map(x):
    plt.figure()
    plt.imshow(x)
    plt.title('New Deep Estimated Temprature Map (°C)')
    plt.colorbar()
    plt.savefig('New Deep Estimated Temprature Map.jpg',dpi=300)















