import numpy as np
import pandas as pd

sigma = 15
xs_length = 4
data_size = 20000
pv = 0.9


#generate Xv
xv = np.zeros(data_size)
for i in range(data_size):
    if round(np.random.uniform(0, 1), 1) <= 0.5:
        xv[i] = np.random.normal(1,sigma)
    else:
        xv[i] = np.random.normal(-1,sigma)
data = pd.DataFrame(data=xv,columns=['xv'] )

#generate Y
y = np.zeros(data_size)
for i in range(data_size):
    if round(np.random.uniform(0, 1), 1) <= pv:
        y[i] = np.sign(data.values[i, 0])
    else:
        y[i] = -1*np.sign(data.values[i, 0])

data['y'] = y

#generate xs

def envs(input):
    if input <= data_size/4:
        probility = 0.5
        env = 1
    elif input <= 2 * data_size / 4 and input > data_size / 4:
        probility = 0.6
        env = 2
    elif input <= 3 * data_size / 4 and input > 2 * data_size / 4:
        probility = 0.2
        env = 3
    elif input <= 4 * data_size / 4 and input > 3 * data_size / 4:
        probility = 0.1
        env = 4
    return probility, env

def envs_train(input):
    if input <= data_size/2:
        probility = 0.999
        env = 1
    else :
        probility = 0.8
        env = 2
    return probility, env

def envs_test(input):
    if input <= data_size/2:
        probility = 0.2
        env = 1
    else :
        probility = 0.1
        env = 2
    return probility, env

xs_total = []
xs_temp = np.zeros(xs_length)
env_label = np.zeros(data_size)
for i in range(data_size):
    k, env = envs(i)
    for j in range(xs_length):
        if round(np.random.uniform(0, 1), 1) <= k:
            xs_temp[j] = np.random.normal(data.values[i, 1], sigma)
        else:
            xs_temp[j] = np.random.normal(-1*data.values[i, 1], sigma)
    env_label[i] = env
    xs_temp.reshape(1,-1)
    xs_total.append(xs_temp)
    xs_temp = np.zeros(xs_length)

xs_column = ['xs' + str(i) for i in range(xs_length)]
data[xs_column] = xs_total
data['env'] = env_label

print(data)

data.to_csv('Syntheic_data(pv=0.9,0.5,0.6,0.2,0.1).csv')