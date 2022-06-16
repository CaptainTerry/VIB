import numpy as np
import pandas as pd

sigma = 0.1
xs_length = 4
data_size = 2000
pv = 0.8

# breakpoint()
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
        y[i] = np.sign(data.values[i, 0])

data['y'] = y

#generate xs

def ps(input):
    if input <= xs_length/4:
        probility = 0.9
    elif input <= 2*xs_length/4 and input > xs_length/4:
        probility = 0.8
    elif input <= 3*xs_length/4 and input > 2*xs_length/4:
        probility = 0.7
    elif input <= 4*xs_length/4 and input > 3*xs_length/4:
        probility = 0.6

    return probility

def envs(input):
    if input <= data_size/2:
        probility = 0.9
        env = 1
    else:
        probility = 0.8
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

data.to_csv('Syntheic_data_test.csv')