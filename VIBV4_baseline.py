import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing

tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


input_dim = 5
label_dim = 2

n_hidden_1 = 16
n_hidden_2 = 16
n_hidden_3 = 4
n_hidden_4 = label_dim

BETA = 0.01


input = tf.placeholder(tf.float32, shape=[None, input_dim], name='input')
labels = tf.placeholder(tf.int64, shape=[None,label_dim], name='labels')
ds = tf.contrib.distributions

class Data_Buffer:

    def __init__(self):
        self.input_buf = None
        self.label_buf = None
        self.env_buf = None

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(input = self.input_buf[idxs],
                    label = self.label_buf[idxs],
                    env = self.env_buf[idxs])

    def load_from_csv(self,colum_dict, data_size):
        self.input_buf = colum_dict['input'].to_numpy()
        self.label_buf = colum_dict['label'].to_numpy().reshape(-1,1)

        encoder = preprocessing.OneHotEncoder()
        encoder.fit(self.label_buf)
        self.label_buf = encoder.transform(self.label_buf).toarray()

        self.env_buf = colum_dict['env'].to_numpy()
        self.size = data_size

def weights(shape, Vname):
    initial = tf.truncated_normal(shape, stddev=0.1, name=Vname)
    return tf.Variable(initial)

def bias(shape, Vname):
    bias = tf.constant(0.1, shape=shape, name=Vname)
    return tf.Variable(bias)


def mulitlayer_perceptron (images, weights, bias):

    # First Hidden Layer
    W1 = weights([input_dim, n_hidden_1], 'W1')
    tf.summary.histogram('W1', W1)

    b1 = bias([n_hidden_1], 'b1')
    tf.summary.histogram('b1', b1)

    layer_1 = tf.add(tf.matmul(images, W1), b1)
    tf.summary.histogram('layer1', layer_1)

    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('relu1', layer_1)

    # Second Hidden Layer
    W2 = weights([n_hidden_1, n_hidden_2], 'W2')
    tf.summary.histogram('W2', W2)

    b2 = bias([n_hidden_2], 'b2')
    tf.summary.histogram('b2', b2)

    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    tf.summary.histogram('layer2', layer_2)

    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('relu2', layer_2)

    #Third Hidden Layer
    W3 = weights([n_hidden_2, n_hidden_3], 'W3')
    tf.summary.histogram('W3', W3)

    b3 = bias(([n_hidden_3]), 'b3')
    tf.summary.histogram('b3', b3)

    layer_3 = tf.add(tf.matmul(layer_2, W3), b3)
    tf.summary.histogram('Encoder', layer_3)

    # Defining Pz(z)
    mu, rho = layer_3[:, :int(n_hidden_3/2)], layer_3[:, int(n_hidden_3/2):]
    tf.summary.histogram('mu_z', mu)
    tf.summary.histogram('sigma_z', rho)

    encoding = tf.contrib.distributions.NormalWithSoftplusScale(mu, rho)
    tf.summary.histogram('P(z|x)', encoding.sample())

    # Forth Hidden Layer = Decoder  #layer4 decoder
    W4 = weights([int(n_hidden_3/2), n_hidden_4], 'W4')
    tf.summary.histogram('W4', W4)

    b4 = bias([n_hidden_4], 'b4')
    tf.summary.histogram('b4', b4)

    layer_4 = tf.add(tf.matmul(encoding.sample(), W4), b4)
    tf.summary.histogram('Decoder', layer_4)

    return encoding, layer_4


with tf.name_scope('Model'):
    encoding, pred = mulitlayer_perceptron(input, weights, bias)

with tf.name_scope('aproximation_to_prior'):
    prior = tf.contrib.distributions.Normal(0.0, 1.0)
    tf.summary.histogram('r(z)', prior.sample())
    #prior_2 = tf.contrib.distributions.Normal(mu_zy, rho_zy)

with tf.name_scope('Class_Loss'):
    class_loss = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=labels)



with tf.name_scope('Info_Loss'):
    info_loss = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(encoding, prior), 0)) /math.log(2)

with tf.name_scope('Total_Loss'):
    total_loss = class_loss + BETA * info_loss
    tf.summary.scalar('Total_Loss', total_loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred, 1), tf.arg_max(labels, 1)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

IZY_bound = math.log(10, 2) - class_loss
tf.summary.scalar('I(Z;Y)', IZY_bound)

IZX_bound = info_loss
tf.summary.scalar('I(Z;X)', IZX_bound)

batch_size = 50
steps_per_batch = int(20000 / batch_size)

summary_writer = tf.summary.FileWriter('/home/ali/TensorBoard', graph=tf.get_default_graph())

#Optimization
with tf.name_scope('Optimizer'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update = ma.apply(tf.model_variables())

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt, global_step, update_ops=[ma_update])

tf.global_variables_initializer().run()
merged_summary_op = tf.summary.merge_all()

def evaluate_test():
    testbatch = test_buffer.sample_batch(20000)
    IZY, IZX, acc = sess.run([IZY_bound, IZX_bound, accuracy], feed_dict={input: testbatch['input'], labels: testbatch['label']})
    return IZY, IZX, acc, 1-acc

#traing data loading
train_dataset = pd.read_csv("Syntheic_data(pv=0.9,0.999,0.8).csv")

input_column1 = ['xv']
input_column2 = ['xs' + str(i) for i in range(input_dim-1)]

column_dict = {'input':train_dataset[input_column1+input_column2],
               'label':train_dataset['y'],
               'env':train_dataset['env']}

train_buffer = Data_Buffer()
train_buffer.load_from_csv(column_dict, len(train_dataset))

#test data loading
test_dataset = pd.read_csv("Syntheic_data(pv=0.9,0.5,0.6,0.2,0.1).csv")

test_buffer = Data_Buffer()
test_buffer.load_from_csv(column_dict, len(test_dataset))

# main
import sys
c1 = np.zeros(steps_per_batch)

for epoch in range(500):
    for step in range(int(steps_per_batch)):

        batch = train_buffer.sample_batch(batch_size)

        _, c = sess.run([train_tensor, total_loss], feed_dict={input: batch['input'], labels: batch['label']})

    print("{}: IZY_Test={:.2f}\t IZX_Test={:.2f}\t acc_Test={:.4f}\t err_Test={:.4f}".format(epoch, *evaluate_test()))
    sys.stdout.flush()

savepth = saver.save(sess, '/tmp/mnistvib', global_step)
