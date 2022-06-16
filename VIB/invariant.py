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

sparsity_lambda = 1
continuity_lambda = 5
sparsity_percentage = 0.2
diff_lambda = 10
BETA = 0.01
# learning_rate = 1e-3

n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_3 = 512
n_hidden_4 = label_dim
n_hidden_5 = label_dim

input = tf.placeholder(tf.float32, shape=[None, input_dim], name='input')
labels = tf.placeholder(tf.int64, shape=[None, label_dim], name='labels')
envs = tf.placeholder(tf.float32, shape=[None, 1], name='envs')
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

        self.env_buf = colum_dict['env'].to_numpy().reshape(-1,1)
        self.size = data_size


def weights(shape, Vname):
    initial = tf.truncated_normal(shape, stddev=0.1, name=Vname)
    return tf.Variable(initial)

def bias(shape, Vname):
    bias = tf.constant(0.1, shape=shape, name=Vname)
    return tf.Variable(bias)

def mulitlayer_perceptron (input, envs, weights, bias):

    # First Hidden Layer
    W1 = weights([input_dim, n_hidden_1], 'W1')
    tf.summary.histogram('W1', W1)

    b1 = bias([n_hidden_1], 'b1')
    tf.summary.histogram('b1', b1)

    layer_1 = tf.add(tf.matmul(input, W1), b1)
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
    mu, rho = layer_3[:, :256], layer_3[:, 256:]
    tf.summary.histogram('mu_z', mu)
    tf.summary.histogram('sigma_z', rho)

    encoding = tf.contrib.distributions.NormalWithSoftplusScale(mu, rho)
    tf.summary.histogram('P(z|x)', encoding.sample())

    # Forth Hidden Layer = Env-agnostic predictors (decoder)
    W4 = weights([256, n_hidden_4], 'W4')
    tf.summary.histogram('W4', W4)

    b4 = bias([n_hidden_4], 'b4')
    tf.summary.histogram('b4', b4)

    layer_4 = tf.add(tf.matmul(encoding.sample(), W4), b4)
    tf.summary.histogram('Env-agnostic', layer_4)

    #Five Hidden Layer = Env-aware predictors
    W5 = weights([257, n_hidden_5], 'W5')
    tf.summary.histogram('W5', W5)

    b5 = bias([n_hidden_5], 'b5')
    tf.summary.histogram('b5', b5)

    layer_5 = tf.add(tf.matmul(tf.concat([encoding.sample(), tf.reshape(envs, [-1,1])], axis=-1), W5), b5)
    tf.summary.histogram('Env-aware', layer_5)

    return encoding, layer_4, layer_5

def inv_rat_loss(env_inv_logits, env_enable_logits, labels):
    """
    Compute the loss for the invariant rationalization training.
    Inputs:
        env_inv_logits -- logits of the predictor without env index
                          (batch_size, num_classes)
        env_enable_logits -- logits of the predictor with env index
                          (batch_size, num_classes)
        labels -- the groundtruth one-hot labels
    """
    env_inv_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=env_inv_logits, labels=labels)

    env_enable_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=env_enable_logits, labels=labels)

    env_inv_loss = tf.reduce_mean(env_inv_losses)
    env_enable_loss = tf.reduce_mean(env_enable_losses)

    diff_loss = tf.math.maximum(0., env_inv_loss - env_enable_loss)

    return env_inv_loss, env_enable_loss, diff_loss

with tf.name_scope('Model'):
    rationale, pred_agnostic, pred_aware = mulitlayer_perceptron(input, envs, weights, bias)

with tf.name_scope('aproximation_to_prior'):
    prior = tf.contrib.distributions.Normal(0.0, 1.0)
    tf.summary.histogram('r(z)', prior.sample())

with tf.name_scope('Info_Loss'):
    info_loss = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(rationale, prior), 0)) / math.log(2)

with tf.name_scope('Loss'):
    env_inv_loss, env_enable_loss, diff_loss = inv_rat_loss(pred_agnostic, pred_aware, labels)

with tf.name_scope('Gen_Loss'):
    gen_loss = diff_lambda * diff_loss + env_inv_loss + BETA*info_loss
    tf.summary.scalar('Gen_Loss', gen_loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred_agnostic, 1), tf.arg_max(labels, 1)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

batch_size = 100
steps_per_batch = 2000
summary_writer = tf.summary.FileWriter('/home/ali/TensorBoard', graph=tf.get_default_graph())

#Optimization

with tf.name_scope('Optimizer1'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update = ma.apply(tf.model_variables())

with tf.name_scope('Optimizer2'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
    opt2 = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma2 = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update2 = ma2.apply(tf.model_variables())

with tf.name_scope('Optimizer3'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
    opt3 = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma3 = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update3 = ma3.apply(tf.model_variables())

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

gen_tensor = tf.contrib.training.create_train_op(gen_loss, opt, global_step, update_ops=[ma_update])
env_inv_tensor = tf.contrib.training.create_train_op(env_inv_loss, opt2, global_step, update_ops=[ma_update2])
env_enable_tensor = tf.contrib.training.create_train_op(env_enable_loss, opt3, global_step, update_ops=[ma_update3])

tf.global_variables_initializer().run()
merged_summary_op = tf.summary.merge_all()


GEN_Loss = gen_loss
tf.summary.scalar('generator_loss', GEN_Loss)
ENV_INV_Loss = env_inv_loss
tf.summary.scalar('env_inv_loss', ENV_INV_Loss)
ENV_ENABLE_Loss = env_enable_loss
tf.summary.scalar('env_enable_loss', ENV_ENABLE_Loss)

def evaluate_test():
    testbatch = test_buffer.sample_batch(batch_size)
    GEN, ENV_INV, ENV_ENABLE,acc = sess.run([GEN_Loss, ENV_INV_Loss, ENV_ENABLE_Loss,accuracy], feed_dict={input: testbatch['input'], labels: testbatch['label'], envs: testbatch['env']})
    return GEN, ENV_INV, ENV_ENABLE,acc

# main

#traing data loading
train_dataset = pd.read_csv("Syntheic_data.csv")

input_column = ['xv', 'xs0', 'xs1', 'xs2', 'xs3']

column_dict = {'input':train_dataset[input_column],
               'label':train_dataset['y'],
               'env':train_dataset['env']}

train_buffer = Data_Buffer()
train_buffer.load_from_csv(column_dict, len(train_dataset))

#test data loading
test_dataset = pd.read_csv("Syntheic_data_test.csv")

test_buffer = Data_Buffer()
test_buffer.load_from_csv(column_dict, len(test_dataset))

for epoch in range(100):
    for step in range(int(steps_per_batch)):
        path = step %7
        #get data
        batch = train_buffer.sample_batch(batch_size)


        if path == 0:
            # update the generator
            _, c = sess.run([gen_tensor, gen_loss], feed_dict={input: batch['input'], labels: batch['label'], envs: batch['env']})
        if path == [1,2,3]:
            _, c = sess.run([env_inv_tensor, env_inv_loss], feed_dict={input: batch['input'], labels: batch['label'], envs: batch['env']})
        if path == [4,5,6]:
            _, c = sess.run([env_enable_tensor, env_enable_loss], feed_dict={input: batch['input'], labels: batch['label'], envs: batch['env']})

    print("{}: gen_Test={:.2f}\t inv_Test={:.2f}\t enable_Test={:.2f}\t acc={:.4f}".format(epoch, *evaluate_test()))
    sys.stdout.flush()

savepth = saver.save(sess, '/tmp/mnistvib')
