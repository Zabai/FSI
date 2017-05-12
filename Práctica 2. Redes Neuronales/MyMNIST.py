import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
#
# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]


# TODO: the neural net!!
x_trainingData = train_x
y_trainingData = one_hot(train_y, 10)

x_validationData = valid_x
y_validationData = one_hot(valid_y, 10)

x_testingData = test_x
y_testingData = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W = tf.Variable(np.float32(np.random.rand(28*28, 10)) * 0.1)	# 28*28 entradas x 10 neuronas en capa oculta, 0.1 para reducir los valores
b = tf.Variable(np.float32(np.random.rand(10)) * 0.1) 		    # 10 bias, una para cada neurona

y = tf.nn.softmax(tf.matmul(x, W) + b)	# Softmax de [(Matmul: multiplica la salida de las neuronas por pesos) + el bias]

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))   # Metodo de TF

#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
                                                                # Optimizador descenso por el gradiente para Minimizar el "error"
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)   # learning rate: 0.5

init = tf.initialize_all_variables()	# Inicializa variables

sess = tf.Session()	# Crea una sesion
sess.run(init)		# Volcar en GPU las variables

print "----------------------"
print "   Start training...  "
print "----------------------"

epoch = 0
batch_size = 100	# Cantidad del lote
error = 0
prevError = -1

while(1):
    for jj in xrange(len(x_trainingData) / batch_size):
        batch_xs = x_trainingData[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_trainingData[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_validationData, y_: y_validationData})

    if(abs(error - prevError) < 0.00005): break

    print "Epoch #:", epoch, "Error: ", error

    epoch += 1
    prevError = error

# for epoch in xrange(12):	# Epocas
#     for jj in xrange(len(x_trainingData) / batch_size):
#         batch_xs = x_trainingData[jj * batch_size: jj * batch_size + batch_size]
#         batch_ys = y_trainingData[jj * batch_size: jj * batch_size + batch_size]
#         sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
#
#     if(error > 0){
#         previousError = error
#     }
#     error = sess.run(loss, feed_dict={x: x_validationData, y_: y_validationData})
#     print "Epoch #:", epoch, "Error: ", error


print "----------------------------------------------------------------------------------"
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: x_testingData, y_: y_testingData}))

result = sess.run(y, feed_dict={x: x_testingData})
ok = 0
bad = 0
for b, r in zip(y_testingData, result):
    #print b, "-->\n", r.round(0)
    if(np.array_equal(b, r.round(0))):
        ok += 1
        #print "OK\n\n"
    else:
        bad += 1
        #print "Miss\n\n"
print "OK = ", ok
print "Miss = ", bad
error = float(bad) / float(ok + bad) * 100.
print "Error = ", error, "%"
print "----------------------------------------------------------------------------------"
