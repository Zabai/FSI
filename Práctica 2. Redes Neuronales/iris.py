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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)	# 4 entradas x 5 neuronas en capa oculta, 0.1 para reducir los valores
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1) 		# 5 bias, una para cada neurona

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)	# 5 neuronas capa oculta x 3 neuronas capa final
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)		# 3 bias, una para cada neurona final

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)	# Sigmoide de [(Matmul: multiplica entradas por pesos) + el bias]
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)	# Softmax de [(Matmul: multiplica la salida de las neuronas por pesos) + el bias]
											# Softmax garantiza que la suma de las salidas sea igual a 1

loss = tf.reduce_sum(tf.square(y_ - y))		# Calcula el error

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
																# Optimizador descenso por el gradiente para Minimizar el "error"

init = tf.initialize_all_variables()	# Inicializa variables

sess = tf.Session()	# Crea una sesion
sess.run(init)		# Volcar en GPU las variables

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20	# Cantidad del lote

for epoch in xrange(100):	# 100 epocas
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
