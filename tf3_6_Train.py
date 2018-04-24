import tensorflow as tf
import numpy as np

#一次几口
BATCH_SIZE = 8
seed = 23455

#基于seed产生随机数
rng = np.random.RandomState(seed)
X = rng.rand(32,2)
#体积重量,评判合格标准是小于1合格
Y = [[int(x0+x1<1)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y)

#定义神经网络输入参数和输出
#体积/质量
x = tf.placeholder(tf.float32,shape=(None,2))
#标签
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 = tf.Variable(tf.random_normal([2, 3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3, 1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数
loss = tf.reduce_mean(tf.square(y - y_))
#梯度优化
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("Non Train DATA:\n")
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))

    #开始训练
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x: X[start:end], y_: Y[start:end]})
        if i%500 == 0:
            total_loss  = sess.run(loss, feed_dict={x: X , y_: Y})
            print("After %d training steps,loss of all data is %g \n" % (i,total_loss))
    print("w1:\n",sess.run(w1))
    print("w2:\n", sess.run(w2))

