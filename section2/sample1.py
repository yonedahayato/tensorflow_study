import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets

def sample1():
    sess = tf.Session()

    x_vals = np.array([1., 3., 5., 7., 9.])
    x_data = tf.placeholder(tf.float32)

    m_const = tf.constant(3.)

    my_product = tf.multiply(x_data, m_const)

    for x_val in x_vals:
        print(sess.run(my_product, feed_dict={x_data: x_val}))


def sample2():
    sess = tf.Session()
    my_array = np.array([[1., 3., 5., 7., 9.],
                         [-2., 0., 2., 4., 6.],
                         [-6., -3., 0., 3., 6.]])

    x_vals = np.array([my_array, my_array + 1])
    x_data = tf.placeholder(tf.float32, shape=(3,5))

    m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
    m2 = tf.constant([[2.]])
    a1 = tf.constant([[10.]])

    prod1 = tf.matmul(x_data, m1)
    prod2 = tf.matmul(prod1, m2)
    add1 = tf.add(prod2, a1)

    for x_val in x_vals:
        print(sess.run(add1, feed_dict={x_data: x_val}))


def sample3():
    sess = tf.Session()
    x_shape = [1, 4, 4, 1]
    x_val = np.random.uniform(size=x_shape)
    x_data = tf.placeholder(tf.float32, shape=x_shape)

    my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    move_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides,
                                  padding="SAME", name="Moving_Avg_Window")

    def custom_layer(input_matrix):
        input_matrix_sqeezed = tf.squeeze(input_matrix)
        A = tf.constant([[1., 2.], [-1., 3.]])
        b = tf.constant(1., shape=[2, 2])
        temp1 = tf.matmul(A, input_matrix_sqeezed)
        temp = tf.add(temp1, b)
        return tf.sigmoid(temp)

    with tf.name_scope("Cuntom_Layer") as scope:
        custom_layer1 = custom_layer(move_avg_layer)

    print(sess.run(custom_layer1, feed_dict={x_data: x_val}))


def sample4():
    sess = tf.Session()
    x_vals = tf.linspace(-1., 1., 500)
    target = tf.constant(0.)

    l2_y_vals = tf.square(target - x_vals)
    l2_y_out = sess.run(l2_y_vals)

    l1_y_vals = tf.abs(target - x_vals)
    l1_y_out = sess.run(l1_y_vals)

    delta1 = tf.constant(0.25)
    phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
    phuber1_y_out = sess.run(phuber1_y_vals)

    delta2 = tf.constant(5.)
    phuber2_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
    phuber2_y_out = sess.run(phuber2_y_vals)


def sample5():
    sess = tf.Session()
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)

    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1]))

    my_output = tf.multiply(x_data, A)
    loss = tf.square(my_output - y_target)

    init = tf.global_variables_initializer()
    sess.run(init)


    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    train_step = my_opt.minimize(loss)

    for i in range(1000):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%25 == 0:
            print("Step #{} A={}".format(i+1, str(sess.run(A))))
            print("Loss = {}".format(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))


def sample6():
    sess = tf.Session()
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
    my_output = tf.add(x_data, A)

    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y_target, 0)

    init = tf.global_variables_initializer()
    sess.run(init)

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)

    for i in range(14000):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%200 == 0:
            print("Step #{} A={}".format(i+1, sess.run(A)))
            print("Loss={}".format(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))


def sample7():
    sess = tf.Session()
    batch_size = 20

    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1, 1]))

    my_output = tf.matmul(x_data, A)
    loss = tf.reduce_mean(tf.square(my_output - y_target))

    init = tf.global_variables_initializer()
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    loss_batch = []
    for i in range(100):
        rand_index = np.random.choice(100, size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])

        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%5 == 0:
            print("Step #{} A={}".format(i+1, sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print("Loss={}".format(temp_loss))

            loss_batch.append(temp_loss)


def sample8():
    sess = tf.Session()

    iris = datasets.load_iris()
    binary_target = np.array([1. if x==0 else 0. for x in iris.target])

    iris_2d = np.array([[x[2], x[3]] for x in iris.data])

    batch_size = 20
    x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    my_mult = tf.matmul(x2_data, A)
    my_add = tf.add(my_mult, b)
    my_output = tf.subtract(x1_data, my_add)

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        rand_index = np.random.choice(len(iris_2d), size=batch_size)
        rand_x = iris_2d[rand_index]
        rand_x1 = np.array([[x[0]] for x in rand_x])
        rand_x2 = np.array([[x[1]] for x in rand_x])
        rand_y = np.array([[y] for y in binary_target[rand_index]])

        sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})

        if (i+1)%200 == 0:
            print("Step: {}, A={}, b={}".format(i+1, sess.run(A), sess.run(b)))


def sample9():
    sess = tf.Session()
    batch_size = 25

    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

    x_vals_train = x_vals[train_indices]
    y_vals_train = y_vals[train_indices]

    x_vals_test = x_vals[test_indices]
    y_vals_test = y_vals[test_indices]

    A = tf.Variable(tf.random_normal(shape=[1, 1]))

    my_output = tf.matmul(x_data, A)
    loss = tf.reduce_mean(tf.square(my_output - y_target))

    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = np.transpose([x_vals_train[rand_index]])
        rand_y = np.transpose([y_vals_train[rand_index]])

        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%25 == 0:
            print("Step #{} A={}".format(i+1, sess.run(A)))
            print("loss = {}".format(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))


    mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
    mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})

    print("MSE on test:{}".format(np.round(mse_test, 2)))
    print("MSE on train:{}".format(np.round(mse_train, 2)))

def main():
    sample9()


if __name__ == "__main__":
    main()
