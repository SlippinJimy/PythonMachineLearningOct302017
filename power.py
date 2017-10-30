import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from sklearn import preprocessing
from datetime import datetime

RANDOM_SEED = 193
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)

def forwardprop(X, w_1, b_1,w_2,b_2,  w_3, b_3, w_4, b_4):
    h1=tf.nn.tanh(tf.add(tf.matmul(X,w_1), b_1))
    h2=tf.nn.relu(tf.add(tf.matmul(h1,w_2), b_2))
    h3=tf.nn.relu(tf.add(tf.matmul(h1,w_3), b_3))
    yhat=tf.add(tf.matmul(h3,w_4),b_4)
    return yhat

print(str(datetime.now()))
for j in range(1):
    path = '/home/machinelearningstation/PycharmProjects/power/Two/data/data'+str(j)+'.csv'
    print('Dataset '+ str(j+1))
    df0 = pd.read_csv(path, error_bad_lines=False)
    df = df0.replace([np.inf, -np.inf], np.nan).dropna()
    df.drop(df.columns[0], axis=1, inplace=True)
    df_features = df.drop('marker', axis=1)
    data = preprocessing.scale(df_features)

    le = preprocessing.LabelEncoder()
    target = le.fit_transform(df['marker'])

    all_X=data

    num_labels=2
    all_Y=np.eye(num_labels)[target]
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.2, random_state=RANDOM_SEED)

    x_size=train_X.shape[1]
    h1_size=500
    h2_size=1000
    h3_size=500
    y_size=train_y.shape[1]

    X=tf.placeholder("float", shape=[None, x_size], name='X')
    y=tf.placeholder("float", shape=[None, y_size], name='y')

    w_1=init_weights((x_size,h1_size), 'w_1')
    b_1=init_weights((1,h1_size),'b_1')
    w_2=init_weights((h1_size,h2_size), 'w_2')
    b_2=init_weights((1,h2_size), 'b_2')
    w_3=init_weights((h1_size,h3_size), 'w_3')
    b_3=init_weights((1,h3_size), 'b_3')
    w_4=init_weights((h3_size, y_size), 'w_4')
    b_4=init_weights((1, y_size), 'b_4')

    yhat=forwardprop(X, w_1, b_1, w_2, b_2,w_3, b_3, w_4, b_4 )
    y_predict=tf.argmax(yhat,axis=1, name='y_prediction')
    y_true=np.argmax(test_y, axis=1)

    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates=tf.train.AdamOptimizer(0.01).minimize(cost)

    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    J0=0
    J1=10
    epoch=1
    tolerance=1e-8
    while abs(J1-J0)>=tolerance and epoch<10000:
        J0=J1
        sess.run(updates, feed_dict={X: train_X, y: train_y})

        J1 = sess.run(cost, feed_dict={X: train_X, y: train_y})
        epoch+=1

    accuracy=np.mean(y_true==sess.run(y_predict, feed_dict={X:test_X, y:test_y}))
    precison=sklm.precision_score(y_true, sess.run(y_predict, feed_dict={X:test_X, y:test_y}))
    recall=sklm.recall_score(y_true, sess.run(y_predict, feed_dict={X:test_X, y:test_y}))
    f1=sklm.f1_score(y_true, sess.run(y_predict, feed_dict={X:test_X, y:test_y}))
    confusion=sklm.confusion_matrix(y_true, sess.run(y_predict, feed_dict={X:test_X, y:test_y}))

    print("accuracy = %.2f%%, precision = %.2f%%, recall =%.2f%%, f1_score=%.2f%% "
            % (100.* accuracy, 100.* precison, 100.*recall, 100.*f1))
    print "confusion_matrix"
    print confusion

    sess.close()
print(str(datetime.now()))
