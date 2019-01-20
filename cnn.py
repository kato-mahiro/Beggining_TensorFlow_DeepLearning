#coding:utf-8

"""
畳み込みニューラルネットワークのサンプル
&
tf.train.Saverにより学習モデルを保存する
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("data/", one_hot=True)
train_images, train_labels = mnist.train.next_batch(50)
test_images = mnist.test.images
test_labels = mnist.test.labels

#入力データを定義
x = tf.placeholder(tf.float32,[None,784]) 
#バッチサイズ、高さ、横幅、チャンネル数に変更
img = tf.reshape(x,[-1,28,28,1])

#畳み込み層1
f1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1)) #フィルタの定義フィルタは[縦x横xチャンネル数xフィルタの枚数]
conv1 = tf.nn.conv2d(img,f1,strides=[1,1,1,1],padding='SAME') #tf.conv2dに画像とフィルタを渡すことで畳み込みNNを実装する
#ストライドは[バッチ方向、縦方向、横方向、チャンネル方向]
#畳み込みを行うと画像が小さくなるのでpaddingで調節する
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(conv1+b1)
#プーリング層1
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#マックスプーリング:フィルタの中から最も大きい画素値を採用する
#ksize:[バッチ方向、縦方向、横方向、チャンネル方向]
#2x2のフィルタでプーリングしたので28x28の画像が14x14になった

#畳み込み層2
f2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
conv2 = tf.nn.conv2d(h_pool1,f2,strides=[1,1,1,1],padding='SAME')
b2 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2 = tf.nn.relu(conv2+b2)

#プーリング層2
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#最終的に7x7 64チャンネルの画像になった

#畳み込まれているものをフラットな形に変換(二次元の画像から一次元の配列にする)
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])

#全結合層
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#出力層
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

#教師データを定義
y = tf.placeholder(tf.float32,[None,10])
#誤差関数を定義
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1])) #クロスエントロピー
#訓練を定義
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss) #確率的勾配降下法,学習率0.5

#評価を定義
correct = tf.equal(tf.argmax(out,1),tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

#変数の初期化
init = tf.global_variables_initializer()

"""実行部分"""
saver = tf.train.Saver() #全てのモデル・変数を対象とする
with tf.Session() as sess:
    sess.run(init)
    #テストデータをロード
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for i in range(101):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x:train_images, y:train_labels})

        if step % 100 == 0:
            acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
            print('Step %d: accuracy = %.2f' % (step,acc_val))

    """モデルの保存"""
    saver.save(sess,'cpkt/cnn_model')
