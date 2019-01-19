#coding:utf-8

"""
MNISTを用いた手書き数字認識のサンプルプログラム
MNISTデータは28x28=784のグレースケール画像
入力層はユニット数784
中間層はユニット数64, 活性化関数はReLU
出力層は10(0~9の各数字である確率), 活性化関数はSoftmax
誤差関数は二乗誤差関数

入力テンソルは[ミニバッチサイズ x 画素数(784)]の二階テンソル
出力テンソルは[ミニバッチサイズ x 数字の数(10)]の二階テンソル
"""

#MNISTパッケージのインポート
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("data/", one_hot=True)

#全訓練データの取得
#訓練用のデータ、正解データをミニバッチ数を指定して取得
#バッチ: すべての学習データから適当な数だけ選んで学習に用いる
train_images, train_labels = mnist.train.next_batch(50)

#テスト用の画像データを取得
test_images = mnist.test.images
#テスト用の全正解データを取得
test_labels = mnist.test.labels

# ---------------------------------
#入力層
#入力データを定義
x = tf.placeholder(tf.float32,[None,784]) #Noneはバッチサイズが未定義であることを示す

#入力層から中間層
w_1 = tf.Variable(tf.truncated_normal([784, 64],stddev=0.1),name="w1") #標準偏差0.1で,784x64の重み行列を生成
b_1 = tf.Variable(tf.zeros([64]),name="b1") #初期値0でバイアスを生成
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1) #出力を活性化関数=Reluで定義

#中間層から出力層
w_2 = tf.Variable(tf.truncated_normal([64, 10],stddev=0.1),name="w2")
b_2 = tf.Variable(tf.zeros([10]),name="b2")
out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2) #最終的な出力はsoftmax関数で

#教師データを定義
y = tf.placeholder(tf.float32,[None,10])
#誤差関数を定義
loss = tf.reduce_mean(tf.square(y - out)) #tf.squareで二乗を計算 tf.reduce_meanで平均を取る
#訓練を定義
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss) #確率的勾配降下法,学習率0.5

#評価を定義
correct = tf.equal(tf.argmax(out,1),tf.argmax(y,1)) #argmax()で最大値を持っている要素番号を求めている
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

#変数の初期化
init = tf.global_variables_initializer()

# ---------------------------------
with tf.Session() as sess:
    sess.run(init)
    #テストデータをロード
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for i in range(1000):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x:train_images, y:train_labels})

        if step % 10 == 0:
            acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
            print('Step %d: accuracy = %.2f' % (step,acc_val))
