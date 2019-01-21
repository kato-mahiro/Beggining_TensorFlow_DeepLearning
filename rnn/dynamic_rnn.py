#coding:utf-8

"""dynamic_rnnを用いたrnnモデルの構築例"""
import tensorflow as tf
max_time = 50 #rnnの時系列の最大長

#入力データ
x = tf.placeholder(tf.float32,[None,max_time,input_size])
#rnn_cellを定義
cell = tf.nn.rnn_cell.LSTMCell(num_units=100,use_peepholes=True)
#時間展開して出力を取得
outputs,state = tf.nn.dynamic_rnn(cell=cell,inputs=x,dtype=tf.float32)

"""
このモデルが出力するのは時間展開された中間層出力である。
これは[バッチサイズ、時間長、出力長]の3階テンソルであり、
そのままでは重みをかけたり足したりできない
必要なのは[バッチサイズ、出力長]の2階テンソルなので
テンソルをスライスして加工する
"""
#最後の時間軸のTensorを取得
last_output = outputs[:,-1,:] 

#あとは同じ
w = tf.Variable(tf.truncated_normal([128,10],stddev=0.1))
b = tf.Variable(tf.zeros([10]))

out = tf.nn.sotfmax(tf.matmul(last_output,w)+b)

