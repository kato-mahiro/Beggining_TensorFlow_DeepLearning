#coding:utf-8
"""
rnn_cell,ラッピング,セルの時間展開の例
"""
import tensorflow as tf

"""まずrnn_cellの基本的な型を決める"""
#中間層のユニット数100の普通のRNNのrnn_cellを定義
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=100)

#LSTMのrnn_cellを定義
cell = tf.nn.rnn_cell.LSTMCell(num_units=100,use_peepholes=True)

"""ドロップアウトや多層化を行いたい場合には以下のようにラッピングしてcellを構築する"""
#普通のRNNのrnn_cellを定義
cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=100)
#LSTMのrnn_cellを定義
cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=100,use_peepholes=True)
#ドロップアウトをcell_2に付与
cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.6)

#cell_1,cell_2の順番でrnn_cellを構築する
cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_1,cell_2])

"""上記のように定義したcellを、XXX_rnnで時間展開する"""
#dynamic_rnnの場合
x = tf.placeholder(tf.float32,[None,max_time,input_size]) #[バッチサイズ、最大時間長、入力長]

#static_rnnの場合
inputs = []
for i in range(time_step):
    x = tf.placeholder(tf.float32,[None,input_size])
    inputs.append(x)
    #実行時に各placeholderを参照するためにcollectionに格納
    tf.add_to_collection("x",x)
