#coding:utf-8

"""
TensorFlowの概念を説明する
「ノード」はテンソルを受け取り、(オペレーション)計算を行い、別の「ノード」にテンソルを渡す。
「エッジ」は「ノード」間のテンソルの流れを表す。
"""

import tensorflow as tf

a = tf.constant(3, name='const1') # tf.constantは定数を出力するオペレーション
b = tf.Variable(0, name='val1') # tf.Variableは変数を出力するオペレーション

# aとbを足す
add = tf.add(a,b)

#変数bに足した結果をアサイン
assign = tf.assign(b, add) #tf.assignは値を変数に割り当てるオペレーション ここではbにaddを代入
c = tf.placeholder(tf.int32, name='input') # tf.placeholderは実行時に値が決まる変数(型だけ定義しておく)

#アサインした結果とcを掛け算
mul = tf.multiply(assign,c) #tf.multiplyは直積

#変数の初期化
init = tf.global_variables_initializer() #tf.g..initializerはモデル内のすべての変数をinitializerに従って初期化する

with tf.Session() as sess:
    #初期化を実行
    sess.run(init)
    for i in range(3):
        #掛け算を計算するまでのループを実行
        print(sess.run(mul,feed_dict={c:3})) #feed_dictでplaceholderに値を与える

with tf.Session() as sess:
    #初期化を実行
    sess.run(init)
    for i in range(3):
        #掛け算を計算するまでのループを実行
        print(sess.run(add)) 
