import tensorflow as tf
# H(x) = Wx + b
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b
# Variance
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# 1. 그래프 구현
# Gradient descent
# Cost를 Minimize하기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
# 2. 실행(세션 만들기), 결과 받기
sess = tf.Session
sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    # 약 2000번 정도 step을 진행하는데, 그냥 20번 중 1번씩만 출력.
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# PlaceHolder
# 직접 값을 주지 않고, 필요할 때 값을 넣는다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
# Result: 7.5 / [ 3. 7.]

# PlaceHolder를 이용한 LinearRegression
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

for step in range(2001):
    # train에 들어가는 value는 필요 없으니까 _로 둠
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
# PlaceHolder 예제
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# [None]: 1차원이고, 아무 값이나 들어올 수 있음. 1개 2개 20개... 등등
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
# Graph launch
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # train 이라는 노드를 실행시킬 때, 값을 넘겨준다.
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         # 값을 넘겨줌
                                         feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing
print(sess.run(hypothesis, feed_dict={X: [5]}))
# Result: [6.10045338]
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
# Result: [3.59963846]
print(sess.run(hypothesis, feed_dict={X: [1.2, 3.5]}))
# Result: [2.59931231 4.59996414]
