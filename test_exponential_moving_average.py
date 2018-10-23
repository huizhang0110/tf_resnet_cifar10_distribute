import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


v1 = tf.Variable(0, dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(0.99)
moving_average_op = ema.apply([v1])  # 运行滑动平均的op, 会创建一个 shadow_v1
# 每次运行的时候 shadow_variable = decay * shadow_variable + (1 - decay) * variable
# 通过 ema.average(v1)得到 shadow_v1

with tf.Session() as sess:
    sess.run([
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ])

    print(sess.run([v1, ema.average(v1)]))   # 初始值都是0
    sess.run(tf.assign(v1, 5))  # 5->v1

    sess.run(moving_average_op)
    print(sess.run([v1, ema.average(v1)]))  # 5, 0.99 * 0 + 0.01 * 5 = 0.05
    sess.run(tf.assign(v1, 10))

    sess.run(moving_average_op)
    print(sess.run([v1, ema.average(v1)]))  # 10, 0.99 * 0.05 + 0.01 * 10 = 0.1495

    

"""Note:
tf.train.ExponentialMovingAverage()提供了自动更新decay的计算方法，通过传入 step
decay = min(decay, (1 + steps) / (10 + steps))
"""
