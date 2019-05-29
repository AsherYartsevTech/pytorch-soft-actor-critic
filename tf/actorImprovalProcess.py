import tensorflow as tf
import numpy as np
import actor

params = np.ones([2,1200])
with tf.name_scope('actorImprovalProcess'):

    actor = actor.actor()
    topNode = actor.debug_getBodyHead()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        fcn = sess.run([topNode], feed_dict={actor.actorTfNodes['NN_feeds']['batchStateInput']: params})
        print(fcn)