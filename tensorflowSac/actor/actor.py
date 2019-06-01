from tensorflowSac.model_config import actorArchSettings
import tensorflow as tf
import numpy as np
import os


class actor:

    def constructPredictor(self,inputPlaceholders):
        self.input = inputPlaceholders
        arch = actorArchSettings

        # inisitial value is special, therefor explicit init
        layering = inputPlaceholders
        for key in arch.keys():
            layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, arch[key]['name'])

        self.predictor = layering

    # todo: complete function
    def constructOptimizer(self):
        pass

    def predict(self,tfSession, state):
        return tfSession.run([self.predictor], feed_dict={self.input: state})


params = np.ones([2,200])
inputPlaceholder = tf.placeholder(tf.float32, shape=(None,200), name='inputToNN')
actor = actor()
actor.constructPredictor(inputPlaceholder)
actor.constructOptimizer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    fcn = actor.predict(sess, params)
    print(fcn)
    summary_writer = tf.summary.FileWriter(os.getcwd(), graph=tf.get_default_graph())

