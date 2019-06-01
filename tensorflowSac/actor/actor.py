from tensorflowSac.model_config import policyArchSettings
import tensorflow as tf
import numpy as np
import os


class actor:

    def constructPredictor(self,inputPlaceholders):
        self.input = inputPlaceholders
        arch = policyArchSettings

        # inisitial value is special, therefor explicit init
        layering = inputPlaceholders
        with tf.name_scope('actionMeans'):
            for key in arch.keys():
                if key.startswith('mean'):
                    layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, arch[key]['name'])
            mean_vector = layering
        # inisitial value is special, therefor explicit init
        layering = inputPlaceholders
        with tf.name_scope('actionLogStds'):
            for key in arch.keys():
                if key.startswith('log_std'):
                    layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, arch[key]['name'])
        log_std_vector = layering

        self.predictor = [mean_vector, log_std_vector]

    # todo: complete function
    def constructOptimizer(self):
        pass

    def predict(self,tfSession, state):
        return tfSession.run(self.predictor, feed_dict={self.input: state})


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

