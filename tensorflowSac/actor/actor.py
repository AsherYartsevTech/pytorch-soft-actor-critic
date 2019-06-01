from tensorflowSac.model_config import policyArchSettings
import tensorflow as tf
from tensorflow_probability import distributions as distLib

import numpy as np
import os

def tfNameScoping(method):

    def methodWithTfNameScope(layerBuilderInstance, inputLayer, nameScope):
        with tf.name_scope(nameScope):
            return method(layerBuilderInstance, inputLayer, nameScope)

    return methodWithTfNameScope

class actor:
    @tfNameScoping
    def constructPredictor(self,inputPlaceholders, nameScope):
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

        std_vector = tf.exp(log_std_vector, name='logStdToStd')
        self.actionMeanInput = tf.placeholder(tf.float32,mean_vector.get_shape(), name='actionMean')
        self.actionStdInput = tf.placeholder(tf.float32,std_vector.get_shape(), name='actionStd')
        # todo: NormalDist yields a suspiciously correlated values
        NormalDist = tf.random_normal((),mean=mean_vector, stddev=std_vector, name='Gauss')

        self.predictor = [mean_vector, std_vector, NormalDist]

    # todo: complete function
    def constructOptimizer(self):
        pass

    def predict(self,tfSession, inputPlaceholders):
        def predictActionAndLogProb(meanVec, stdVec, NormalDistVec):
            return meanVec, stdVec, NormalDistVec

        meanVec, stdVec, NormalDist = tfSession.run(self.predictor, feed_dict={self.input: inputPlaceholders['state'],
                                                                                     self.actionMeanInput: inputPlaceholders['normalMean'],
                                                                                     self.actionStdInput: inputPlaceholders['normalStd']})
        return predictActionAndLogProb(meanVec, stdVec, NormalDist)



params = np.ones([2,200])
inputPlaceholders = {
    'state': params,
    'normalMean': np.zeros((2,1)),
    'normalStd': np.ones((2,1))
}
inputPlaceholder = tf.placeholder(tf.float32, shape=(None,200), name='inputToNN')
actor = actor()
actor.constructPredictor(inputPlaceholder, 'actor')
actor.constructOptimizer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    meanVec, stdVec, vecNormal = actor.predict(sess, inputPlaceholders)
    print('meanVec:\n', meanVec)
    print('stdVec:\n', stdVec)
    print('vecNormal:\n', vecNormal)

    summary_writer = tf.summary.FileWriter(os.getcwd(), graph=tf.get_default_graph())

