from tensorflowSac.model_config import policyArchSettings, batchSize
import tensorflow as tf
from tensorflow_probability import distributions as distLib
import numpy as np
import sys
def tfNameScoping(method):

    def methodWithTfNameScope(classInstance, *kwargs):
        with tf.name_scope(classInstance.nameScope):
            return method(classInstance, *kwargs)

    return methodWithTfNameScope

class actor:
    def __init__(self, nameScope):
        self.nameScope = nameScope
        self.alpha = 0.9
    @tfNameScoping
    def constructPredictor(self,inputPlaceholders):
        self.input = inputPlaceholders

        with tf.name_scope('MeansOfEachActionEntry'):
            meanArch = policyArchSettings['meanNN']
            # initial value is special, therefor explicit init
            layering = inputPlaceholders
            for key in meanArch.keys():
                layering = meanArch[key]['builder'](meanArch[key]['builder_params']).construct(layering, meanArch[key]['name'])
            mean_vector = layering

        with tf.name_scope('StddevOfEachActionEntry'):
            logStdArch = policyArchSettings['logStdNN']
            # initial value is special, therefor explicit init
            layering = inputPlaceholders
            for key in logStdArch.keys():
                layering = logStdArch[key]['builder'](logStdArch[key]['builder_params']).construct(layering, logStdArch[key]['name'])
            log_std_vector = layering
            std_vector = tf.exp(log_std_vector, name='logStdToStd')

        with tf.name_scope('finalActionAndLogProbPredictor'):
            NormalDist = distLib.Normal(loc=0., scale=1., name='NormalTensor')
            NormalSamples = NormalDist.sample(tf.shape(std_vector))
            actionsLogits = tf.math.add(mean_vector,tf.math.multiply(std_vector,NormalSamples))
            actions = tf.nn.tanh(actionsLogits, name='LateNonLinearitySolvesRandomDependency')
            # todo: implement the normalization of the logProbs
            # log_prob -= torch.log(1 - action.pow(2) + epsilon)
            # log_prob = log_prob.sum(1, keepdim=True)
            logProbs = NormalDist.log_prob(tf.clip_by_value(actionsLogits,clip_value_min=1e-10,clip_value_max=1e+10),name='logProbsOfActionLogits')
            self.predictor = [actions, logProbs]

    @tfNameScoping
    def constructOptimizer(self, targetCriticGrndTruthPlaceHolder):
        LOGPROBS = 1
        # todo: change horrible name grndTruth
        self.grndTruth = targetCriticGrndTruthPlaceHolder
        self.loss = tf.reduce_sum(tf.pow(tf.math.scalar_mul(self.alpha, self.predictor[LOGPROBS]) - self.grndTruth, 2))
        optimizer = tf.train.AdamOptimizer()
        self.optimizationOp = optimizer.minimize(self.loss)

    def predict(self,tfSession,inputPlaceholders):
        def adoptActionToEnv(debugAction):
            return np.ceil(debugAction)



        actions, logProbs = tfSession.run(self.predictor, feed_dict={self.input: inputPlaceholders['state']})
        if inputPlaceholders['state'].shape == (1, 4):
            print("action:{}".format(actions))
        # todo: a cleaner solution here
        actions = adoptActionToEnv(actions)
        return actions, logProbs

    def optimize(self, sess, grndTruth,nextState):
        sess.run(self.optimizationOp, {self.grndTruth: grndTruth, self.input: nextState['state']})
        return sess.run(self.loss, feed_dict={self.grndTruth: grndTruth, self.input: nextState['state']})



