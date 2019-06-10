from tensorflowSac.model_config import policyArchSettings, batchSize
import tensorflow as tf
from tensorflow_probability import distributions as distLib
import numpy as np

def tfNameScoping(method):

    def methodWithTfNameScope(classInstance, *kwargs):
        with tf.name_scope(classInstance.nameScope):
            return method(classInstance, *kwargs)

    return methodWithTfNameScope

class actor:
    def __init__(self, nameScope):
        self.nameScope = nameScope
        self.alpha = 0
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
            logProbs = NormalDist.log_prob(actionsLogits,name='logProbsOfActionLogits')
            self.predictor = [actions, logProbs]

    @tfNameScoping
    def constructOptimizer(self, targetCriticGrndTruthPlaceHolder):
        self.grndTruth = targetCriticGrndTruthPlaceHolder
        self.loss = tf.reduce_sum(tf.pow(tf.math.scalar_mul(self.alpha, self.predictor[1]) - targetCriticGrndTruthPlaceHolder, 2))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.99)
        self.optimizationOp = optimizer.minimize(tf.cast(self.loss, dtype=tf.float32))

    def predict(self,tfSession,inputPlaceholders):
        def predictActionAndLogProb(actions, logProbs):
            # todo: implement!, if you'll decide that all logProb extraction should appear here
            return actions, logProbs

        def adoptActionToEnv(debugAction):
            return np.clip(debugAction.astype(int), a_min=0, a_max=1)



        actions, logProbs = tfSession.run(self.predictor, feed_dict={self.input: inputPlaceholders['state']})

        # todo: a cleaner solution here
        actions = adoptActionToEnv(actions)
        return predictActionAndLogProb(actions, logProbs)

    def optimize(self, sess, grndTruth,nextState):
        return sess.run(self.optimizationOp, {self.grndTruth: grndTruth, self.input: nextState['state']})



