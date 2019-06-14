from tensorflowSac.model_config import policyArchSettings, batchSize
import tensorflow as tf
from tensorflow_probability import distributions as distLib
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)
# def tfNameScoping(method):
#
#     def methodWithTfNameScope(classInstance, *kwargs):
#         with tf.name_scope(classInstance.nameScope):
#             return method(classInstance, *kwargs)
#
#     return methodWithTfNameScope

class actor:
    def __init__(self, nameScope):
        self.nameScope = nameScope
        self.alpha = 0.9
    # @tfNameScoping
    def constructPredictor(self,inputPlaceholders,trainingCriticOpinionOnPolicyChoicesPlaceHolder):
        self.input = inputPlaceholders

        with tf.name_scope('MeansOfEachActionEntry'):
            meanArch = policyArchSettings['deepMean']
            # initial value is special, therefor explicit init
            layering = inputPlaceholders
            for key in meanArch.keys():
                layering = meanArch[key]['builder'](meanArch[key]['builder_params']).construct(layering, nameScope=key)
            mean_vector = layering

        with tf.name_scope('StddevOfEachActionEntry'):
            logStdArch = policyArchSettings['deepStddev']
            # initial value is special, therefor explicit init
            layering = inputPlaceholders
            for key in logStdArch.keys():
                layering = logStdArch[key]['builder'](logStdArch[key]['builder_params']).construct(layering, nameScope=key)
            log_std_vector = layering
            std_vector = tf.exp(log_std_vector, name='logStdToStd')

        with tf.name_scope('finalActionAndLogProbPredictor'):
            NormalDist = distLib.Normal(loc=0., scale=1., name='NormalTensor')
            NormalSamples = NormalDist.sample(tf.shape(std_vector))

            actionsLogits = tf.math.add(mean_vector, tf.math.multiply(std_vector,NormalSamples))

            actions = tf.nn.tanh(actionsLogits, name='LateNonLinearitySolvesRandomDependency')

            logProbs = NormalDist.log_prob(tf.clip_by_value(actionsLogits, clip_value_min=1e-10,clip_value_max=1e+10, name="clipLogProbsOfActionLogits"),
                                                                                    name='logProbsOfActionLogits')

            self.predictor = [actions, logProbs]
            tf.summary.histogram("actorSugeestedActions", actions)
            tf.summary.histogram("actorSugeestedlogProbs", logProbs)

        with tf.name_scope('semi_KL_loss'):
            # it's not a real KL loss since those are not realdistributions which are summed to 1
            LOGPROBS = 1
            self.trainingCriticOpinionOnPolicyChoices = trainingCriticOpinionOnPolicyChoicesPlaceHolder
            # from original KL: p*(log(p)-log(q)) we ommit the p since its the 'p' of the ground truth part, and there for serves as constant
            self.loss = tf.reduce_mean(tf.abs(tf.scalar_mul(self.alpha, self.predictor[LOGPROBS]) - self.trainingCriticOpinionOnPolicyChoices))
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('backprop_gradients'):

            optimizer = tf.train.AdamOptimizer()
            self.optimizationOp = optimizer.minimize(self.loss)

    def predict(self,tfSession,inputPlaceholders):
        def adoptActionToEnv(debugAction):
            return np.ceil(debugAction)


        actions, logProbs = tfSession.run(self.predictor, feed_dict={self.input: inputPlaceholders['state']})
        if inputPlaceholders['state'].shape == (1, 4):
            np.set_printoptions(threshold=sys.maxsize)

        # print("action:{}, actionLogits:{}".format(actions, printActionLogits))
        # todo: a cleaner solution here
        actions = adoptActionToEnv(actions)
        return actions, logProbs

    def optimize(self, sess, trainingCriticOpinionOnPolicyChoices ,nextState, summary_writer, summaries):
        sess.run(self.optimizationOp, {self.trainingCriticOpinionOnPolicyChoices: trainingCriticOpinionOnPolicyChoices, self.input: nextState['state']})


        summaryOutput = sess.run(summaries, feed_dict={self.trainingCriticOpinionOnPolicyChoices: trainingCriticOpinionOnPolicyChoices,
                                       self.input: nextState['state']})
        summary_writer.add_summary(summaryOutput)




