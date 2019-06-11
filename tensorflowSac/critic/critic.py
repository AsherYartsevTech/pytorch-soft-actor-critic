from tensorflowSac.model_config import criticArchSettings
import tensorflow as tf


def tfNameScoping(method):

    def methodWithTfNameScope(classInstance, *kwargs):
        with tf.name_scope(classInstance.nameScope):
            return method(classInstance, *kwargs)

    return methodWithTfNameScope

class critic:
    def __init__(self, nameScope):
        self.nameScope = nameScope

    @tfNameScoping
    def construct(self,inputPlaceholders):
        self.input = inputPlaceholders
        self.baseLayer = tf.concat(self.input, axis=-1 ,name='stateAndActionConcatinator')
        arch = criticArchSettings
        # inisitial value is special, therefor explicit init
        layering = self.baseLayer
        for key in arch.keys():
            layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, arch[key]['name'])
        log_prob_vector = layering
        # probs = tf.exp(log_prob_vector, name='logProbToProb')
        #todo: add another net to produce 2 independent log_prob_vector
        self.criticisor = log_prob_vector

    def constructOptimizer(self, grndTruthPlaceHolder):
        self.grndTruth = grndTruthPlaceHolder
        if not self.nameScope.startswith('target'):
            self.mseLoss = tf.losses.mean_squared_error(grndTruthPlaceHolder, self.criticisor)
            optimizer = tf.train.AdamOptimizer()
            self.optimizationOp = optimizer.minimize(self.mseLoss)

    def optimize(self,sess,grndTruth, nextActionStateFeed):

        sess.run(self.optimizationOp,{self.grndTruth: grndTruth,
                                             self.input[0]: nextActionStateFeed['action'],
                                             self.input[1]: nextActionStateFeed['state'] })

        return sess.run(self.mseLoss,feed_dict={self.grndTruth: grndTruth,
                                             self.input[0]: nextActionStateFeed['action'],
                                             self.input[1]: nextActionStateFeed['state'] } )


    def criticize(self,tfSession, runTimeInputs):

        logProbs = tfSession.run(self.criticisor, feed_dict={self.input[0]: runTimeInputs['action'],
                                                                               self.input[1]: runTimeInputs['state']})

        return logProbs

    #todo: resolve identical namescopes for each method called under this decorator\
    #resulting in tedious default numerating in tensorboard
    @tfNameScoping
    def softCopyWeightsToOtherCritic(self,sess, otherCritic):
        '''

        :param sess: an active tf.Session() instance
        :param otherCritic: object of class critic with same architechture
        :return: the values of all variables of calling critic, which are a result of tf.assign op evaluation
        '''
        # todo: optimize
        # create N tf ops of assigning values. N is the quantity of trainable variables
        update_weights = [tf.assign(dstCriticVar, srcCriticVar) for (dstCriticVar, srcCriticVar) in
                          zip(tf.trainable_variables(otherCritic.nameScope), tf.trainable_variables(self.nameScope))]
        return sess.run(update_weights)


