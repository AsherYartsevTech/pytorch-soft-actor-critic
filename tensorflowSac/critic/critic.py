from tensorflowSac.model_config import leftHemisphereCriticArchSettings, rightHemisphereCriticArchSettings
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

        # construct left hemisphere
        arch = leftHemisphereCriticArchSettings
        # inisitial value is special, therefor explicit init
        layering = self.baseLayer
        for key in arch.keys():
            layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, nameScope=key)
        log_prob_vector = layering
        self.leftCriticisor = log_prob_vector

        # construct right hemisphere
        arch = rightHemisphereCriticArchSettings
        # inisitial value is special, therefor explicit init
        layering = self.baseLayer
        for key in arch.keys():
            layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, nameScope=key)
        log_prob_vector = layering
        self.rightCriticisor = log_prob_vector




    def constructOptimizer(self, grndTruthPlaceHolder):
        self.grndTruth = grndTruthPlaceHolder
        if not self.nameScope.startswith('target'):
            self.leftMseLoss = tf.losses.mean_squared_error(grndTruthPlaceHolder, self.leftCriticisor)
            self.rightMseLoss = tf.losses.mean_squared_error(grndTruthPlaceHolder, self.rightCriticisor)

            optimizer = tf.train.AdamOptimizer()
            self.optimizeLeftHemisphere = optimizer.minimize(self.leftMseLoss)
            self.optimizeRightHemisphere = optimizer.minimize(self.rightMseLoss)

    def optimize(self,sess,grndTruth, nextActionStateFeed):

        sess.run(self.optimizeLeftHemisphere,{self.grndTruth: grndTruth,
                                             self.input[0]: nextActionStateFeed['action'],
                                             self.input[1]: nextActionStateFeed['state'] })

        sess.run(self.optimizeRightHemisphere, {self.grndTruth: grndTruth,
                                               self.input[0]: nextActionStateFeed['action'],
                                               self.input[1]: nextActionStateFeed['state']})

        return sess.run([self.leftMseLoss, self.rightMseLoss], feed_dict={self.grndTruth: grndTruth,
                                             self.input[0]: nextActionStateFeed['action'],
                                             self.input[1]: nextActionStateFeed['state'] } )


    def criticize(self,tfSession, runTimeInputs):

        leftHemisphereLogProbs,rightHemisphereLogProbs = tfSession.run([self.leftCriticisor, self.rightCriticisor], feed_dict={self.input[0]: runTimeInputs['action'],
                                                                               self.input[1]: runTimeInputs['state']})

        return leftHemisphereLogProbs,rightHemisphereLogProbs

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


