from tensorflowSac.model_config import leftHemisphereCriticArchSettings, rightHemisphereCriticArchSettings
import tensorflow as tf


def tfNameScoping(method):

    def methodWithTfNameScope(classInstance, *args):
        with tf.name_scope(classInstance.nameScope) as scope:
            return method(classInstance, *args)

    return methodWithTfNameScope

class critic:
    def __init__(self, nameScope):
        self.nameScope = nameScope
        #todo: extract it later to hyperparameter
        self.tau = 0.005

    @tfNameScoping
    def construct(self,inputPlaceholders, expectedRewardAsGrndTruth):
        self.input = inputPlaceholders
        self.baseLayer = tf.concat(self.input, axis=-1 ,name='stateAndActionConcatinator')
        with tf.name_scope("leftHemisphere"):
            # construct left hemisphere
            arch = leftHemisphereCriticArchSettings
            # inisitial value is special, therefor explicit init
            layering = self.baseLayer
            for key in arch.keys():
                layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, nameScope=key)
                expectedReward = layering
            self.leftExpectedReward = expectedReward
        with tf.name_scope("rightHemisphere"):
            # construct right hemisphere
            arch = rightHemisphereCriticArchSettings
            # inisitial value is special, therefor explicit init
            layering = self.baseLayer
            for key in arch.keys():
                layering = arch[key]['builder'](arch[key]['builder_params']).construct(layering, nameScope=key)
            expectedReward = layering
            self.rightExpectedReward = expectedReward

        if not self.nameScope.startswith('target'):
            with tf.name_scope("mse_losses"):
                self.expectedRewardAsGrndTruth= expectedRewardAsGrndTruth
                self.leftMseLoss = tf.losses.mean_squared_error(self.expectedRewardAsGrndTruth, self.leftExpectedReward, scope="leftHemiLoss")
                self.rightMseLoss = tf.losses.mean_squared_error(self.expectedRewardAsGrndTruth, self.rightExpectedReward, scope="rightHemiLoss")
                # todo: find better place to put this histograms
                tf.summary.scalar("leftMseLoss", self.leftMseLoss)
                tf.summary.scalar("rightMseLoss", self.rightMseLoss)

            with tf.name_scope("backprop_gradients_leftHemiGrads"):
                optimizer = tf.train.GradientDescentOptimizer(0.0001,name="leftHemiGrads_")
                self.optimizeLeftHemisphere = optimizer.minimize(self.leftMseLoss, name="leftHemiGrads")

            with tf.name_scope("backprop_gradients_rightHemiGrads"):
                optimizer = tf.train.GradientDescentOptimizer(0.0001,name="rightHemiGrads_")
                self.optimizeRightHemisphere = optimizer.minimize(self.rightMseLoss, name="rightHemiGrads")
        else:
            self.leftMseLoss = tf.losses.mean_squared_error(expectedReward,tf.stop_gradient(self.leftExpectedReward),
                                                            scope="leftHemiLoss")
            self.rightMseLoss = tf.losses.mean_squared_error(expectedReward,tf.stop_gradient(self.rightExpectedReward),
                                                             scope="rightHemiLoss")
            tf.summary.scalar("Traget_leftMseLoss", self.leftMseLoss)
            tf.summary.scalar("Target_rightMseLoss", self.rightMseLoss)
    def optimize(self,sess,expectedRewardAsGrndTruth, nextActionStateFeed,summary_writer, summaries):


        # todo: decide how to utilize metadata
        # Define options for the `sess.run()` call.
        options = tf.RunOptions()
        options.output_partition_graphs = True
        options.trace_level = tf.RunOptions.FULL_TRACE

        # Define a container for the returned metadata.
        metadata = tf.RunMetadata()

        sess.run(self.optimizeLeftHemisphere,{self.expectedRewardAsGrndTruth: expectedRewardAsGrndTruth,
                                             self.input[0]: nextActionStateFeed['action'],
                                             self.input[1]: nextActionStateFeed['state'] },options=options, run_metadata=metadata)

        # # Print the subgraphs that executed on each device.
        # print(metadata.partition_graphs)
        #
        # # Print the timings of each operation that executed.
        # print(metadata.step_stats)
        sess.run(self.optimizeRightHemisphere, {self.expectedRewardAsGrndTruth: expectedRewardAsGrndTruth,
                                               self.input[0]: nextActionStateFeed['action'],
                                               self.input[1]: nextActionStateFeed['state']})

        summaryOutput = sess.run(summaries, {self.expectedRewardAsGrndTruth: expectedRewardAsGrndTruth,
                                            self.input[0]: nextActionStateFeed['action'],
                                            self.input[1]: nextActionStateFeed['state']})
        summary_writer.add_summary(summaryOutput)

    def criticize(self,tfSession, runTimeInputs):

        leftHemisphereLogProbs,rightHemisphereLogProbs = tfSession.run([self.leftExpectedReward, self.rightExpectedReward], feed_dict={self.input[0]: runTimeInputs['action'],
                                                                               self.input[1]: runTimeInputs['state']})

        return leftHemisphereLogProbs,rightHemisphereLogProbs

    def softCopyWeightsToOtherCritic(self,sess, otherCritic):
        '''

        :param sess: an active tf.Session() instance
        :param otherCritic: object of class critic with same architechture
        :return: the values of all variables of calling critic, which are a result of tf.assign op evaluation
        '''
        # todo: create here some soft copying
        with tf.name_scope("softWeightsCopyTo_{}".format(otherCritic.nameScope)):

            # create N tf ops of assigning values. N is the quantity of trainable variables
            self.update_weights = [tf.assign(dstCriticVar, tf.subtract(tf.scalar_mul(1-self.tau, dstCriticVar),tf.scalar_mul(self.tau, srcCriticVar)))\
                              for (dstCriticVar, srcCriticVar) in
                              zip(tf.trainable_variables(otherCritic.nameScope), tf.trainable_variables(self.nameScope))]
            return sess.run(self.update_weights)

