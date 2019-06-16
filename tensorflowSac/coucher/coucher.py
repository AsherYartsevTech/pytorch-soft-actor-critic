
import os
import numpy as np
import tensorflow as tf

import sys
np.set_printoptions(threshold=sys.maxsize)

## custom modules ##
from tensorflowSac.model_config import observationSpaceShape, actionSpaceShape, batchSize, memory





class coucher:
    def __init__(self, actor, trainingCritic,targetCritic):
        self.actor = actor
        self.trainingCritic = trainingCritic
        self.targetCritic = targetCritic

        self.gamma=0.99
        self.target_update_interval = 100

    def update_parameters(self,sess, memory, batch_size, updates, summary_writer, trainCriticSummary,targetCriticSummary ,actorSummary):

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = np.reshape(state_batch,newshape=(batch_size, observationSpaceShape))
        nextStateOfAction, nextStateLogProbOfAction = self.actor.predict(sess, {'state': next_state_batch})

        # start all equation (5) from paper
        leftHemisphereLogNextStateValues, rightHemisphereLogNextStateValues = self.targetCritic.criticize(sess, {'state': next_state_batch,'action': nextStateOfAction})
        minLogNextStateValues = tf.minimum(leftHemisphereLogNextStateValues, rightHemisphereLogNextStateValues).eval()

        expectedValueForNextState = minLogNextStateValues - self.actor.alpha * nextStateLogProbOfAction
        expectedRewardAsGrndTruth = reward_batch + mask_batch * self.gamma * (expectedValueForNextState)

        expectedRewardAsGrndTruth = expectedRewardAsGrndTruth.eval()
        self.trainingCritic.optimize(sess,expectedRewardAsGrndTruth,{'state': state_batch, 'action': action_batch},summary_writer, trainCriticSummary)
        # end all equation (5) from paper

        actBatch, log_pi = self.actor.predict(sess, {'state': state_batch})
        # Training critic expresses opinion by giving it's own logarithmic probability for the actions the policy suggests
        leftHemisphereLogCurrStateValues, rightHemisphereLogCurrStateValues = self.trainingCritic.criticize(sess, {'state': state_batch, 'action': actBatch})
        minLogCurrStateValues = tf.minimum(leftHemisphereLogCurrStateValues, rightHemisphereLogCurrStateValues).eval()


        # the process self.actor.alpha * log_pi - min_qf_pi goes inside this optimize
        self.actor.optimize(sess, minLogCurrStateValues,{'state': next_state_batch},summary_writer, actorSummary)


        if updates % self.target_update_interval == 0:
            self.trainingCritic.softCopyWeightsToOtherCritic(sess, self.targetCritic)



