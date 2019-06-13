
import os
import numpy as np
import tensorflow as tf

import sys
np.set_printoptions(threshold=sys.maxsize)

## custom modules ##
from tensorflowSac.model_config import observationSpaceShape, actionSpaceShape, batchSize, memory





class coucher:
    def __init__(self, actor, trainingCritic,targetCritic):
        with tf.name_scope('couch'):
            self.actor = actor
            self.trainingCritic = trainingCritic
            self.targetCritic = targetCritic

            self.state_batch =      tf.placeholder(tf.float32,(batchSize,)+(observationSpaceShape,),name='state_batch')
            self.next_state_batch = tf.placeholder(tf.float32,(batchSize,)+(observationSpaceShape,),name='next_state_batch')
            self.action_batch =     tf.placeholder(tf.float32,(batchSize,)+(actionSpaceShape,),name='action_batch')
            self.reward_batch =     tf.placeholder(tf.float32,(batchSize,1),name='reward_batch')
            self.mask_batch =       tf.placeholder(tf.float32,(batchSize,1),name='mask_batch')
            self.gamma=0.99
            self.target_update_interval = 1

    def update_parameters(self,sess, memory, batch_size, updates):

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = np.reshape(state_batch,newshape=(batch_size, observationSpaceShape))
        nextStateOfAction, nextStateLogProbOfAction = self.actor.predict(sess, {'state': next_state_batch})

        # start all equation (5) from paper
        leftHemisphereLogNextStateValues, rightHemisphereLogNextStateValues = self.targetCritic.criticize(sess, {'state': next_state_batch,'action': nextStateOfAction})
        minLogNextStateValues = tf.minimum(leftHemisphereLogNextStateValues, rightHemisphereLogNextStateValues).eval()

        expectedValueForNextState = minLogNextStateValues - self.actor.alpha * nextStateLogProbOfAction
        expectedRewardAsGrndTruth = reward_batch + mask_batch * self.gamma * (expectedValueForNextState)


        leftHemisphereLoss, rightHemisphereLoss = self.trainingCritic.optimize(sess,grndTruth=expectedRewardAsGrndTruth,
                                                    nextActionStateFeed={'state': state_batch, 'action': action_batch})
        # end all equation (5) from paper

        actBatch, log_pi = self.actor.predict(sess, {'state': state_batch})
        # Training critic expresses opinion by giving it's own logarithmic probability for the actions the policy suggests
        leftHemisphereLogCurrStateValues, rightHemisphereLogCurrStateValues = self.trainingCritic.criticize(sess, {'state': state_batch, 'action': actBatch})
        minLogCurrStateValues = tf.minimum(leftHemisphereLogCurrStateValues, rightHemisphereLogCurrStateValues).eval()

        # the process self.actor.alpha * log_pi - min_qf_pi goes inside this optimize
        policy_loss = self.actor.optimize(sess, trainingCriticOpinionOnPolicyChoices=minLogCurrStateValues, nextState={'state': next_state_batch})

        # print('trainingCriticLoss:{criticLoss}, policyLoss:{policyLoss}'.format(criticLoss=qf1_loss, policyLoss=policy_loss))

        if updates % self.target_update_interval == 0:
            self.trainingCritic.softCopyWeightsToOtherCritic(sess, self.targetCritic)

        return leftHemisphereLoss, rightHemisphereLoss, policy_loss

