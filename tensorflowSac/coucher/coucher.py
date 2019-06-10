
import os
import numpy as np
import tensorflow as tf
import itertools
import gym
from gym import spaces

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
            self.gamma=0.5
            self.target_update_interval = 50

    def update_parameters(self,sess, memory, batch_size, updates):

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = np.reshape(state_batch,newshape=(batch_size, observationSpaceShape))
        # todo: decide weather is needed to use stop gradient
        #with tf.stop_gradient() as stop:
        next_state_action, next_state_log_pi = self.actor.predict(sess, {'state': next_state_batch})

        qf1_next_target = self.targetCritic.criticize(sess, {'state': next_state_batch,'action': next_state_action})
        min_qf_next_target = qf1_next_target
        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        actBatch, log_pi = self.actor.predict(sess, {'state': state_batch})

        qf1_pi = self.trainingCritic.criticize(sess, {'state': state_batch,'action': actBatch})

        min_qf_pi = qf1_pi

        qf1_loss = self.trainingCritic.optimize(sess, next_q_value,{'state': next_state_batch,'action': next_state_action})
        policy_loss= self.actor.optimize(sess, min_qf_pi, {'state': next_state_batch})


        if updates % self.target_update_interval == 0:
            self.trainingCritic.softCopyWeightsToOtherCritic(sess, self.targetCritic)

        return qf1_loss, policy_loss

