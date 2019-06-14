import os
import sys
import numpy as np
import tensorflow as tf
import gym
import itertools
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
## custom modules ##
from tensorflowSac.actor.actor import actor
from tensorflowSac.critic.critic import critic
from tensorflowSac.coucher.coucher import coucher
from tensorflowSac.model_config import observationSpaceShape, actionSpaceShape, batchSize, memory

env = gym.make("CartPole-v1")
observation_space = observationSpaceShape
action_space = actionSpaceShape


def createActorTfGraph(name):
    criticValueAsGrndTruthPlaceHolder = tf.placeholder(tf.float32,shape=(batchSize,1), name='criticValueAsGrndTruth')
    statePlaceholder = tf.placeholder(tf.float32, shape=(batchSize, observation_space), name='obsInput')

    with tf.name_scope(name):
        actor1 = actor(name)
        actor1.constructPredictor(statePlaceholder, criticValueAsGrndTruthPlaceHolder)
    return actor1

def createCriticTfGraph(name):
    expectedRewardAsGrndTruth = tf.placeholder(tf.float32,shape=(batchSize,1), name='expectedRewardAsGrndTruth')
    actionPlaceholder = tf.placeholder(tf.float32, shape=(batchSize, action_space), name='actionInput')
    statePlaceholder = tf.placeholder(tf.float32, shape=(batchSize, observation_space), name='obsInput')

    critic1 = critic(name)
    #todo: make construct private later when debug is finished
    critic1.construct([actionPlaceholder, statePlaceholder], expectedRewardAsGrndTruth)
    return critic1


argSettings = {
'start_steps': 100,
'batch_size': 9,
'updates_per_step': 1,
    'num_steps': 1000000,
    'eval': False
}
class arguments:
    def __init__(self,settings):
        self.start_steps = settings['start_steps']
        self.batch_size = settings['batch_size']
        self.updates_per_step = settings['updates_per_step']
        self.num_steps = settings['num_steps']
        self.eval = settings['eval']

args = arguments(argSettings)

with tf.Session() as sess:
    actor = createActorTfGraph('actor')
    trainingCritic = createCriticTfGraph('trainingCritic')
    targetCritic = createCriticTfGraph('targetCritic')
    couch = coucher(actor,trainingCritic,targetCritic)
    sess.run(tf.global_variables_initializer())
    trainingCritic.softCopyWeightsToOtherCritic(sess, targetCritic)

    trainCriticSummary = tf.summary.merge_all(scope=trainingCritic.nameScope)
    targetCriticSummary = tf.summary.merge_all(scope=targetCritic.nameScope)

    day = datetime.today()
    currDir = os.getcwd()
    createDir = currDir+'/experiment_day-{}'.format(day)
    os.mkdir(path=createDir)
    summary_writer = tf.summary.FileWriter(createDir, graph=tf.get_default_graph())
    actorSummary = tf.summary.merge_all(scope=actor.nameScope)
    saver = tf.train.Saver()

    # Training Loop
    total_numsteps = 0
    updates = 0
    for i_episode in itertools.count(1):
        if total_numsteps > args.num_steps:
            break

        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            env.render()
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                state = np.reshape(state, newshape=(1, observationSpaceShape))
                action, _ = actor.predict(sess, {'state': state})# Sample action from policy
                action = np.int(action)
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    couch.update_parameters(sess,memory, args.batch_size, updates,summary_writer ,
                                                trainCriticSummary,targetCriticSummary ,actorSummary)
                    updates += 1


            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward


            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state

        if total_numsteps > args.num_steps or episode_reward > 200:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(\
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    path = saver.save(sess, createDir+'/modelWeights')
    print("model is save in: {}".format(path))
env.close()
