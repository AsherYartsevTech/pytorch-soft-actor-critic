import os
import numpy as np
import tensorflow as tf
import itertools
import gym
from gym import spaces

## custom modules ##
from tensorflowSac.actor.actor import actor
from tensorflowSac.critic.critic import critic
from tensorflowSac.coucher.coucher import coucher
from tensorflowSac.model_config import observationSpaceShape, actionSpaceShape, batchSize, memory

env = gym.make("CartPole-v1")
observation_space = observationSpaceShape
action_space = actionSpaceShape


'''
constract all SAC algorithm left update functions

'''

def createActorTfGraph(name):
    grndTruthPlaceHolder = tf.placeholder(tf.float32,shape=(batchSize,1), name='gtruthForActorUpdate')
    statePlaceholder = tf.placeholder(tf.float32, shape=(batchSize, observation_space), name='obsInput')

    actor1 = actor(name)
    actor1.constructPredictor(statePlaceholder)
    actor1.constructOptimizer(grndTruthPlaceHolder)
    return actor1

def createCriticTfGraph(name):
    grndTruthPlaceHolder = tf.placeholder(tf.float32,shape=(batchSize,1), name='outputOfTargetCriticAndActorProb')
    actionPlaceholder = tf.placeholder(tf.float32, shape=(batchSize, action_space), name='actionInput')
    statePlaceholder = tf.placeholder(tf.float32, shape=(batchSize, observation_space), name='obsInput')
    critic1 = critic(name)
    #todo: make construct private later when debug is finished
    critic1.construct([actionPlaceholder, statePlaceholder])
    critic1.constructOptimizer(grndTruthPlaceHolder)
    return critic1


argSettings = {
'start_steps': 10,
'batch_size': 42,
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
    summary_writer = tf.summary.FileWriter(os.getcwd(), graph=tf.get_default_graph())
    # critic_1_loss,policy_loss,episode_reward = 0
    # tf.summary.scalar('loss/critic_1', [critic_1_loss])
    # tf.summary.scalar('loss/policy', [policy_loss])
    # tf.summary.scalar('reward/train', [episode_reward])

    summaries = tf.summary.merge_all()

    # Training Loop
    total_numsteps = 0
    updates = 0
    for i_episode in itertools.count(1):
        # asher todo: temporal patch
        if i_episode > 100:
            break
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            # env.render()
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                state = np.reshape(state, newshape=(1, observationSpaceShape))
                action, _ = actor.predict(sess, {'state': state})# Sample action from policy
                # action = np.reshape(action, newshape=(actionSpaceShape,))
                action = np.int(action)
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, policy_loss = couch.update_parameters(sess,memory, args.batch_size, updates)
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

        if total_numsteps > args.num_steps:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        #
        # if i_episode % 10 == 0 and args.eval == True:
        #     avg_reward = 0.
        #     episodes = 10
        #     for _  in range(episodes):
        #         state = env.reset()
        #         episode_reward = 0
        #         done = False
        #         while not done:
        #             action = agent.select_action(state, eval=True)
        #
        #             next_state, reward, done, _ = env.step(action)
        #             episode_reward += reward
        #
        #
        #             state = next_state
        #         avg_reward += episode_reward
        #     avg_reward /= episodes
        #
        #
        #     writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        #
        #     print("----------------------------------------")
        #     print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        #     print("----------------------------------------")

env.close()