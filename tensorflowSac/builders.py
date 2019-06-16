
from abc import ABC
import tensorflow as tf
from tensorflow_probability import distributions as distLib
import random
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def tfNameScoping(method):

    def methodWithTfNameScope(layerBuilderInstance, inputLayer, nameScope):
        with tf.name_scope(nameScope):
            return method(layerBuilderInstance, inputLayer, nameScope)

    return methodWithTfNameScope

class TfGraphTracker(object):
    def __init__(self):
        self.actorTfNodes = dict()
    def track(self, graphNodeName, graphNode):
        # todo raise on non-str name?
        self.actorTfNodes[graphNodeName] = graphNode

class tfNeauralStructure(ABC):

    #ensuring there is a single tracker for all tensorflowSac nodes created in this module
    graphTracker = TfGraphTracker()
    def __init__(self):
        super().__init__()

    def trackNode(self, graphNodeName, graphNode):
        tfNeauralStructure.graphTracker.track(graphNodeName, graphNode)

class fullyConnectedLayerBuilder(tfNeauralStructure):
    '''
    input:
    fullyConnectedSettings, dict with labels:
        -'weightMatrixShape'
        -'biasShape'
        -'stddev'
        -'nonLinearity'
    '''
    def __init__(self, fullyConnectedSettings):
        super().__init__()
        self.settings = fullyConnectedSettings

    @tfNameScoping
    def construct(self,inputLayer, nameScope):
        '''
        :param inputLayer: to which layer attach current construction.
        :param nameScope:  parameter for tfNameScoping decorator
        :return: top of tensorflowSac operations and Variables serialization
        '''
        W_fc = tf.Variable(tf.truncated_normal(self.settings['weightMatrixShape'],
                                   stddev=self.settings['stddev']), name='weightMatrix')
        tf.summary.histogram("weightMatrix", W_fc)

        b_fc = tf.Variable(tf.constant(0.1, shape=self.settings['biasShape']), name='bias')
        tf.summary.histogram("bias", b_fc)

        if W_fc.dtype != inputLayer.dtype:
            inputLayer = tf.cast(inputLayer, dtype=W_fc.dtype)

        head = tf.matmul(W_fc, tf.transpose(inputLayer))
        head = tf.add(tf.transpose(head), b_fc)
        tf.summary.histogram("weightAndBiasedInput", head)

        if self.settings['nonLinearity'] is not None:
            head = self.settings['nonLinearity'](head)
            tf.summary.histogram("activations", head)

        self.trackNode(nameScope +'_head', head)
        return head



class ReplayMemoryBuilder:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        startIndex = random.randint(0, len(self.buffer)-batch_size)

        # batch = random.sample(self.buffer, batch_size)
        batch = self.buffer[startIndex: startIndex+batch_size]
        state, action, reward, next_state, done = map(np.ma.row_stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


