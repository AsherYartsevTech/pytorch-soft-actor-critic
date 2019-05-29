
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class actorBasedNN(ABC):
    def __init__(self):
        super().__init__()

    # @abstractmethod
    # def verbose(self):
    #     pass

class TfGraphTracker(actorBasedNN):
    def __init__(self):
        self.actorTfNodes=dict()
    def track(self, graphNode,graphNodeName):
        # todo raise on non-str name?
        self.actorTfNodes[graphNodeName]=graphNode


class fullyConnectedLayerBuilder(actorBasedNN):
    '''
    input:
    fullyConnectedSettings, dict with labels:
        -'weightMatrixShape'
        -'biasShape'
        -'stddev'
    '''
    def __init__(self, fullyConnectedSettings):
        self.settings = fullyConnectedSettings

    def construct(self,inputLayer,nameScope):
        with tf.name_scope(nameScope):
            W_fc = tf.Variable(tf.truncated_normal(self.settings['weightMatrixShape'],
                                       stddev=self.settings['stddev']))
            b_fc = tf.Variable(tf.constant(0.0, shape=self.settings['biasShape']))
            opMatMul = tf.matmul(W_fc, tf.transpose(inputLayer))
            nonLinearity = tf.nn.relu(tf.add(tf.transpose(opMatMul), b_fc))

            return nonLinearity

class actorNNBuilder(actorBasedNN):
    def __init__(self):
        self.graphTracker = TfGraphTracker()

    def construct(self,inputPlaceholders):
        fc_l1Config = {'weightMatrixShape': [100, 200], 'biasShape': [100], 'stddev': 0.03}
        fc_l1 = fullyConnectedLayerBuilder(fc_l1Config).construct(inputPlaceholders, 'Layer1')
        self.graphTracker.track(fc_l1, 'Layer1')
        return self.graphTracker, fc_l1




params = np.ones([2,200])
inputPlaceholder = tf.placeholder(tf.float32, shape=(None,200), name='inputToNN')
graphTracker, actorNNTop = actorNNBuilder().construct(inputPlaceholder)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    fcn = sess.run([actorNNTop], feed_dict={inputPlaceholder: params})
    print(fcn)

