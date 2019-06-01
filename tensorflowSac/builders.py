
from abc import ABC
import tensorflow as tf
from tensorflow_probability import distributions as distLib


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
        b_fc = tf.Variable(tf.constant(0.0, shape=self.settings['biasShape']), name='bias')
        head = tf.matmul(W_fc, tf.transpose(inputLayer))

        if self.settings['nonLinearity'] is not None:
            head = self.settings['nonLinearity'](tf.add(tf.transpose(head), b_fc))

        self.trackNode(nameScope +'_head', head)
        return head
