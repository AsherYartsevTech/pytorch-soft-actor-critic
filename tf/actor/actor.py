



class actorBasedNN(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def verbose(self):
        pass

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
        with tf.name_scope("actor"):
            W_fc = tf.Variable(tf.truncated_normal(self.settings['weightMatrixShape'],
                                       stddev=self.settings['stddev']))
            b_fc = tf.Variable(tf.constant(0.0, shape=self.settings['biasShape']))
            opMatMul = tf.matmul(W_fc, tf.transpose(inputLayer))
            nonLinearity = tf.nn.relu(tf.add(tf.transpose(opMatMul), b_fc))

            return nonLinearity
