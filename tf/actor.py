from tf_utils import weight_variable, bias_variable
import tensorflow as tf

actorTfNodes=dict()

class actor(object):
    '''
    policy object is an Actor who absorbs the state of environment
    and outputs action data.
    '''
    def __init__(self, stateInput = (1200,)):
        def constructNN():
            def constructBody():
                with tf.name_scope("actor"):
                    inputPlaceholder = tf.placeholder(tf.float32, (None,) + stateInput, "batchStateInput")
                    global actorTfNodes
                    actorTfNodes["NN_feeds"] = {'batchStateInput': inputPlaceholder}
                    # todo replace sanity demo with real code
                    W_fc1 = weight_variable([784, 1200], 0.03)
                    b_fc1 = bias_variable([784])
                    opMatMul1=tf.matmul(W_fc1, tf.transpose(inputPlaceholder))
                    h_fc1 = tf.nn.relu(tf.add(tf.transpose(opMatMul1), b_fc1))

                    return h_fc1
            def constructOptimizer(bodyHead):
                pass

            bodyHead = constructBody()
            # optimizer = constructOptimizer(bodyHead)
            return bodyHead
        self.bodyHead = constructNN()
    def debug_getBodyHead(self):
        return self.bodyHead

    def getBestActionFromHere(self,state):

        pass

    def learnFromExperience(self,historyOfEvents):

        pass
