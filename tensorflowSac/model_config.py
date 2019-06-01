import tensorflow as tf
from tensorflowSac.builders import fullyConnectedLayerBuilder



actorArchSettings = {
    'L1' :      {'name': 'Layer1',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [100, 200],
                    'biasShape': [100],
                    'stddev': 0.03,
                    'nonLinearity': tf.nn.relu
                    }
                },
    'L2' :      {'name': 'Layer2',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [50, 100],
                    'biasShape': [50],
                    'stddev': 0.03,
                    'nonLinearity': tf.nn.tanh
                    }
                },
    'L3' :      {'name': 'Layer3',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [10, 50],
                    'biasShape': [10],
                    'stddev': 0.03,
                    'nonLinearity': tf.nn.tanh
                    }
                },
    'head' :    {'name': 'head',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [3, 10],
                    'biasShape': [3],
                    'stddev': 0.03,
                    'nonLinearity': None
                    }
                }
}