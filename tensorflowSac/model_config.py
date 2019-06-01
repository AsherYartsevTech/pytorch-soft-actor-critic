import tensorflow as tf
from tensorflowSac.builders import fullyConnectedLayerBuilder



policyArchSettings = {
    'mean_L1':
        {
            'name': 'Layer1',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [100, 200],
                'biasShape': [100],
                'stddev': 0.03,
                'nonLinearity': tf.nn.relu
            }
        },
    'mean_L2':
        {
            'name': 'Layer2',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [50, 100],
                'biasShape': [50],
                'stddev': 0.03,
                'nonLinearity': tf.nn.tanh
            }
        },
    'mean_L3':
        {
            'name': 'Layer3',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [10, 50],
                'biasShape': [10],
                'stddev': 0.03,
                'nonLinearity': tf.nn.tanh
            }
        },
    # todo: make head to produce a vector of means and vector of log_std - of the size of action_space.dim
    'mean_head':
        {
            'name': 'head',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [2, 10],
                'biasShape': [2],
                'stddev': 0.03,
                'nonLinearity': None
            }
        },
    'log_std_L1':
        {
            'name': 'Layer1',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [100, 200],
                'biasShape': [100],
                'stddev': 0.03,
                'nonLinearity': tf.nn.relu
            }
        },
    'log_std_L2':
        {
            'name': 'Layer2',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [50, 100],
                'biasShape': [50],
                'stddev': 0.03,
                'nonLinearity': tf.nn.tanh
            }
        },
    'log_std_L3':
        {
            'name': 'Layer3',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [10, 50],
                'biasShape': [10],
                'stddev': 0.03,
                'nonLinearity': tf.nn.tanh
            }
        },
    # todo: make head to produce a vector of means and vector of log_std - of the size of action_space.dim
    'log_std_head':
        {
            'name': 'head',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [2, 10],
                'biasShape': [2],
                'stddev': 0.03,
                'nonLinearity': None
            }
        }
}


criticArchSettings = {
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
                    'nonLinearity': tf.nn.relu
                    }
                },
    'L3' :      {'name': 'Layer3',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [10, 50],
                    'biasShape': [10],
                    'stddev': 0.03,
                    'nonLinearity': tf.nn.relu
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