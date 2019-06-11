import tensorflow as tf
from tensorflowSac.builders import fullyConnectedLayerBuilder, ReplayMemoryBuilder

# training hyperparameters

batchSize = None


#   parameters which will determine the shape of Spaces in a gym.env object
actionSpaceShape = 1
observationSpaceShape = 4


# Memory
#todo replace 10000 with parameter
memory = ReplayMemoryBuilder(1000000)

#todo: revise 'stddev' init value for weight matrices
#todo: revise 'nonLinearity' put in each layer

policyArchSettings = {
'meanNN':
    {
    'L1':
        {
            'name': 'Layer1',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [100, observationSpaceShape],
                'biasShape': [100],
                'stddev': 0.03,
                'nonLinearity': tf.nn.tanh
            }
        },
    'L2':
        {
            'name': 'Layer2',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [50, 100],
                'biasShape': [50],
                'stddev': 0.1,
                'nonLinearity': tf.nn.tanh
            }
        },
    'L3':
        {
            'name': 'Layer3',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [10, 50],
                'biasShape': [10],
                'stddev': 0.14142,
                'nonLinearity': tf.nn.tanh
            }
        },
    'head':
        {
            'name': 'head',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [actionSpaceShape, 10],
                'biasShape': [actionSpaceShape],
                'stddev': 0.316,
                'nonLinearity': None
            }
        }
    },
'logStdNN':
    {
    'L1':
        {
            'name': 'Layer1',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [100, observationSpaceShape],
                'biasShape': [100],
                'stddev': 0.03,
                'nonLinearity': tf.nn.relu
            }
        },
    'L2':
        {
            'name': 'Layer2',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [50, 100],
                'biasShape': [50],
                'stddev': 0.1,
                'nonLinearity': tf.nn.relu
            }
        },
    'L3':
        {
            'name': 'Layer3',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [10, 50],
                'biasShape': [10],
                'stddev': 0.14142,
                'nonLinearity': tf.nn.relu
            }
        },
    'head':
        {
            'name': 'head',
            'builder': fullyConnectedLayerBuilder,
            'builder_params':
            {
                'weightMatrixShape': [actionSpaceShape, 10],
                'biasShape': [actionSpaceShape],
                'stddev': 0.316,
                'nonLinearity': None
            }
        }
    }
}


criticArchSettings = {
    'L1':      {'name': 'Layer1',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [100, actionSpaceShape + observationSpaceShape],
                    'biasShape': [100],
                    'stddev': 0.03,
                    'nonLinearity': tf.nn.relu
                    }
                },
    'L2':      {'name': 'Layer2',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [50, 100],
                    'biasShape': [50],
                    'stddev': 0.1,
                    'nonLinearity': tf.nn.relu
                    }
                },
    'L3':      {'name': 'Layer3',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [10, 50],
                    'biasShape': [10],
                    'stddev': 0.14142,
                    'nonLinearity': tf.nn.relu
                    }
                },
    #'head' produces logProbs of action|state
    'head':    {'name': 'head',
                'builder': fullyConnectedLayerBuilder,
                'builder_params': {
                    'weightMatrixShape': [1, 10],
                    'biasShape': [1],
                    'stddev': 0.316,
                    'nonLinearity': None
                    }
                }
}