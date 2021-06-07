import tensorflow as tf
from tf_agents.trajectories import TimeStep, Trajectory
from tf_agents.specs import BoundedTensorSpec, BoundedArraySpec, TensorSpec
from tensorflow.python.training.tracking.data_structures import _TupleWrapper

# specifications for python to tensorflow environments
ts_spec = TimeStep(
    step_type = TensorSpec(shape=(1,), dtype='int32', name='step_type'),
    reward = TensorSpec(shape=(1,), dtype='float32', name='reward'),
    discount = BoundedArraySpec(shape=(1,), dtype='float32', name='discount', minimum=np.array(0., dtype='float32'), maximum=np.array(1., dtype='float32')),
    observation = {'submodel_1_obs': TensorSpec(shape=(1, 100, 8, 1), dtype='float32'),
                  'submodel_2_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_3_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_4_obs': TensorSpec(shape=(1, 100, 5, 1), dtype='float32'),
                  'submodel_5_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_6_obs': TensorSpec(shape=(1, 1), dtype='float32'),
                  'submodel_7_obs': TensorSpec(shape=(1, 1), dtype='float32')},
    )

act_spec = BoundedTensorSpec(shape=(1,), dtype=tf.int32, name='action', minimum=np.array(0, dtype='int32'), maximum=np.array(1, dtype='int32'))

data_spec = _TupleWrapper(Trajectory(
    action = BoundedTensorSpec(shape=(1,), dtype=tf.int32, name='action', minimum=np.array(0, dtype='int32'), maximum=np.array(1, dtype='int32')),
    discount = BoundedTensorSpec(shape=(1,), dtype='float32', name='discount', minimum=np.array(0., dtype='float32'), maximum=np.array(1., dtype='float32')),
    next_step_type = TensorSpec(shape=(1,), dtype=tf.int32, name='step_type'),
    observation = {'submodel_1_obs': TensorSpec(shape=(1, 100, 8, 1), dtype='float32'),
                  'submodel_2_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_3_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_4_obs': TensorSpec(shape=(1, 100, 5, 1), dtype='float32'),
                  'submodel_5_obs': TensorSpec(shape=(1, 100, 3, 1), dtype='float32'),
                  'submodel_6_obs': TensorSpec(shape=(1, 1), dtype='float32'),
                  'submodel_7_obs': TensorSpec(shape=(1, 1), dtype='float32')},
    policy_info = (),
    reward = TensorSpec(shape=(1,), dtype='float32', name='reward'), 
    step_type = TensorSpec(shape=(1,), dtype='int32', name='step_type') 
    ))