import tensorflow as tf
from model import model_fn
from input_data import train_input_fn, eval_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags
flags.DEFINE_string("model_dir", "experiments/hello_high_api_no_pool_1", "")
FLAGS = flags.FLAGS


# train_distribute_strategy = tf.contrib.distribute.MirroredStrategy(
    # devices=None,
    # num_gpus=2,
    # num_gpus_per_worker=None,
    # cross_tower_ops=None,
    # prefetch_on_device=None
# )

run_config = tf.estimator.RunConfig(
    # train_distribute=train_distribute_strategy
)

cifar10_classifier_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=run_config,
    params=None,
)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                    max_steps=None,
                                    hooks=None)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                  steps=None,
                                  name="cifar10_test",
                                  hooks=None,
                                  exporters=None,
                                  start_delay_secs=120,
                                  throttle_secs=300)

tf.estimator.train_and_evaluate(cifar10_classifier_estimator, 
                                train_spec, 
                                eval_spec)

