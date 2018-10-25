import tensorflow as tf
from model import model_fn
from input_data import eval_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags
flags.DEFINE_string("model_dir", "experiments/demo", "")
FLAGS = flags.FLAGS


cifar10_classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir
)
cifar10_classifier.evaluate(
    input_fn=eval_input_fn,        
)
