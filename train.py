import tensorflow as tf
from model import model_fn
from input_data import train_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags
flags.DEFINE_string("model_dir", "experiments/demo", "")
FLAGS = flags.FLAGS


train_config = tf.estimator.RunConfig(

)

cifar10_classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir
)


def main(_):
    cifar10_classifier.train(
        input_fn=train_input_fn,
    )


if __name__ == "__main__":
    tf.app.run()