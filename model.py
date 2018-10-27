import tensorflow as tf
from utils import layer_tools


flags = tf.app.flags
flags.DEFINE_integer("n_classes", 10, "")
flags.DEFINE_float("weight_decay", 1e-4, "")
FLAGS = flags.FLAGS


def get_loss(logits, labels):
    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=10), 
            logits=logits, scope="cross_entropy_loss")

    weight_l2_losses = [tf.nn.l2_loss(x) for x in tf.get_collection("weights")]
    weight_decay_loss = tf.multiply(FLAGS.weight_decay, tf.add_n(weight_l2_losses), name="weight_decay_loss")
    tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay_loss)

    losses_list = tf.get_collection(tf.GraphKeys.LOSSES)
    tf.logging.info("Losses Tensor: {}".format(losses_list))
    for var in losses_list:
        tf.summary.scalar("losses/" + var.op.name, var)
    total_loss  = tf.add_n(losses_list)
    return total_loss


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    images = features["image"]
    logits = layer_tools.resnet50_cifar10(images, FLAGS.n_classes, is_training)
    predictions = {"pred_classes": tf.argmax(input=logits, axis=1, name="pred_classes")}

    # PREDICT Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss_tensor = get_loss(logits, labels)
    accuracy, update_op = tf.metrics.accuracy(labels=labels, 
            predictions=predictions["pred_classes"], name="accuracy")

    # For Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # optimizer = tf.train.AdamOptimizer()
        boundaries = [200000, 400000]
        learning_rates = [1.0, 0.1, 0.01]
        learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), 
                boundaries=boundaries, values=learning_rates)
        tf.summary.scalar("lr", learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_tensor, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_tensor, train_op=train_op)

    # For EVAL mode
    eval_metric_ops = {"accuracy": (accuracy, update_op)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_tensor, eval_metric_ops=eval_metric_ops)

