import tensorflow as tf


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
        tensor: A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])
    return combined_shape


def get_weight(shape, trainable=True, name="weight", initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    w = tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=trainable
    )
    tf.add_to_collection("weights", w)
    return w


def get_bias(shape, trainable=True, name="bias", initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0)
    b = tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=trainable,
    )
    tf.add_to_collection("biases", b)
    return b


def conv2d(x, n_in, n_out, k, s, p="SAME", bias=False, scope="conv", trainable=True):
    with tf.variable_scope(scope):
        w = get_weight(shape=[k, k, n_in, n_out], trainable=trainable)
        conv = tf.nn.conv2d(x, w, [1, s, s, 1], padding=p)
        if bias:
            bias = get_bias(shape=[n_out], trainable=trainable)
            conv = tf.nn.bias_add(conv, bias)
    return conv


def fc(x, n_out, bias=True, scope="fc", trainable=True):
    shape = combined_static_and_dynamic_shape(x)
    if len(shape) == 4:
        size = shape[1] * shape[2] * shape[3]
    else:
        size = shape[-1]
    with tf.variable_scope(scope):
        w = get_weight(shape=[size, n_out], trainable=trainable)
        flat_x = tf.reshape(x, [-1, size])
        y = tf.matmul(flat_x, w)
        if bias:
            bias = get_bias(shape=[n_out], trainable=trainable)
            y = tf.nn.bias_add(y, bias)
    return y


def batch_norm(x, n_out, is_training, scope="bn"):
    y = tf.layers.batch_normalization(
        x, 
        training=is_training,
        name=scope,
    )
    return y


def res_block_low(x, n_in, n_out, subsample, is_training, scope="res_block"):
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_out, 3, 2, "SAME", False, scope="conv_1")
            shortcut = conv2d(x, n_in, n_out, 3, 2, "SAME", False, scope="shortcut")
        else:
            y = conv2d(x, n_in, n_out, 3, 1, "SAME", False, scope="conv_1")
            shortcut = tf.identity(x, name="shortcut")

        y = batch_norm(y, n_out, is_training, scope="bn_1")
        y = tf.nn.relu(y, name="relu_1")

        y = conv2d(y, n_out, n_out, 3, 1, "SAME", True, scope="conv_2")
        y = batch_norm(y, n_out, is_training, scope="bn_2")
        y = y + shortcut
        y = tf.nn.relu(y, name="relu_2")

    return y


def res_stage_low(x, n_in, n_out, n, first_subsample, is_training, scope="res_stage"):
    with tf.variable_scope(scope):
        y = res_block_low(x, n_in, n_out, first_subsample, is_training, scope="block_1")
        for i in range(n - 1):
            y = res_block_low(y, n_out, n_out, False, is_training, scope="block_%d"%(i+2))
    return y


def res_block_high(x, n_hide, n_out, subsample, is_training, first_stride=2, scope="res_block"):
    n_in = combined_static_and_dynamic_shape(x)[-1]
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_hide, 1, first_stride, "SAME", False, scope="conv_1")  
            shortcut = conv2d(x, n_in, n_out, 1, first_stride, "SAME", False, scope="shortcut")
        else:
            y = conv2d(x, n_in, n_hide, 1, 1, "SAME", False, scope="conv_1")
            shortcut = tf.identity(x, name="shortcut")

        y = batch_norm(y, n_hide, is_training, scope="bn_1")
        y = tf.nn.relu(y, name="relu_1")

        y = conv2d(y, n_hide, n_hide, 3, 1, "SAME", True, scope="conv_2")
        y = batch_norm(y, n_hide, is_training, scope="bn_2")
        y = tf.nn.relu(y, name="relu_2")

        y = conv2d(y, n_hide, n_out, 1, 1, "SAME", True, scope="conv_3")
        y = batch_norm(y, n_out, is_training, scope="bn_3")
        y = y + shortcut
        y = tf.nn.relu(y, name="relu_3")
    return y


def res_stage_high(x, n_hide, n_out, n, first_subsample, first_stride, is_training, scope="res_stage"):
    with tf.variable_scope(scope):
        y = res_block_high(x, n_hide, n_out, first_subsample, is_training, first_stride, scope="block_1")
        for i in range(n - 1):
            y = res_block_high(y, n_hide, n_out, False, is_training, 1, scope="block_%d"%(i+2))
    return y


def resnet18(x, n_classes, is_training, scope="resnet18"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 7, 2, "SAME", True, scope="conv_1")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_low(y,  64,  64, 2, False, is_training, scope="stage_2")        # --> (b, 56, 56, 64)
        y = res_stage_low(y,  64, 128, 2, True,  is_training, scope="stage_3")        # --> (b, 28, 28, 128)
        y = res_stage_low(y, 128, 256, 2, True,  is_training, scope="stage_4")        # --> (b, 14, 14, 256)
        y = res_stage_low(y, 256, 512, 2, True,  is_training, scope="stage_5")        # --> (b,  7,  7, 512)
        y = tf.layers.average_pooling2d(y, [7, 7], [1, 1], "VALID", name="avg_pool")  # --> (b, 1, 1, 512)
        y = tf.squeeze(y, [1, 2])  # --> (b, 512)
        y = fc(y, n_classes, scope="fc_final")  # --> (b, n_classes)
    return y


def resnet34(x, n_classes, is_training, scope="resnet34"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 7, 2, "SAME", True, scope="conv_1")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_low(y,  64,  64, 3, False, is_training, scope="stage_2")        # --> (b, 56, 56, 64)
        y = res_stage_low(y,  64, 128, 4, True,  is_training, scope="stage_3")        # --> (b, 28, 28, 128)
        y = res_stage_low(y, 128, 256, 6, True,  is_training, scope="stage_4")        # --> (b, 14, 14, 256)
        y = res_stage_low(y, 256, 512, 3, True,  is_training, scope="stage_5")        # --> (b,  7,  7, 512)

        y = tf.layers.average_pooling2d(y, [7, 7], [1, 1], "VALID", name="avg_pool")  # --> (b, 1, 1, 512)
        y = tf.squeeze(y, [1, 2])  # --> (b, 512)
        y = fc(y, n_classes, scope="fc_final")  # --> (b, n_classes)
    return y


def resnet50(x, n_classes, is_training, scope="resnet50"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 7, 2, "SAME", True, scope="conv_2")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_high(y, 64,  256,  3, True, 1, is_training, scope="stage_2")
        y = res_stage_high(y, 128, 512,  4, True, 2, is_training, scope="stage_3")
        y = res_stage_high(y, 256, 1024, 6, True, 2, is_training, scope="stage_4")
        y = res_stage_high(y, 512, 2048, 3, True, 2, is_training, scope="stage_5") 

        y = tf.layers.average_pooling2d(y, [7, 7], [1, 1], "VALID", name="avg_pool")
        y = tf.squeeze(y, [1, 2])
        y = fc(y, n_classes, scope="fc_final")
    return y


def resnet101(x, n_classes, is_training, scope="resnet101"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 7, 2, "SAME", True, scope="conv_2")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_high(y, 64,  256,  3,  True, 1, is_training, scope="stage_2")
        y = res_stage_high(y, 128, 512,  4,  True, 2, is_training, scope="stage_3")
        y = res_stage_high(y, 256, 1024, 23, True, 2, is_training, scope="stage_4")
        y = res_stage_high(y, 512, 2048, 3,  True, 2, is_training, scope="stage_5") 

        y = tf.layers.average_pooling2d(y, [7, 7], [1, 1], "VALID", name="avg_pool")
        y = tf.squeeze(y, [1, 2])
        y = fc(y, n_classes, scope="fc_final")
    return y


def resnet152(x, n_classes, is_training, scope="resnet152"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 7, 2, "SAME", True, scope="conv_2")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_high(y, 64,  256,  3,  True, 1, is_training, scope="stage_2")
        y = res_stage_high(y, 128, 512,  8,  True, 2, is_training, scope="stage_3")
        y = res_stage_high(y, 256, 1024, 36, True, 2, is_training, scope="stage_4")
        y = res_stage_high(y, 512, 2048, 3,  True, 2, is_training, scope="stage_5") 

        y = tf.layers.average_pooling2d(y, [7, 7], [1, 1], "VALID", name="avg_pool")
        y = tf.squeeze(y, [1, 2])
        y = fc(y, n_classes, scope="fc_final")
    return y


def resnet50_cifar10(x, n_classes, is_training, scope="resnet50_cifar10"):
    with tf.variable_scope(scope):
        with tf.variable_scope("stage_1"):
            y = conv2d(x, 3, 64, 3, 1, "SAME", True, scope="conv_2")
            y = tf.layers.max_pooling2d(y, [3, 3], [2, 2], "SAME", name="max_pool_1")
            y = tf.nn.relu(y, name="relu_1")

        y = res_stage_high(y, 64,  256,  3,  True, 2, is_training, scope="stage_2")
        y = res_stage_high(y, 128, 512,  4,  True, 2, is_training, scope="stage_3")
        y = res_stage_high(y, 256, 1024, 6, True, 2, is_training, scope="stage_4")
        y = res_stage_high(y, 512, 2048, 3,  True, 2, is_training, scope="stage_5") 

        y = conv2d(y, 2048, n_classes, 1, 1, "SAME", True, scope="conv_classifier")
        y = tf.squeeze(y, [1, 2])
    return y
