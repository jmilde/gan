from util_tf import tf, placeholder,normalize
from util_np import np


def gan(data, z, data_dim, z_dim, dense_dim, noise):

    def generator(x, dense_dim, data_dim):
        for i,d in enumerate(dense_dim):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        x = tf.layers.dense(x, data_dim, use_bias=False)
        return tf.clip_by_value(x, 0.0, 1.0)

    def discriminator(x, dense_dim):
        for i,d in enumerate(reversed(dense_dim)):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        logits = tf.layers.dense(x, 1, use_bias=False)
        return tf.squeeze(logits, axis=1)

    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, data_dim], data[0], "Input")
        z_inpt = placeholder(tf.float32, [None, z_dim], z, "Noise")

    with tf.variable_scope("generator"):
        x_fake = generator(z_inpt, dense_dim, data_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("discriminator"):
        y_real = discriminator(inpt, dense_dim)
    with tf.variable_scope("discriminator",reuse=True):
        y_fake = discriminator(x_fake, dense_dim)

    with tf.variable_scope("loss"):
        with tf.variable_scope("d_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_real)*0.9, logits=y_real))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_fake), logits=y_fake))
            d_loss = d_loss_real + d_loss_fake

        with tf.variable_scope("g_loss"):
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_fake), logits=y_fake))


    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


    return dict(step=step,
                inpt=inpt,
                d_loss=d_loss,
                g_loss=g_loss,
                d_step=d_step,
                g_step=g_step,
                x_fake=x_fake,
                z_inpt=z_inpt,
                y_real=y_real)
