from util_tf import tf, placeholder,normalize
from util_np import np
### todo:
# add batch normalization
# check out loss functions

def wgan(data, z, data_dim, z_dim, dense_dim, noise, clip_limit):

    def generator(x, dense_dim, data_dim):
        for i,d in enumerate(dense_dim):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        x = tf.layers.dense(x, data_dim, use_bias=False)
        return x
        #return tf.clip_by_value(x, 0.0, 1.0)

    def discriminator(x, dense_dim, reuse=False):
        for i,d in enumerate(reversed(dense_dim)):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        logits = tf.layers.dense(x, 1, use_bias=False)
        return tf.squeeze(logits, axis=1)

    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, data_dim], data[0], "Input")
        y_inpt = placeholder(tf.float32, [None, 1], data[1], "Input_label")
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
            d_loss = tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)
        with tf.variable_scope("g_loss"):
            g_loss = tf.reduce_mean(y_fake)

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("clipping"):
        d_clip = [v.assign(tf.clip_by_value(v, -clip_limit, clip_limit)) for v in d_vars]

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

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
                d_clip=d_clip,
                z_inpt=z_inpt,
                y_real=y_real)
