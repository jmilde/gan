from util_tf import tf, placeholder, normalize
from util_np import np


def gan(data, btlnk_dim, data_dim, z_dim, dense_dim):

    def generator(x, cond, btlnk_dim, dense_dim, data_dim):
        x = tf.layers.dense(x, data_dim, use_bias=False)
        cond = tf.layers.dense(cond, data_dim, use_bias=False)
        x = normalize(tf.nn.leaky_relu(x+cond), "layer_norm_1")
        #for i,d in enumerate(dense_dim[1:]):
            #with tf.variable_scope("dense_layer{}".format(i)):
                #x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        # bottleneck
        x = normalize(tf.layers.dense(x, btlnk_dim, use_bias=False, activation=tf.nn.leaky_relu), "layer_norm_2")
        # output
        logits = tf.layers.dense(x, data_dim, use_bias=False)
        return tf.nn.sigmoid(logits)
        #return tf.clip_by_value(x, 0.0, 1.0)


    def discriminator(x, cond, btlnk_dim, dense_dim, data_dim):
        x = tf.layers.dense(x, data_dim, use_bias=False)
        cond = tf.layers.dense(cond, data_dim, use_bias=False)
        x = normalize(tf.nn.leaky_relu(x+cond), "layer_norm_1")
        #for i,d in enumerate(reversed(dense_dim)):
            #with tf.variable_scope("dense_layer{}".format(i)):
                #x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        # bottleneck
        x = normalize(tf.layers.dense(x, btlnk_dim, use_bias=False, activation=tf.nn.leaky_relu),"layer_norm_2")
        logits = tf.layers.dense(x, data_dim, use_bias=False)
        return tf.nn.sigmoid(logits)



    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, data_dim], data[0], "input")
    with tf.variable_scope("Conditional"):
        cond = placeholder(tf.float32, [None, 1], data[1], "conditional")
    with tf.variable_scope("Noise"):
        z_inpt = placeholder(tf.float32, [None, z_dim], data[2], "noise")

    with tf.variable_scope("generator"):
        x_fake = generator(z_inpt, cond, btlnk_dim, dense_dim, data_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("discriminator") as scope:
        y_real = discriminator(inpt, cond, btlnk_dim, dense_dim, data_dim)
    with tf.variable_scope(scope,reuse=True):
        y_fake = discriminator(x_fake, cond, btlnk_dim, dense_dim, data_dim)

    with tf.variable_scope("loss"):
        with tf.variable_scope("d_loss"):
            d_loss_real = tf.reduce_mean(tf.losses.absolute_difference(inpt,y_real))
            d_loss_fake = tf.reduce_mean(tf.losses.absolute_difference(x_fake, y_fake))
            #d_loss_real = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=inpt, logits=y_real))
            #d_loss_fake = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=x_fake, logits=y_fake))
            d_loss = d_loss_real - d_loss_fake

        with tf.variable_scope("g_loss"):
            g_loss_g = tf.reduce_mean(tf.losses.absolute_difference(inpt, x_fake))
            g_loss_d = tf.reduce_mean(tf.losses.absolute_difference(x_fake, y_fake))
            #g_loss_g = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=inpt, logits=x_fake))
            #g_loss_d = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=x_fake, logits=y_fake))
            g_loss = g_loss_g + g_loss_d


    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


    return dict(step=step,
                inpt=inpt,
                cond=cond,
                d_loss=d_loss,
                g_loss=g_loss,
                d_step=d_step,
                g_step=g_step,
                x_fake=x_fake,
                z_inpt=z_inpt,
                y_fake=y_fake,
                y_real=y_real)
