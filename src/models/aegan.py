try:
    from util_tf import tf, placeholder,normalize
    from util_np import np
except ImportError:
    from src.util_tf import tf, placeholder,normalize
    from src.util_np import np


def gan(data, data_dim, dense_dim, btlnk_dim, w=10):

    def encoder(x, btlnk_dim):
        x = normalize(tf.nn.relu(tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)), "layer_norm_1")
        return x

    def decoder(x, data_dim):
        x = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        return tf.clip_by_value(x, 0.0, 1.0)

    def discriminator(x, dense_dim):
        for i,d in enumerate(reversed(dense_dim)):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
                logits = tf.layers.dense(x, 1, use_bias=False)
        return tf.squeeze(logits, axis=1)

    with tf.variable_scope("Input"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "Input")
        y = placeholder(tf.float32, [None], data[1], "y")

    with tf.variable_scope("generator"):
        with tf.variable_scope("encoder"):
            z = encoder(x, btlnk_dim)
        with tf.variable_scope("decoder"):
            gx = decoder(z, data_dim)

    with tf.variable_scope("discriminator"):
        y_x = discriminator(x, dense_dim)
    with tf.variable_scope("discriminator",reuse=True):
        y_gx = discriminator(gx, dense_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("loss"):
        with tf.variable_scope("d_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_x)*0.9, logits=y_x))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_gx), logits=y_gx))
            d_loss = d_loss_real + d_loss_fake
        with tf.variable_scope("g_loss"):
            c_loss = tf.reduce_mean(abs(x-gx))
            a_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_gx), logits=y_gx))
            g_loss = c_loss +a_loss
            loss = d_loss_real + g_loss

    with tf.variable_scope("AUC"):
        anomaly_score = tf.reduce_mean((x-gx)**2, axis=1)
        _, auc = tf.metrics.auc(y, anomaly_score)

    #g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    #d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        train_step = optimizer.apply_gradients(
            [((- grad if var.name.startswith("generator") else grad), var)
             for grad, var in optimizer.compute_gradients(loss)]
            , step)
        #d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
        #g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


    return dict(step=step,
                x=x,
                y=y,
                auc=auc,
                d_loss=d_loss,
                g_loss=g_loss,
                gx=gx,
                y_x=y_x,
                train_step=train_step)
#d_step=d_step,
#g_step=g_step,)
