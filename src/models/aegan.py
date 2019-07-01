try:
    from util_tf import tf, placeholder,normalize
    from util_np import np
except ImportError:
    from src.util_tf import tf, placeholder,normalize
    from src.util_np import np


def gan(data, data_dim, dense_dim, btlnk_dim, w=1):

    def generator(x, bltnk_dim, data_dim):
        x = normalize(tf.nn.relu(tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)), "layer_norm_1")
        x = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        #return tf.clip_by_value(x, 0.0, 1.0)
        return tf.nn.sigmoid(x)

    def discriminator(x, dense_dim, data_dim):
        for i,d in enumerate(dense_dim):
            with tf.variable_scope("dense_layer{}".format(i)):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        logits = tf.layers.dense(x, 1, use_bias=False)
        return logits
        #return tf.clip_by_value(x, 0.0, 1.0)



    with tf.variable_scope("Input"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "Input")
        y = placeholder(tf.float32, [None], data[1], "y")
    with tf.variable_scope("generator"):
        gx = generator(x, btlnk_dim, data_dim)

    with tf.variable_scope("discriminator"):
        dx = discriminator(x, dense_dim, data_dim)
    with tf.variable_scope("discriminator",reuse=True):
        dgx = discriminator(gx, dense_dim, data_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("loss"):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx)*0.9, logits=dx))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dgx), logits=dgx))
        d_loss = d_loss_real + d_loss_fake
        g_loss = w*tf.reduce_mean(tf.abs(x - gx))+  tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dgx), logits=dgx))

    with tf.variable_scope("AUC"):
        anomaly_score = tf.reduce_mean((x-gx)**2, axis=1)
        _, auc = tf.metrics.auc(y, anomaly_score)
        _, auc_d = tf.metrics.auc(y, tf.nn.sigmoid(dx))

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


    return dict(step=step
                , x=x
                , y=y
                , auc=auc
                , auc_d=auc_d
                , d_loss=d_loss
                , g_loss=g_loss
                , gx=gx
                #, train_step=train_step)
                , d_step=d_step
                , g_step=g_step)
