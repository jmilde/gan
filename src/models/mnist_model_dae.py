try:
    from util_tf import tf, placeholder, normalize
    from util_np import np
except ImportError:
    from src.util_tf import tf, placeholder, normalize
    from src.util_np import np

def gan(data, btlnk_dim, data_dim, dense_dim, y_dim):

    def generator(x, btlnk_dim, dense_dim, data_dim):
        x = normalize(tf.nn.relu(tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)), "layer_norm_1")
        logits = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        return tf.clip_by_value(logits, 0.0, 1.0)

    def discriminator(x, btlnk_dim, dense_dim, data_dim):
        x = normalize(tf.nn.relu(tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)), "layer_norm_1")
        logits = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        return tf.clip_by_value(logits, 0.0, 1.0)



    with tf.variable_scope("x"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "x")
    with tf.variable_scope("y"):
        y = placeholder(tf.float32, [None], data[1], "y")

    with tf.variable_scope("generator"):
        gz = generator(x, btlnk_dim, dense_dim, data_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("discriminator") as scope:
        dx = discriminator(x, btlnk_dim, dense_dim, data_dim)
    with tf.variable_scope(scope,reuse=True):
        dgz = discriminator(gz, btlnk_dim, dense_dim, data_dim)

    with tf.variable_scope("loss"):
        a = tf.reduce_mean(tf.abs(x - dx))
        b = tf.reduce_mean(tf.abs(gz - dgz))
        c =  tf.reduce_mean(tf.abs(x - gz))
        d_loss = a - b
        g_loss = b + c
        loss = a - b - c

    with tf.variable_scope("AUC"):
        anomaly_score = tf.reduce_mean((x-dgz)**2, axis=1)
        auc, _ = tf.metrics.auc(y, anomaly_score)

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        train_step = optimizer.apply_gradients(
            [((- grad if var.name.startswith("generator/") else grad), var)
             for grad, var in optimizer.compute_gradients(loss)], step)



    return dict(step=step,
                x=x,
                y=y,
                gz=gz,
                dgz=dgz,
                dx=dx,
                auc=auc,
                train_step=train_step,
                g_loss=g_loss,
                d_loss=d_loss)
