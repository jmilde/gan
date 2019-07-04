try:
    from util_tf import tf, placeholder, normalize
    from util_np import np
except ImportError:
    from src.util_tf import tf, placeholder, normalize
    from src.util_np import np

def ae(data, btlnk_dim, data_dim, dense_dim, y_dim, loss_type):

    def encoder(x, btlnk_dim):
        x = normalize(tf.nn.relu(tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)), "layer_norm_1")
        return x

    def decoder(x, data_dim):
        x = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        return tf.clip_by_value(x, 0.0, 1.0)

    with tf.variable_scope("x"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "x")
    with tf.variable_scope("y"):
        y = placeholder(tf.float32, [None], data[1], "y")

    with tf.variable_scope("encoder"):
        z = encoder(x, btlnk_dim)

    with tf.variable_scope("decoder"):
        logits = decoder(z, data_dim)

    with tf.variable_scope("loss"):
        if loss_type == "xtrpy":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=logits))
        else:
            loss = tf.reduce_mean(tf.abs(x - logits))
    step = tf.train.get_or_create_global_step()

    with tf.variable_scope("AUC"):
        anomaly_score = tf.reduce_mean((x-logits)**2, axis=1)
        _, auc = tf.metrics.auc(y, anomaly_score)

    with tf.variable_scope("train_step"):
        train_step = tf.train.AdamOptimizer().minimize(loss, step)


    return dict(step=step,
                x=x,
                y=y,
                logits=logits,
                auc=auc,
                train_step=train_step,
                loss=loss)
