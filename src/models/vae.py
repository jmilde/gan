try:
    from util_tf import tf, placeholder, Linear, Normalize
    from util_np import np
except ImportError:
    from src.util_tf import tf, placeholder, Linear, Normalize
    from src.util_np import np

def VAE(data, btlnk_dim, data_dim, dense_dim, y_dim, loss_type, accelerate):

    def encoder(x, dim_btlnk, dim_x):
        x = Normalize(dim_btlnk, "nrm")(tf.nn.elu(Linear(dim_btlnk, dim_x, name= 'lin')(x)))
        with tf.variable_scope('latent'):
            mu = Linear(dim_btlnk, dim_btlnk, name= 'mu')(x)
            lv = Linear(dim_btlnk, dim_btlnk, name= 'lv')(x)
            #lv = Linear(dim_btlnk, dim_x, name= 'lv')(x)
            #mu = Linear(dim_btlnk, dim_x, name= 'mu')(x)
        with tf.variable_scope('z'):
            z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        return z, mu, lv

    def decoder(x, data_dim, btlnk_dim):
        x = Linear(data_dim, btlnk_dim)(x)
        #return tf.clip_by_value(x, 0.0, 1.0)
        return tf.nn.sigmoid(x)

    with tf.variable_scope("x"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "x")
    with tf.variable_scope("y"):
        y = placeholder(tf.float32, [None], data[1], "y")

    with tf.variable_scope("encoder"):
        z, mu, lv = encoder(x, btlnk_dim, data_dim)

    with tf.variable_scope("decoder"):
        logits = decoder(z, data_dim, btlnk_dim)

    with tf.variable_scope("step"):
        step = tf.train.get_or_create_global_step()
        rate = accelerate * tf.to_float(step)
        rate_anneal = tf.tanh(rate)

    with tf.variable_scope("loss"):
        kl_loss = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))
        if loss_type == "xtrpy":
            #loss_rec = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=x, logits=logits))
            epsilon = 1e-10
            loss_rec = tf.reduce_mean(-tf.reduce_sum(x * tf.log(epsilon+logits) +
                                                     (1-x) * tf.log(epsilon+1-logits),  axis=1))
        else:
            loss_rec = tf.reduce_mean(tf.abs(x - logits))

        loss = loss_rec + kl_loss*rate_anneal

    with tf.variable_scope("AUC"):
        anomaly_score = tf.reduce_mean((x-logits)**2, axis=1)
        _, auc = tf.metrics.auc(y, anomaly_score)

    with tf.variable_scope("train_step"):
        train_step = tf.train.AdamOptimizer().minimize(loss, step)


    return dict(step=step,
                x=x,
                y=y,
                z=z,
                mu=mu,
                logits=logits,
                auc=auc,
                train_step=train_step,
                loss=loss,
                kl_loss=kl_loss,
                loss_rec=loss_rec)
