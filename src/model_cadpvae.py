from src.util_tf import tf, placeholder, normalize
from src.util_np import np


### todo
# try: clipping because sigmoid
# add variational part
#  fix absolute loss
# loss could be tried with sigmoid loss

def gan(data, btlnk_dim, data_dim, cond_dim, z_dim, dense_dim, accelerate):

    def generator(x, cond, btlnk_dim, dense_dim, data_dim):
        #x = tf.layers.dense(x, data_dim, use_bias=False)
        #cond = tf.layers.dense(cond, data_dim, use_bias=False)
        x = tf.keras.layers.Dense(dense_dim, use_bias=False)(x)
        cond = tf.keras.layers.Dense(dense_dim, use_bias=False)(cond)
        x = normalize(tf.nn.relu(x+cond), "layer_norm_1")

        with tf.variable_scope('latent'): # (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        # h = layer_aff(h, dim_emb, name='in')
            mu = tf.keras.layers.Dense(btlnk_dim, name='mu')(x)
            lv = tf.keras.layers.Dense(btlnk_dim, name='lv')(x)

        with tf.variable_scope('z'):
            x = mu
            x += tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
            z = x

        x = tf.keras.layers.Dense(dense_dim, use_bias=False)(x)
        logits = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        #for i,d in enumerate(dense_dim[1:]):
            #with tf.variable_scope("dense_layer{}".format(i)):
                #x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        # bottleneck
        #x = normalize(tf.layers.dense(x, btlnk_dim, use_bias=False, activation=tf.nn.leaky_relu), "layer_norm_2")
        # output
        logits = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        #return logits
        #return tf.nn.sigmoid(logits)
        return tf.clip_by_value(logits, 0.0, 1.0), mu, lv, z


    def discriminator(x, cond, btlnk_dim, dense_dim, data_dim):
        #x = tf.layers.dense(x, data_dim, use_bias=False)
        #cond = tf.layers.dense(cond, data_dim, use_bias=False)
        x = tf.keras.layers.Dense(dense_dim, use_bias=False)(x)
        cond = tf.keras.layers.Dense(dense_dim, use_bias=False)(cond)
        x = normalize(tf.nn.relu(x+cond), "layer_norm_1")
        #for i,d in enumerate(reversed(dense_dim)):
            #with tf.variable_scope("dense_layer{}".format(i)):
                #x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        # bottleneck
        x = tf.keras.layers.Dense(btlnk_dim, use_bias=False)(x)

        x = tf.keras.layers.Dense(dense_dim, name="bottleneck", use_bias=False)(x)
        logits = tf.keras.layers.Dense(data_dim, use_bias=False)(x)
        #return logits
        #return tf.nn.sigmoid(logits)
        return tf.clip_by_value(logits, 0.0, 1.0)



    with tf.variable_scope("x"):
        x = placeholder(tf.float32, [None, data_dim], data[0], "input")
    with tf.variable_scope("Conditional"):
        y = placeholder(tf.float32, [None, cond_dim], data[1], "conditional")
    #with tf.variable_scope("Noise"):
    #    z_inpt = placeholder(tf.float32, [None, z_dim], data[2], "noise")

    with tf.variable_scope("generator"):
        #x_fake = generator(z_inpt, cond, btlnk_dim, dense_dim, data_dim)
        gz, mu, lv, z = generator(x, y, btlnk_dim, dense_dim, data_dim)

    with tf.variable_scope("discriminator"):
        dx = discriminator(x, y, btlnk_dim, dense_dim, data_dim)
    with tf.variable_scope("discriminator", reuse=True):
        dgz = discriminator(gz, y, btlnk_dim, dense_dim, data_dim)

    with tf.variable_scope("step"):
        step = tf.train.get_or_create_global_step()
        rate = accelerate * tf.to_float(step)
        rate_anneal = tf.tanh(rate)

    with tf.variable_scope("loss"):
        with tf.variable_scope("KL_loss"):
            kl_loss = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))
        with tf.variable_scope("d_loss"):
            d_loss_real = tf.reduce_mean(tf.abs(x - dx))
            d_loss_fake = tf.reduce_mean(tf.abs(gz - dgz))
            d_loss = d_loss_real - d_loss_fake

            #d_loss_real = tf.reduce_mean(
            #    tf.nn.sigmoid_cross_entropy_with_logits(labels=inpt, logits=y_real))
            #d_loss_fake = tf.reduce_mean(
            #    tf.nn.sigmoid_cross_entropy_with_logits(labels=x_fake, logits=y_fake))
            #d_loss = d_loss_real + d_loss_fake

        with tf.variable_scope("g_loss"):
            g_loss_g = tf.reduce_mean(tf.abs(x - gz))
            g_loss_d = tf.reduce_mean(tf.abs(gz - dgz))
            g_loss = g_loss_g + g_loss_d + kl_loss * rate_anneal

            #g_loss_g = tf.reduce_mean(
            #    tf.nn.sigmoid_cross_entropy_with_logits(labels=, logits=x_fake))
            #g_loss_d = tf.reduce_mean(
            #    tf.nn.sigmoid_cross_entropy_with_logits(labels=x_fake, logits=y_fake))
            #g_loss = g_loss_g + g_loss_d + kl_loss * rate_anneal

        #a = tf.reduce_mean(tf.abs(x - dx))
        #b = tf.reduce_mean(tf.abs(gz - dgz))
        #c =  tf.reduce_mean(tf.abs(x - gz))
        #d_loss = a - b
        #g_loss = b + c
        #loss = a - b - c - kl_loss * rate_anneal

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        #train_step = optimizer.apply_gradients(
        #    [((- grad if var.name.startswith("generator/") else grad), var)
        #     for grad, var in optimizer.compute_gradients(loss)], step)

        d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


    return dict(step=step,
                x=x,
                y=y,
                gz=gz,
                dgz=dgz,
                dx=dx,
                g_loss=g_loss,
                d_loss=d_loss,
                kl_loss=kl_loss,
                #train_step=train_step,
                d_step=d_step,
                g_step=g_step)

                #,z_inpt=z_inpt)
