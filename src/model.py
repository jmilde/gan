from util_tf import tf, placeholder,normalize
from util_np import np
### todo:
# add batch normalization
# check out loss functions

def gan(data, img_dim, z_dim, dense_dim, noise, model_type, clip_limit):

    def generator(x, dense_dim, img_dim):
        for d in dense_dim:
            x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
        #x = tf.layers.dense(x, img_dim, activation=tf.nn.tanh)
        x = tf.layers.dense(x, img_dim, use_bias=False)
        return tf.clip_by_value(x, 0.0, 1.0)

    def discriminator(x, dense_dim, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            for d in reversed(dense_dim):
                x = normalize(tf.layers.dense(x, d, activation=tf.nn.leaky_relu, use_bias=False))
            logits = tf.layers.dense(x, 1, activation=tf.nn.leaky_relu, use_bias=False)
            return tf.squeeze(logits, axis=1)

    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, img_dim], data[0], "Input")
        z_inpt = placeholder(tf.float32, [None, z_dim], data[1], "Noise")
        #if conditional:
        #y_inpt = placeholder(tf.float32, [None, y_dim], "y")

    with tf.variable_scope("generator"):
        x_fake = generator(z_inpt, dense_dim, img_dim)


    step = tf.train.get_or_create_global_step()
    #with tf.variable_scope("learn_rate"):
        #step = tf.train.get_or_create_global_step()
        #t = tf.to_float(step + 1)
        #learn_rate = (emb_dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))

    #with tf.variable_scope("noise"):
        #if noise:
        #t = tf.to_float(step)+50
        #noise = (t+100)**-0.25
        #inpt = inpt + tf.random_normal(tf.shape(inpt), mean=0.0, stddev=noise)
        #x_fake = x_fake + tf.random_normal(tf.shape(x_fake), mean=0.0, stddev=noise)

        #if conditional:
        #inpt = tf.concat((y_inpt, inpt), 0)
        #x_fake = tf.concat((y_inpt, inpt), 0)

    y_real = discriminator(inpt, dense_dim)
    y_fake = discriminator(x_fake, dense_dim, reuse=True)

    with tf.variable_scope("loss"):
        if model_type == "gan":

            with tf.variable_scope("d_loss"):
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_real)*0.9, logits=y_real))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_fake), logits=y_fake))
                d_loss = d_loss_real + d_loss_fake

            with tf.variable_scope("g_loss"):
                g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_fake), logits=y_fake))
        else:
            with tf.variable_scope("d_loss"):
                d_loss = tf.reduce_mean(y_fake) - tf.reduce_mean(y_real)
            with tf.variable_scope("g_loss"):
                g_loss = - tf.reduce_mean(y_fake)

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("clipping"):
        d_clip = tf.group(*[v.assign(tf.clip_by_value(v, -clip_limit, clip_limit)) for v in d_vars])

    with tf.variable_scope("Optimizer"):
        if model_type == "gan":
            optimizer = tf.train.AdamOptimizer()
        else:
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
                z_inpt=z_inpt)
