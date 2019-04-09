from util_tf import tf, placeholder

def gan(data, batch_size, img_dim, dense_dim):
        # generator
    def generator(noise, dense_dim, img_dim):
        x = tf.layers.dense(noise, dense_dim, activation=tf.nn.relu)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)
        x_generated = tf.layers.dense(x, img_dim, activation=tf.nn.relu)
        return x_generated

    # discriminator
    def discriminator(x, x_generated, dense_dim):
        x = tf.concat((x, x_generated),0)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)

        y = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
        y_data = tf.slice(y, [0, 0], [batch_size, -1], name=None)
        y_generated = tf.slice(y, [batch_size, 0], [-1, -1], name=None)

        return y_data, y_generated

    # Input
    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, img_dim], data[0], "Input")
        noise = placeholder(tf.float32, [None, img_dim], data[1], "Noise")


    # generator
    with tf.variable_scope("generator"):
        x_generated = generator(noise, dense_dim, img_dim)

    with tf.variable_scope("discriminator"):
        y_data, y_generated = discriminator(inpt, x_generated, dense_dim)

    with tf.variable_scope("learn_rate"):
        step = tf.train.get_or_create_global_step()
        #t = tf.to_float(step + 1)
        #learn_rate = (emb_dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))

    with tf.variable_scope("loss"):
        # log(1)=0, log(<1)= <0
        with tf.variable_scope("dscr_loss"):
            d_loss = tf.reduce_mean(- (tf.log(y_data) + tf.log(1 - y_generated)))
        with tf.variable_scope("gnrt_loss"):
            g_loss = tf.reduce_mean(- tf.log(y_generated)) # to minimize, y_generated has to become 1 -> fool the discriminator

    gnrt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    dscr_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss, step, var_list=dscr_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=gnrt_vars)


    return dict(step=step,
                d_loss=d_loss,
                g_loss=g_loss,
                d_step=d_step,
                g_step=g_step)
