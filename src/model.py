from util_tf import tf

def gan(data, batch_size, img_dim, dense_dim, train_g, train_d):
        # generator
    def generator(noise, dense_dim, img_dim, train):
        x = tf.layers.dense(noise, dense_dim, activation=tf.nn.relu, trainable=train)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu, trainable=train)
        x_generated = tf.layers.dense(x, img_dim, activation=tf.nn.relu, trainable=train)
        return x_generated

    # discriminator
    def discriminator(data, x_generated, dense_dim, train):
        x = tf.concat(data, x_generated)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu, trainable=train)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu, trainable=train)

        y = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, trainable=train)
        y_data = tf.slice(y, [0, 0], [batch_size, -1], name=None)
        y_generated = tf.slice(y, [batch_size, 0], [-1, -1], name=None)

        return y_data, y_generated

    # Input
    with tf.variable_scope("Input"):
        inpt = placeholder(tf.int32, [None, img_dim], data[0], "Input")
        noise = placeholder(tf.float32, [None, img_dim], data[1], "Noise")
        train_d = placeholder(tf.bool, (), train_d, 'train_dscr')
        train_g = placeholder(tf.bool, (), train_g, 'train_gnrt')


    # generator
    with tf.variable_scope("generator"):
        x_generated = generator(noise, dense_dim, img_dim, train_g)

    with tf.variable_scope("discriminator"):
        y_data, y_generated = discriminator(data, x_generated, dense_dim, train_d)

    with tf.variable_scope("learn_rate"):
        step = tf.train.get_or_create_global_step()
        #t = tf.to_float(step + 1)
        #learn_rate = (emb_dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))

    with tf.variable_scope("loss"):
        with tf.variable_scope("dscr_loss"):
            d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
        with tf.variable_scope("gnrt_loss"):
            g_loss = - tf.log(y_generated)

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss) #todo, collect d_params
        g_step = optimizer.minimze(g_loss)


    return dict(step=step,
                d_train=d_train,
                g_train=g_train,
                d_loss=d_loss,
                g_loss=g_loss,
                d_step=d_step,
                g_step=g_step)
