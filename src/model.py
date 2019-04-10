from util_tf import tf, placeholder
### todo:
# add batch normalization
# check out loss functions

def gan(data, batch_size, img_dim, noise_dim, dense_dim):

    def generator(noise, dense_dim, img_dim):
        x = tf.layers.dense(noise, dense_dim, activation=tf.nn.relu)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)
        x_generated = tf.layers.dense(x, img_dim, activation=tf.nn.tanh) #tanh for -1 to 1
        return x_generated

    def discriminator(x, dense_dim):
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)
        x = tf.layers.dense(x, dense_dim, activation=tf.nn.relu)
        y = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
        # seperate real and fake data and add constant (to avoid log(0)=nan)
        y_data = tf.slice(y, [0, 0], [batch_size, -1], name="y_data") + tf.constant(0.00001)
        y_generated = tf.slice(y, [batch_size, 0], [-1, -1], name="y_generated") + tf.constant(0.00001)
        return y_data, y_generated

    with tf.variable_scope("Input"):
        inpt = placeholder(tf.float32, [None, img_dim], data[0], "Input")
        noise = placeholder(tf.float32, [None, noise_dim], data[1], "Noise")

    with tf.variable_scope("generator"):
        x_generated = generator(noise, dense_dim, img_dim)

    with tf.variable_scope("generated_img"):
        #[batch_size, image_width, image_height, channels]
        x_image = tf.reshape(x_generated, [-1, 28, 28, 1])

    with tf.variable_scope("discr_input"):
        x = tf.concat((inpt, x_generated), 0)
        x = x + tf.random_normal(tf.shape(x), stddev=0.1)

    with tf.variable_scope("discriminator"):
        y_data, y_generated = discriminator(x, dense_dim)

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
        optimizer = tf.train.AdamOptimizer(0.0001)

    with tf.variable_scope("train_step"):
        d_step = optimizer.minimize(d_loss, step, var_list=dscr_vars)
        g_step = optimizer.minimize(g_loss, step, var_list=gnrt_vars)


    return dict(step=step,
                d_loss=d_loss,
                g_loss=g_loss,
                d_step=d_step,
                g_step=g_step,
                x_image=x_image,
                x_generated=x_generated)
