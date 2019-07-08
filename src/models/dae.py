try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image

def sigmoid(x,shift=0,mult=1):
    return tf.constant(1.) / (tf.constant(1.)+ tf.exp(-tf.constant(1.0)*(x*mult)))

class Gen(Record):

    def __init__(self, dim_x, dim_btlnk, name= 'generator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_btlnk, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_btlnk, name= 'nrm')
            self.lex = Linear(dim_x, dim_btlnk, name= 'lex')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.clip_by_value(self.lex(self.nrm(tf.nn.relu(self.lin(x)))), 0.0, 1.0)
            #return tf.nn.sigmoid(self.lex(self.nrm(tf.nn.leaky_relu(self.lin(x)))))

class Dis(Record):

    def __init__(self, dim_x, dim_btlnk, name= 'discriminator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_btlnk, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_btlnk, name= 'nrm')
            self.lex = Linear(dim_x, dim_btlnk, name= 'lex')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.clip_by_value(self.lex(self.nrm(tf.nn.leaky_relu(self.lin(x)))), 0.0, 1.0)
            #return tf.nn.sigmoid(self.lex(self.nrm(tf.nn.leaky_relu(self.lin(x)))))


class DAE(Record):

    @staticmethod
    def new(dim_x, dim_btlnk):
        return DAE(dim_x= dim_x
                   , gen= Gen(dim_x, dim_btlnk)
                   , dis= Dis(dim_x, dim_btlnk))

    def build(self, x, y, lr_max, mult):
        with tf.variable_scope("x"):
            x = placeholder(tf.float32, [None, self.dim_x], x, "x")
        with tf.variable_scope("y"):
            y = placeholder(tf.float32, [None], y, "y")

        gx = self.gen(x)
        dx, dgx = self.dis(x), self.dis(gx)

        with tf.variable_scope("loss"):
            a = tf.reduce_mean(tf.abs(x - dx))
            b = tf.reduce_mean(tf.abs(gx - dgx))
            c =  tf.reduce_mean(tf.abs(x - gx))
            d_vs_g = a-(b+c)/2 # for balancing the learnign rate
            lr_d = sigmoid(d_vs_g, mult=mult)
            lr_g = (tf.constant(1.0) - lr_d)*lr_max
            lr_d = lr_d*lr_max

            # balance parameter for discriminator caring more about autoencoding real, or discriminating fake
            sigma = 0.5
            w_fake = tf.clip_by_value(
                sigmoid(b*sigma- a, shift=0. , mult=mult),
                0., 0.9) # hold the discrim proportion fake aways at less than half
            d_loss = a - b*w_fake


            # weights for generator
            wg_fake = tf.clip_by_value(
                sigmoid(b - c, shift=0. , mult=mult),
                0., 1.0)
            wg_reconstruct = 1-wg_fake
            g_loss = b*wg_fake + c*wg_reconstruct

        with tf.variable_scope("AUC"):
            _, auc_dgx = tf.metrics.auc(y, tf.reduce_mean((x-dgx)**2, axis=1))
            _, auc_dx = tf.metrics.auc(y, tf.reduce_mean((x-dx)**2, axis=1))
            _, auc_gx = tf.metrics.auc(y, tf.reduce_mean((x-gx)**2, axis=1))

        with scope('down'):
            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
            d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
            step = tf.train.get_or_create_global_step()
            d_step = tf.train.AdamOptimizer(lr_d).minimize(d_loss, step, var_list=d_vars)
            g_step =  tf.train.AdamOptimizer(lr_g).minimize(g_loss, step, var_list=g_vars)

        return DAE(self
                   , step=step
                   , x=x
                   , y=y
                   , gx=gx
                   , dgx=dgx
                   , dx=dx
                   , auc_dgx=auc_dgx
                   , auc_gx=auc_gx
                   , auc_dx=auc_dx
                   , g_loss=g_loss
                   , d_loss=d_loss
                   , d_step=d_step
                   , g_step=g_step)
                   #, train_step=train_step)
