try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image

class Gen(Record):

    def __init__(self, dim_x, dim_z, dim_d, name= 'generator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_d, dim_z, name= 'lin')
            self.nrm = Normalize(    dim_d, name= 'nrm')
            self.lin2 = Linear(dim_d, dim_d, name= 'lin2')
            self.nrm2 = Normalize(    dim_d, name= 'nrm2')
            self.lex = Linear(dim_x, dim_d, name= 'lex')

    def __call__(self, z, name= None):
        with scope(name or self.name):
            x = self.nrm(tf.nn.relu(self.lin(z)))
            x = self.nrm2(tf.nn.relu(self.lin2(x)))
            return tf.clip_by_value(self.lex(x), 0.0, 1.0)


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


class GANAE(Record):

    @staticmethod
    def new(dim_x, dim_z, dim_d, dim_btlnk):
        return GANAE(dim_x= dim_x
                   , dim_z=dim_z
                   , gen= Gen(dim_x, dim_z, dim_d)
                   , dis= Dis(dim_x, dim_btlnk))

    def build(self, x, y, z):
        with scope("x"):
            x = placeholder(tf.float32, [None, self.dim_x], x, "x")
        with scope("y"):
            y = placeholder(tf.float32, [None], y, "y")
        with scope("z"):
            z = placeholder(tf.float32, [None, self.dim_z], z, "z")

        gz = self.gen(z)
        dx, dgz = self.dis(x), self.dis(gz)



        with tf.variable_scope("loss"):
            d_loss = tf.reduce_mean(tf.abs(x - dx)) - tf.reduce_mean(tf.abs(gz - dgz))
            g_loss = tf.reduce_mean(tf.abs(gz - dgz))# + tf.reduce_mean(tf.abs(gz - x)) scheint besser ohne

        with tf.variable_scope("AUC"):
            _, auc_dx = tf.metrics.auc(y, tf.reduce_mean((x-dx)**2, axis=1))

        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer()

        with tf.variable_scope("train_step"):
            step = tf.train.get_or_create_global_step()
            d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
            g_step = optimizer.minimize(g_loss, step, var_list=g_vars)


        return GANAE(self
                     , step=step
                     , x=x
                     , y=y
                     , z=z
                     , auc_dx=auc_dx
                     , d_loss=d_loss
                     , g_loss=g_loss
                     , gz=gz
                     , dx=dx
                     , dgz=dgz
                     , d_step=d_step
                     , g_step=g_step)
