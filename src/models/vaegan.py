try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
import math


def NLLNormal(pred, target):
    c = -0.5 * tf.log(2 * tf.constant(math.pi))
    multiplier = 1.0 / (2.0 * 1)
    tmp = tf.square(pred - target)
    tmp *= -multiplier
    tmp += c

    return tmp

class Enc(Record):

    def __init__(self, dim_x, dim_d, dim_btlnk, name= 'encoder'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_d, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_d, name= 'nrm')
            self.l_mu = Linear(dim_btlnk, dim_d, name= 'mu')
            self.l_lv = Linear(dim_btlnk, dim_d, name= 'lv')
    def __call__(self, x, name= None):
        with scope(name or self.name):
            hl = self.nrm(tf.nn.relu(self.lin(x)))
            with tf.variable_scope('latent'):
                mu = self.l_mu(hl)
                lv = self.l_lv(hl)

            with tf.variable_scope('z'):
                z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
            return z, mu, lv, hl

class Gen(Record):
    def __init__(self, dim_x, dim_btlnk, name="generator"):
        self.name = name
        with scope(name):
            self.lex = Linear(dim_x, dim_btlnk, name="lex")

    def __call__(self, x, name=None):
        with scope(name or self.name):
            return tf.clip_by_value(self.lex(x), 0.0, 1.0)

class Dis(Record):

    def __init__(self, dim_x, dim_d, name= 'discriminator'):
        self.name = name
        with scope(name):
            self.lin = Linear(dim_d, dim_x, name= 'lin')
            self.nrm = Normalize(    dim_d, name= 'nrm')
            self.lin2 = Linear(dim_d, dim_d, name= 'lin2')
            self.nrm2 = Normalize(    dim_d, name= 'nrm2')
            self.lex = Linear(1, dim_d, name= 'lex')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            x = self.nrm(tf.nn.leaky_relu(self.lin(x)))
            x = self.nrm2(tf.nn.leaky_relu(self.lin2(x)))
            return tf.clip_by_value(self.lex(x), 0.0, 1.0), x


class VAEGAN(Record):

    @staticmethod
    def new(dim_x, dim_btlnk, dim_d, dim_noise, accelerate):
        return VAEGAN(dim_x= dim_x
                      , dim_btlnk=dim_btlnk
                      , dim_noise=dim_noise
                      , enc = Enc(dim_x, dim_d, dim_btlnk)
                      , gen= Gen(dim_x, dim_btlnk)
                      , dis= Dis(dim_x, dim_d)
                      , accelerate=accelerate)

    def build(self, x, y, z):
        with scope("x"):
            x = placeholder(tf.float32, [None, self.dim_x], x, "x")
        with scope("y"):
            y = placeholder(tf.float32, [None], y, "y")
        with scope("z"):
            z = placeholder(tf.float32, [None, self.dim_noise], z, "z")

        zx, mu, lv, hl_e = self.enc(x)

        gzx = self.gen(zx)
        gz = self.gen(z)

        dx, _ = self.dis(x)
        dgzx, hl_dgzx = self.dis(gzx)
        dgz, hl_dgz = self.dis(gz)

        with tf.variable_scope("step"):
            step = tf.train.get_or_create_global_step()
            rate = self.accelerate * tf.to_float(step)
            rate_anneal = tf.tanh(rate)

        with scope("loss"):
            dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx)*0.9, logits=dx))
            dgzx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dgzx), logits=dgzx))
            dgz_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dgz), logits=dgz))
            d_loss = dx_loss + dgzx_loss + dgz_loss

            kl_loss = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))
            ftr_loss = tf.reduce_mean(tf.reduce_sum(NLLNormal(hl_dgzx, hl_dgz)))

            e_loss = kl_loss*rate_anneal - ftr_loss

            gzx_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dgzx), logits=dgzx))
            gz_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dgz), logits=dgz))
            g_loss = gz_loss +gzx_loss - ftr_loss

        with scope("AUC"):
            _, auc_gzx = tf.metrics.auc(y, tf.reduce_mean((x-gzx)**2, axis=1))
            _, auc_dx = tf.metrics.auc(y, tf.nn.sigmoid(dx))
            _, auc_dgzx = tf.metrics.auc(y, tf.nn.sigmoid(dgzx))

        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
        e_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")


        with scope('train_step'):
            optimizer = tf.train.AdamOptimizer()
            d_step = optimizer.minimize(d_loss, step, var_list=d_vars)
            g_step = optimizer.minimize(g_loss, step, var_list=g_vars)
            e_step = optimizer.minimize(e_loss, step, var_list=e_vars)


        return VAEGAN(self
                      , step=step
                      , x=x
                      , y=y
                      , z=z
                      , gz=gz
                      , gzx=gzx
                      , auc_gzx=auc_gzx
                      , auc_dx=auc_dx
                      , auc_dgzx=auc_dgzx
                      , g_step=g_step
                      , d_step=d_step
                      , e_step=e_step
                      , g_loss=g_loss
                      , d_loss=d_loss
                      , e_loss=e_loss)
