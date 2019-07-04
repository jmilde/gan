try:
    from util import Record, identity, comp, partial
    from util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image
except ImportError:
    from src.util import Record, identity, comp, partial
    from src.util_tf import tf, scope, variable, placeholder, Linear, Affine, Normalize, spread_image

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

    def build(self, x, y):
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
            d_loss = a - b
            g_loss = b + c
            loss = a - b - c

        with tf.variable_scope("AUC"):
            _, auc_dgx = tf.metrics.auc(y, tf.reduce_mean((x-dgx)**2, axis=1))
            _, auc_dx = tf.metrics.auc(y, tf.reduce_mean((x-dx)**2, axis=1))
            _, auc_gx = tf.metrics.auc(y, tf.reduce_mean((x-gx)**2, axis=1))

        with scope('down'):
            step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer()
            train_step = optimizer.apply_gradients(
            [((- grad if var.name.startswith("generator") else grad), var)
             for grad, var in optimizer.compute_gradients(loss)], step)


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
                   , train_step=train_step
                   , g_loss=g_loss
                   , d_loss=d_loss)
