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
            x = self.nrm(tf.nn.relu(self.lin(x)))
            x = self.nrm2(tf.nn.relu(self.lin2(x)))
            return tf.clip_by_value(self.lex(x), 0.0, 1.0)


class GAN(Record):

    @staticmethod
    def new(dim_x, dim_z, dim_d):
        return GAN(dim_x= dim_x
                   , dim_z=dim_z
                   , gen= Gen(dim_x, dim_z, dim_d)
                   , dis= Dis(dim_x, dim_d))

    def build(self, x, y, z):
        with scope("x"):
            x = placeholder(tf.float32, [None, self.dim_x], x, "x")
        with scope("y"):
            y = placeholder(tf.float32, [None], y, "y")
        with scope("z"):
            z = placeholder(tf.float32, [None, self.dim_z], z, "z")

        gz = self.gen(z)
        dx, dgz = self.dis(x), self.dis(gz)

        with scope("loss"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(dx), logits= dx)) \
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.zeros_like(dgz), logits= dgz))

            #with scope("d_loss"):
                #d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_real)*0.9, logits=y_real))
                #d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_fake), logits=y_fake))
                #d_loss = d_loss_real + d_loss_fake
            #with scope("g_loss"):
                #g_loss = tf.reduce_mean(
                    #tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_fake), logits=y_fake))
            #with scope("g/d_loss"):
                #loss = d_loss_real + g_loss

        with scope("AUC"):
            _, auc_d = tf.metrics.auc(y, tf.nn.sigmoid(dx))


        with scope("train_step"):
            step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer()
            train_step = optimizer.apply_gradients(
                        [((- grad if var.name.startswith("generator") else grad), var)
                         for grad, var in optimizer.compute_gradients(loss)]
                        , step)

        return GAN(self
                   , step=step
                   , x=x
                   , y=y
                   , z=z
                   , auc_d=auc_d
                   , gz=gz
                   , train_step=train_step)
