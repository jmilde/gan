from util_tf import tf, pipe, batch2, placeholder, profile
from util_io import pform
from util_np import np
from model_gan import gan
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def analyze():
    p = sess.run(model["y_real"], {model["inpt"]: x_valid}) < 0.0
    y = y_valid.astype(np.bool)
    print("p sum: {}, y sum: {}".format(p.sum(), y.sum()))
    print("precision_score: {}".format(precision_score(y,p)))
    print("recall_score: {}".format(recall_score(y,p)))
    print("positive_cases: {}".format(np.mean(p)))

### LOAD DATA
data = np.load("../data/annthyroid.npz")
x_train = data["x_train"]
y_train = data["y_train"]
x_valid = data["x_valid"]
y_valid = data["y_valid"]

path_log = "../logs/"
path_ckpt = "../ckpt/"


trial = "anty_gan"
epochs = 200
batch_size = 500

z_dim = 64
dense_dim = [64] # add numbers to create more layers, g and d are symatrical
data_dim = 21
noise = 0.1


# data pipeline
batch_fn = lambda: batch2(x_train, y_train, batch_size, z_dim)
data = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
z = tf.random_normal((batch_size, z_dim))

# load graph
model = gan(data, z, data_dim, z_dim, dense_dim, noise)

# start session, initialize variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

### if load pretrained model
# pretrain = "modelname"
#saver.restore(sess, pform(path_ckpt, pretrain))
### else:
tf.global_variables_initializer().run()

# tensorboard summary
fneg = placeholder(tf.float32, (), name= 'fneg')
fpos = placeholder(tf.float32, (), name= 'fpos')
f1   = placeholder(tf.float32, (), name= 'f1')
pos_cases = placeholder(tf.float32, (), name= "pos_cases")

def log(step,
        log= tf.summary.merge(
            (tf.summary.scalar('fneg', fneg),
             tf.summary.scalar('fpos', fpos),
             tf.summary.scalar('f1',   f1),
             tf.summary.scalar("pos_cases", pos_cases),
             tf.summary.scalar("d_loss",model["d_loss"]),
             tf.summary.scalar("g_loss", model["g_loss"]))),
        wtr= tf.summary.FileWriter(path_log + trial),
        y= y_valid.astype(np.bool),
        x= x_valid):
    p = sess.run(model["y_real"], {model["inpt"]: x}) < 0.0
    wtr.add_summary(sess.run(log, {pos_cases: np.mean(p),
                                   fneg: np.mean(y & ~ p),
                                   fpos: np.mean(p & ~ y),
                                   f1: f1_score(y, p)}),
                    step)
    wtr.flush()



for epoch in tqdm(range(epochs)):
    for i in range(int(len(x_train)//batch_size)):
        sess.run(model['d_step'])
        sess.run(model['g_step'])
    # tensorboard writer
    log(sess.run(model["step"]))


analyze()
