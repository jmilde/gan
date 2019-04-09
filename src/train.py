#from scipy.io.arff import loadarff

#path_data = '../data/semantic/Annthyroid/Annthyroid_withoutdupl_norm_07.arff'

#X, Y = [], []

# Y = 0: 6595, 1: 534
#for data in loadarff(path_data)[0]:
#    if data[-1]==b"'no'":
#        Y.append(0)
#    else:
#        Y.append(1)
#    X.append(list(data)[:-2])



from util_tf import tf, pipe, batch, placeholder, profile
from util_io import pform
from util_np import np
from model import gan
from tqdm import tqdm

# Hyperparams
trial = "test"
batch_size = 256 # todo: maybe write model with batch_size//2
dense_dim = 512
img_dim = 28*28

path_data = "../data/mnist/mnist_imgs.npy"
path_log = "../logs/"
path_ckpt = "../ckpt/"
# data pipeline
batch_fn = lambda: batch(batch_size, path_data)
data = pipe(batch_fn, (tf.uint8, tf.float32), prefetch=4)

# load graph
model = gan(data, batch_size, img_dim, dense_dim)

# start session, initialize variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

### if load pretrained model
# pretrain = "modelname"
#saver.restore(sess, pform(path_ckpt, pretrain))
### else:
tf.global_variables_initializer().run()

# tensorboard summary
summary = tf.summary.merge((tf.summary.scalar('gnrt_loss', model['g_loss']),
                            tf.summary.scalar('dscr_loss', model['d_loss'])))

wrtr = tf.summary.FileWriter(pform(path_log, trial))
wrtr.add_graph(sess.graph)

# check computation
#profile(sess, wrtr, model['g_loss'], feed_dict= None, prerun=3, tag='generator')
#wrtr.flush()
#profile(sess, wrtr, model['d_loss'], feed_dict= None, prerun=3, tag='discriminator')
#wrtr.flush()


for epoch in range(100): # 100 epochs
    for _ in range(280): # 1 epoch with batchsize=1
        for i in tqdm(range(250), ncols=70):
            sess.run(model['d_step'])
            sess.run(model['g_step'])
        # tensorboard writer
        step = sess.run(model['step'])
        wrtr.add_summary(sess.run(summary), step)
    saver.save(sess, pform(path_ckpt, trial, epoch), write_meta_graph=False)
wrtr.flush()
