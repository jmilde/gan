from util_tf import tf, pipe, batch, placeholder, profile
from util_io import pform
from util_np import np
from model import gan
from tqdm import tqdm

# Hyperparams
trial = "test3"
batch_size = 64 # todo: maybe write model with batch_size//2
noise_dim = 100
dense_dim = 50
img_dim = 28*28

path_data = "../data/mnist/mnist_imgs.npy"
path_log = "../logs/"
path_ckpt = "../ckpt/"
# data pipeline
batch_fn = lambda: batch(batch_size, noise_dim, path_data)
data = pipe(batch_fn, (tf.uint8, tf.float32), prefetch=4)

# load graph
model = gan(data, batch_size, img_dim, noise_dim, dense_dim)

# start session, initialize variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

### if load pretrained model
# pretrain = "modelname"
#saver.restore(sess, pform(path_ckpt, pretrain))
### else:
tf.global_variables_initializer().run()

# tensorboard summary
smry = tf.summary.merge([tf.summary.scalar('gnrt_loss', model['g_loss']),
                         tf.summary.scalar('dscr_loss', model['d_loss']),
                         tf.summary.image('img_generated', model['x_image'], max_outputs=1)])

wrtr = tf.summary.FileWriter(pform(path_log, trial))
wrtr.add_graph(sess.graph)

# check computation
#profile(sess, wrtr, model['g_loss'], feed_dict= None, prerun=3, tag='generator')
#wrtr.flush()
#profile(sess, wrtr, model['d_loss'], feed_dict= None, prerun=3, tag='discriminator')
#wrtr.flush()


for epoch in range(10): # 10*50 epochs
    for _ in range(50): #
        for i in tqdm(range(250), ncols=70):
            sess.run(model['d_step'])
            sess.run(model['g_step'])
        # tensorboard writer
        step = sess.run(model['step'])
        wrtr.add_summary(sess.run(smry), step)
        wrtr.flush()
    saver.save(sess, pform(path_ckpt, trial, epoch), write_meta_graph=False)
