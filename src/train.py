from util_tf import tf, pipe, batch, placeholder, profile
from util_io import pform
from util_np import np
from model import gan
from tqdm import tqdm

# Hyperparams
path_data = "../data/mnist/mnist_imgs.npy"
path_log = "../logs/"
path_ckpt = "../ckpt/"


trial = "gan"
epochs = 200
batch_size = 500
batch_per_epoch= int(len(np.load(path_data))/500)

z_dim = 64
dense_dim = [256]
img_dim = 28*28
noise = 0.1
model_type = "gan" #"wgan", "wgan-p"
clip_limit = 0.01



# data pipeline
batch_fn = lambda: batch(batch_size, z_dim, path_data)
data = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)

# load graph
model = gan(data, img_dim, z_dim, dense_dim, noise, model_type, clip_limit)

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
                         tf.summary.image('img_fake', tf.reshape(model['x_fake'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('img_inpt', tf.reshape(model['inpt'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('imgs_fake', tf.reshape(tf.transpose(tf.reshape(model["x_fake"], (30,30,28,28)), (0,2,1,3)), (1,30*28,30*28, 1)))])

wrtr = tf.summary.FileWriter(pform(path_log, trial))
wrtr.add_graph(sess.graph)

# check computation
#profile(sess, wrtr, model['g_loss'], feed_dict= None, prerun=3, tag='generator')
#wrtr.flush()
#profile(sess, wrtr, model['d_loss'], feed_dict= None, prerun=3, tag='discriminator')
#wrtr.flush()


for epoch in range(epochs): #
    for i in tqdm(range(batch_per_epoch), ncols=70):
        #if wgan no diminishing gradient problem -> train D more for better start
        extra_iter = 100 if i<25 and model_type!="gan" and epoch==0 else 1
        for __ in range(extra_iter):
            sess.run(model['d_step'])
        if model_type=="wgan": sess.run(model["d_clip"])

        sess.run(model['g_step'])

    # tensorboard writer
    step = sess.run(model['step'])
    wrtr.add_summary(sess.run(smry, {model["z_inpt"]:np.random.normal(size= (30*30, z_dim))}), step)
    wrtr.flush()
saver.save(sess, pform(path_ckpt, trial, epoch), write_meta_graph=False)
