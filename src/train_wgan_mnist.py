from util_tf import tf, pipe, batch2, placeholder, profile
from util_io import pform
from util_np import np
from model_wgan import wgan
from tqdm import tqdm

# Hyperparams
#load data
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(np.concatenate([train_images, test_images]), (70000, 28*28))/255
y_train = np.concatenate([train_labels, test_labels])

path_log = "../logs/"
path_ckpt = "../ckpt/"


trial = "mnist_wgan_l"
epochs = 200
batch_size = 200

z_dim = 64
dense_dim = [256]
img_dim = 28*28
noise = 0.1
clip_limit = 0.01

# percentages derived from how others train models (usually diter=5 and d_xiter=25 with batch=64)
d_iter = max(2,0.005*len(x_train)//batch_size) # train d/g in a n/1 ratio
d_xiter = (0.025*len(x_train))//batch_size # extra iterations every half epoch etc.


# data pipeline
batch_fn = lambda: batch2(x_train, y_train, batch_size, z_dim)
data = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
z = tf.random_normal((batch_size, z_dim))

# load graph
model = wgan(data, z, img_dim, z_dim, dense_dim, noise, clip_limit)

# start session, initialize variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

### if load pretrained model
# pretrain = "modelname"
#saver.restore(sess, pform(path_ckpt, pretrain))
### else:
tf.global_variables_initializer().run()

# tensorboard summary
smry = tf.summary.merge([tf.summary.scalar('g_loss', model['g_loss']),
                         tf.summary.scalar('d_loss', model['d_loss']),
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

bpe = len(x_train)//batch_size # batch per epoch
print(bpe)
for epoch in tqdm(range(epochs)): #
    for i in range(bpe):

        #if wgan no diminishing gradient problem -> train D more for better start
        extra_iter = d_xiter if (i<(bpe*0.025) and epoch==0) or (i==bpe//2) else d_iter
        for __ in range(extra_iter):
            sess.run(model['d_step'])
            sess.run(model["d_clip"])

        sess.run(model['g_step'])

    # tensorboard writer
    step = sess.run(model['step'])
    wrtr.add_summary(sess.run(smry, {model["z_inpt"]:np.random.normal(size= (30*30, z_dim))}), step)
    wrtr.flush()
saver.save(sess, pform(path_ckpt, trial, epoch), write_meta_graph=False)
