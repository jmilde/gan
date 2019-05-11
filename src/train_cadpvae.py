from util_tf import tf, pipe, batch3, placeholder, profile
from util_io import pform
from util_np import np
from model_cadpvae import gan
from tqdm import tqdm
from numpy.random import RandomState

#load data
test_size = 900
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(np.concatenate([train_images, test_images[test_size:]]), (70000-test_size, 28*28))/255
y_train = np.concatenate([train_labels, test_labels[test_size:]])
x_test = np.reshape(test_images[:test_size],(test_size,28*28))/255
y_test= np.reshape(test_labels[:test_size],(test_size,1))

path_log = "../logs/"
path_ckpt = "../ckpt/"


trial = "mnist_cadpvae3"
epochs = 200
batch_size = 500

z_dim = 64
dense_dim = [64] # add numbers to create more layers, g and d are symatrical
btlnk_dim = 4
data_dim = len(x_train[0])

rand = RandomState(0) #fix seed
# data pipeline
batch_fn = lambda: batch3(x_train, y_train, batch_size, z_dim)
data = pipe(batch_fn, (tf.float32, tf.float32, tf.float32), prefetch=4)
#z = tf.random_normal((batch_size, z_dim))

# load graph
model = gan(data, btlnk_dim, data_dim, z_dim, dense_dim)

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
                         tf.summary.image('img_g', tf.reshape(model['x_fake'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('img_d_g', tf.reshape(model['y_fake'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('img_d_r', tf.reshape(model['y_real'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('imgs_g', tf.reshape(
                             tf.transpose(tf.reshape(model["x_fake"],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1))),
                         tf.summary.image('imgs_d_g', tf.reshape(
                             tf.transpose(tf.reshape(model["y_fake"],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1))),
                        tf.summary.image('imgs_d_r', tf.reshape(
                             tf.transpose(tf.reshape(model["y_real"],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1)))])


wrtr = tf.summary.FileWriter(pform(path_log, trial))
wrtr.add_graph(sess.graph)



for epoch in tqdm(range(epochs)):
    for i in range(len(x_train)//batch_size):
        sess.run(model['d_step'])
        sess.run(model['g_step'])
    # tensorboard writer
    step = sess.run(model['step'])
    wrtr.add_summary(sess.run(smry, {model["inpt"]:x_test,
                                     model["cond"]:y_test,
                                     model["z_inpt"]:rand.normal(size=(900,z_dim),)}), step)
    wrtr.flush()
