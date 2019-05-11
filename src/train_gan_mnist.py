from util_tf import tf, pipe, batch2, placeholder, profile
from util_io import pform
from util_np import np
from model_gan import gan
from tqdm import tqdm

#load data
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(np.concatenate([train_images, test_images]), (70000, 28*28))/255
y_train = np.concatenate([train_labels, test_labels])

path_log = "../logs/"
path_ckpt = "../ckpt/"


trial = "mnist_gan"
epochs = 200
batch_size = 500

z_dim = 64
dense_dim = [64] # add numbers to create more layers, g and d are symatrical
data_dim = len(x_train[0])
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
smry = tf.summary.merge([tf.summary.scalar('gnrt_loss', model['g_loss']),
                         tf.summary.scalar('dscr_loss', model['d_loss']),
                         tf.summary.image('img_fake', tf.reshape(model['x_fake'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('img_inpt', tf.reshape(model['inpt'], [-1,28,28,1]), max_outputs=1),
                         tf.summary.image('imgs_fake', tf.reshape(tf.transpose(tf.reshape(model["x_fake"], (30,30,28,28)), (0,2,1,3)), (1,30*28,30*28, 1)))])

wrtr = tf.summary.FileWriter(pform(path_log, trial))
wrtr.add_graph(sess.graph)



for epoch in tqdm(range(epochs)):
    for i in range(len(x_train)//batch_size):
        sess.run(model['d_step'])
        sess.run(model['g_step'])
    # tensorboard writer
    step = sess.run(model['step'])
    wrtr.add_summary(sess.run(smry, {model["z_inpt"]:np.random.normal(size= (30*30, z_dim))}), step)
    wrtr.flush()
