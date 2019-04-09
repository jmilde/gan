from scipy.io.arff import loadarff

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
from util_np import np
from model import gan


# Hyperparam
trial = "test1"
train_d = True
train_g = False
batch_size = 2
dense_dim = 64
img_dim = 28*28

path_data = "../data/mnist/mnist_imgs.npy"
path_log = "../logs/"
path_ckpt = "../ckpt/"
# data pipeline
batch_fn = lambda: batch(batch_size, path_data)
data = pipe(batch_fn, (tf.uint8, tf.float32), prefetch=4)

# load graph
model = gan(data, batch_size, img_sim, dense_dim, train_d, train_g)

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
profile(sess, wrtr, model['loss'], feed_dict= None, prerun=3, tag='flow')
wrtr.flush()


for epoch in range(100): # 100 epochs
    for _ in range(280): # 1 epoch with batchsize=1
        for i in tqdm(range(250), ncols=70):
            sess.run(model['d_step'], {model["d_train"]=True, model["g_train"]=False})
            if i != 0:
                sess.run(model['g_step'], {model["d_train"]=False, model["g_train"]=True})


    saver.save(sess, pform(path_ckpt, trial, epoch), write_meta_graph=False)
