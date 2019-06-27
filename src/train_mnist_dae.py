try:
    from src.util_tf import tf, pipe, batch2, placeholder, profile
    from src.util_io import pform
    from src.util_np import np, unison_shfl
    from src.models.dae import dae
except ImportError:
    from util_tf import tf, pipe, batch2, placeholder, profile
    from util_io import pform
    from util_np import np, unison_shfl
    from models.dae import dae
from tqdm import tqdm
from numpy.random import RandomState
import os

def train(anomaly_class = 8):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #load data
    (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    inlier = train_images[train_labels!=anomaly_class]
    x_train = np.reshape(inlier, (len(inlier), 28*28))/255
    #y_train = train_labels[train_labels!=anomaly_class]
    y_train = np.zeros(len(x_train), dtype=np.int8)

    outlier = train_images[train_labels==anomaly_class]
    x_test = np.reshape(np.concatenate([outlier, test_images])
                        ,(len(outlier)+len(test_images), 28*28))/255
    y_test= np.concatenate([train_labels[train_labels==anomaly_class], test_labels])
    y_test = [0 if y!=anomaly_class else 1 for y in y_test]
    x_test, y_test = unison_shfl(x_test, np.array(y_test))
    path_log = "/cache/tensorboard-logdir/ae"
    path_ckpt = "/project/outlier_detection/ckpt"

    epochs = 200
    batch_size = 64

    dense_dim = 64
    btlnk_dim = 32
    y_dim = 1
    data_dim = len(x_train[0])
    trial = f"dae{anomaly_class}_b{batch_size}_d{dense_dim}_btlnk{btlnk_dim}"


    rand = RandomState(0) #fix seed
    # data pipeline
    batch_fn = lambda: batch2(x_train, y_train, batch_size)
    data = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
    #z = tf.random_normal((batch_size, z_dim))

    # load graph
    model = dae(data, btlnk_dim, data_dim, y_dim)

    # start session, initialize variables
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()


    wrtr = tf.summary.FileWriter(pform(path_log, trial))
    #wrtr.add_graph(sess.graph)

    ### if load pretrained model
    # pretrain = "modelname"
    #saver.restore(sess, pform(path_ckpt, pretrain))
    ### else:
    auc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='AUC')
    init = tf.group(tf.global_variables_initializer(), tf.variables_initializer(var_list=auc_vars))
    sess.run(init)

    def log(step
            , wrtr= wrtr
            , log = tf.summary.merge([tf.summary.scalar('g_loss', model['g_loss'])
                                      , tf.summary.scalar('d_loss', model['d_loss'])
                                      , tf.summary.image('gz900', tf.reshape(
                                          tf.transpose(tf.reshape(model["gz"][:900],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1))),
                                      , tf.summary.image('dgz900', tf.reshape(
                                          tf.transpose(tf.reshape(model["dgz"][:900],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1))),
                                      , tf.summary.image('dx900', tf.reshape(
                                          tf.transpose(tf.reshape(model["dx"][:900],(30,30,28,28)),(0,2,1,3)),(1,30*28,30*28, 1)))
                                      , tf.summary.image('gz', tf.reshape(model['gz'], [-1,28,28,1]), max_outputs=1)
                                      , tf.summary.image('dgz', tf.reshape(model['dgz'], [-1,28,28,1]), max_outputs=1)
                                      , tf.summary.image('dx', tf.reshape(model['dx'], [-1,28,28,1]), max_outputs=1)
                                      , tf.summary.scalar("AUC", model["auc"])])
            , y= y_test
            , x= x_test):
        wrtr.add_summary(sess.run(log, {model["x"]:x_test, model["y"]:y_test}), step)
        wrtr.flush()



    for epoch in tqdm(range(epochs)):
        for i in range(len(x_train)//batch_size):
            sess.run(model['train_step'])
        # tensorboard writer
        log(sess.run(model["step"]))

    saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == "__main__":
    train()
