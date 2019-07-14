from tqdm import tqdm
from numpy.random import RandomState
import os
import math
try:
    from util_np import np, batch_sample
    from util_tf import pipe, tf, spread_image, batch2
    from util_io import pform
    from models.dae import DAE
except ImportError:
    from src.util_np import np, batch_sample,unison_shfl
    from src.util_tf import pipe, tf, spread_image, batch2
    from src.util_io import pform
    from src.models.dae import DAE


def sigmoid(x,shift=0,mult=1):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + math.exp(-(x+shift)*mult))

def plot_sigmoid():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(18,4))
    plt.plot(np.arange(0,1,.01), [sigmoid(i/100.,0,1) for i in range(100)])
    ax.set_xlabel('Mean of Discriminator(Real) or Discriminator(Fake)')
    ax.set_ylabel('Multiplier for learning rate')
    plt.title('Squashing the Learning Rate to balance Discrim/Gen network performance')
    plt.show()

def train(anomaly_class = 8):
    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #load data
    (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    inlier = train_images[train_labels!=anomaly_class]
    x_train = np.reshape(inlier, (len(inlier), 28*28))/255
    #y_train = train_labels[train_labels!=anomaly_class]
    y_train = np.zeros(len(x_train), dtype=np.int8) # dummy

    outlier = train_images[train_labels==anomaly_class]
    x_test = np.reshape(np.concatenate([outlier, test_images])
                        ,(len(outlier)+len(test_images), 28*28))/255
    y_test= np.concatenate([train_labels[train_labels==anomaly_class], test_labels])
    y_test = [0 if y!=anomaly_class else 1 for y in y_test]
    x_test, y_test = unison_shfl(x_test, np.array(y_test))


    path_log = "/cache/tensorboard-logdir/ae"
    path_ckpt = "/project/outlier_detection/ckpt"

    epochs = 400
    batch_size = 700
    dim_btlnk = 32
    mult=20
    lr_max = 1e-4
    trial = f"dae{anomaly_class}_b{batch_size}_btlnk{dim_btlnk}_lr_{lr_max}m{mult}"
    #trial="test1"
    dim_x = len(x_train[0])

    #reset graphs and fix seeds
    tf.reset_default_graph()
    if 'sess' in globals(): sess.close()
    rand = RandomState(0)
    tf.set_random_seed(0)

    # data pipeline
    batch_fn = lambda: batch2(x_train, y_train, batch_size)
    x, y = pipe(batch_fn, (tf.float32, tf.float32), prefetch=4)
    #z = tf.random_normal((batch_size, z_dim))

    # load graph
    dae = DAE.new(dim_x, dim_btlnk)
    model = DAE.build(dae, x, y, lr_max, mult)


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
            , log = tf.summary.merge([tf.summary.scalar('g_loss', model.g_loss)
                                      , tf.summary.scalar('d_loss', model.d_loss)
                                      , tf.summary.image('gx400', spread_image(model.gx[:400], 20, 20, 28 ,28))
                                      , tf.summary.image('dgx400', spread_image(model.dgx[:400], 20, 20, 28 ,28))
                                      , tf.summary.image('dx400', spread_image(model.dx[:400], 20, 20, 28 ,28))
                                      , tf.summary.scalar("AUC_dgx", model.auc_dgx)
                                      , tf.summary.scalar("AUC_dx", model.auc_dx)
                                      , tf.summary.scalar("AUC_gx", model.auc_gx)])
            , y= y_test
            , x= x_test):
        wrtr.add_summary(sess.run(log, {model.x:x, model.y:y}), step)
        wrtr.flush()


    steps_per_epoch = len(x_train)//batch_size
    for epoch in tqdm(range(epochs)):
        for i in range(steps_per_epoch):
            sess.run(model.d_step)
            sess.run(model.g_step)

        # tensorboard writer
        log(sess.run(model["step"])//steps_per_epoch)

    saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == "__main__":
    for i in range(0,10):
        train(i)
