try:
    from src.util_tf import tf, pipe, batch2, placeholder, profile, spread_image
    from src.util_io import pform
    from src.util_np import np, unison_shfl
    from src.models.ganae import GANAE
except ImportError:
    from util_tf import tf, pipe, batch2, placeholder, profile, spread_image
    from util_io import pform
    from util_np import np, unison_shfl
    from models.ganae import GANAE
from tqdm import tqdm
from numpy.random import RandomState
import os

def train(anomaly_class = 8):
    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    dim_z = 64
    dim_d = 64
    dim_btlnk = 32
    trial = f"gana2e{anomaly_class}_b{batch_size}_z{dim_z}_d{dim_d}_b{dim_btlnk}"

    dim_x = len(x_train[0])

     #fix seeds
    rand = RandomState(0)
    tf.set_random_seed(0)

    # data pipeline
    batch_fn = lambda: batch2(x_train, y_train, batch_size, dim_z)
    x,y,z = pipe(batch_fn, (tf.float32, tf.float32, tf.float32), prefetch=4)

    ganae = GANAE.new(dim_x, dim_z, dim_d, dim_btlnk)
    model = GANAE.build(ganae, x, y, z)

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
                                      , tf.summary.image('gz400', spread_image(model.gz[:400], 20,20,28,28))
                                      , tf.summary.image('dx400', spread_image(model.dx[:400], 20,20,28,28))
                                      , tf.summary.image('dgz400', spread_image(model.dgz[:400], 20,20,28,28))
                                      , tf.summary.scalar("AUC_dx", model.auc_dx)])
            , y= y_test
            , x= x_test
            , z= np.random.normal(size=(len(y_test), dim_z))):
        wrtr.add_summary(sess.run(log, {model.x:x
                                        , model.y:y
                                        , model.z:z})
                         , step)
        wrtr.flush()

    steps_per_epoch = len(x_train)//batch_size-1
    for epoch in tqdm(range(epochs)):
        for i in range(steps_per_epoch):
            #sess.run(model["train_step"])
            sess.run(model['d_step'])
            sess.run(model['g_step'])
        # tensorboard writer
        log(sess.run(model["step"])//steps_per_epoch)

    saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == "__main__":
    train()
