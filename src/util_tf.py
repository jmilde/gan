import tensorflow as tf
from util_np import np, sample

def pipe(*args, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def batch3(x, y, size, oh_size, noise_size, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    b, l = [],[]
    for i in sample(len(x), seed):
        if size == len(b):
            z = np.random.normal(size=(size, noise_size))
            yield b, l, z
            b, l = [], []
        b.append(x[i])
        oh = np.zeros((oh_size))
        oh[y[i]]=1
        l.append(oh)

def batch2(x, y, size, noise_size, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    b, l = [],[]
    for i in sample(len(x), seed):
        if size == len(b):
            yield b, l
            b, l = [], []
        b.append(x[i])
        l.append(y[i])

def batch(size, noise_size, path_data, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    data = np.load(path_data)
    b = []
    for i in sample(len(data), seed):
        if size == len(b):
            yield b
            b = []
        #normalize = (data[i]-127.5)/127.5
        normalize = data[i]/255
        b.append(normalize)

def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)

def profile(sess, wrtr, run, feed_dict= None, prerun= 5, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wrtr.add_run_metadata(meta, tag)



def variable(name, shape, init= 'rand',
             initializers={'zero': tf.initializers.zeros(),
                           'unit': tf.initializers.ones(),
                           'rand': tf.glorot_uniform_initializer()}):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


def normalize(x, name="layer_norm"):
    with tf.variable_scope(name):
        dim = x.shape[-1]
        gain = variable('gain', (1, dim), 'unit')
        bias = variable('bias', (1, dim), 'zero')
        mean, var = tf.nn.moments(x, 1, keep_dims= True)
        return (x - mean) * tf.rsqrt(var + 1e-12) * gain + bias
