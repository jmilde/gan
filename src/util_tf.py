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


def batch(size, noise_size, path_data, seed=25):
    """batch function to use with pipe, takes to numpy labels as input"""
    data = np.load(path_data)
    b = []
    for i in sample(len(data), seed):
        if size == len(b):
            z = np.random.normal(0, 1, size=(size, noise_size)).astype(np.float32)
            yield b, z
            b = []
        normalize = (data[i]-127.5)/127.5
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
