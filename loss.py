import numbers
import numpy as np
import tensorflow as tf

def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.
    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).
    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.
    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.

    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]

def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))

def batch_hard(dists, pids, margin, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.