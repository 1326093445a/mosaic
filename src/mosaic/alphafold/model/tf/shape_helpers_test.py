
"""Tests for shape_helpers."""

from alphafold.model.tf import shape_helpers
import numpy as np
import tensorflow.compat.v1 as tf


class ShapeTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.disable_v2_behavior()

  def test_shape_list(self):
    """Test that shape_list can allow for reshaping to dynamic shapes."""
    a = tf.zeros([10, 4, 4, 2])
    p = tf.placeholder(tf.float32, shape=[None, None, 1, 4, 4])
    shape_dyn = shape_helpers.shape_list(p)[:2] + [4, 4]

    b = tf.reshape(a, shape_dyn)
    with self.session() as sess:
      out = sess.run(b, feed_dict={p: np.ones((20, 1, 1, 4, 4))})

    self.assertAllEqual(out.shape, (20, 1, 4, 4))


if __name__ == '__main__':
  tf.test.main()
