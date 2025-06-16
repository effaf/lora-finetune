import unittest
import utils

class TestIsColab(unittest.TestCase):
    def test_is_colab_false(self):
        # Should be False when not running in Colab
        self.assertFalse(utils.is_colab())

if __name__ == "__main__":
    utils.init (__name__)
    unittest.main()
