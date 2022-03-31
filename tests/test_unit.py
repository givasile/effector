# hack to import mdale
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mdale.ale2


# def test_bins():
#     X = np.array([[1, 2], [2.000, 2]])
#     s = 0
#     K = 5
#
#     bins, dx = mdale.ale2.create_bins2(X, s, K)
#
#     assert
#     print(bins)
#     print(dx)
