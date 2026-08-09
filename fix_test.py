import numpy as np

def test():
    X = np.array([['a', 'b'], ['c', 'd']])
    if not np.issubdtype(X.dtype, np.number):
        print("Not a number")
        try:
            print(np.isnan(X).any())
        except Exception as e:
            print("Exception:", e)

test()
