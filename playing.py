import numpy as np
import itertools
a = np.array([[1, 2],[3,4]])
print(a)
a = np.insert(arr = a,
              obj = 2,
              values = 4,
              axis = 1)
print(a)

b = [[1,2],[3,4],[5,6]]
c = [["a","b"],["v", 'c'],["e","f"]]
res = [list(itertools.chain(*i))
       for i in zip(b, c)]
print(res)