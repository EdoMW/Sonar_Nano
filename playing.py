import numpy as np
import itertools

def check():
    if True:
        return
    print(hi)
check()

a = np.array([[1, 2],[3,4]])
print(a)
a = np.insert(arr = a,
              obj = 2,
              values = 4,
              axis = 1)
print(a)
#
# b = [[1,2],[3,4],[5,6]]
# c = [["a","b"],["v", 'c'],["e","f"]]
# res = [list(itertools.chain(*i))
#        for i in zip(b, c)]
# print(res)
#
#
print("np")
for i in range(3):
    print(a)
    corner_list = a[i:i+1,:2]
    corn = corner_list.tolist()
    print(corn)
    print(type(corn))