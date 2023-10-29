import numpy as np

from helpers.helper import Metrics

arr1 = np.random.randint(0, 2, (3, 3))
arr2 = np.random.randint(0, 2, (3, 3))

metrics = Metrics(arr1, arr2)
print(metrics.iou())
print(metrics.dsc())
