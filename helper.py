import numpy as np

class Metrics:
    """
    Segmentation Metrics

    """
    def __init__(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        self.true_mask = true_mask
        self.pred_mask = pred_mask

        self.TP = np.sum((self.true_mask != 0) & (self.pred_mask != 0))
        self.FP = np.sum((self.true_mask == 0) & (self.pred_mask != 0))
        self.TN = np.sum((self.true_mask == 0) & (self.pred_mask == 0))
        self.FN = np.sum((self.true_mask != 0) & (self.pred_mask == 0))

    def iou(self, ) -> float:
        return round(self.TP / (self.TP + self.FP + self.FN), 3)

    def dsc(self, ) -> float:
        return round(2 * self.TP / (2 * self.TP + self.FP + self.FN), 3)
    
    def precision(self, ) -> float:
        return round(self.TP / (self.TP + self.FP), 3)

    def senstivity(self, ) -> float: # Recall
        return round(self.TP / (self.TP + self.FN), 3)

    def specificity(self, ) -> float:        
        return round(self.TN / (self.FP + self.TN), 3)

    def randindex(self, ) -> float:
        return round((self.TP + self.TN) / (self.TP + self.TN + self.FN + self.FP), 3)

    def cohenkappa(self, ) -> float:
        num = 2 * (self.TP * self.TN - self.FN * self.FP)
        den = (self.TP + self.FP) * (self.FP + self.TN) + (self.TP + self.FN) * (self.FN + self.TN)
        return round(num/ den, 3)


if __name__ == "__main__":
    arr1 = np.random.randint(0, 2, (3, 3))
    arr2 = np.random.randint(0, 2, (3, 3))

    print("ARR1:")
    print(arr1)
    print("ARR2:")
    print(arr2, "\n")

    metrics = Metrics(arr1, arr2)
    print("IOU:", metrics.iou())
    print("DSC:", metrics.dsc())
    print("PRECISION:", metrics.precision())
    print("SENSTIVITY:", metrics.senstivity())
    print("SPECIFICITY:", metrics.specificity())
    print("RAND-INDEX:", metrics.randindex())
    print("COHEN-KAPPA:", metrics.cohenkappa())
