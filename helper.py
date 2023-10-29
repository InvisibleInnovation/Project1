import numpy as np

class Metrics:
    """
    Segmentation Metrics

    """
    def __init__(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        self.true_mask = true_mask
        self.pred_mask = pred_mask

        self.__TP = np.sum((self.true_mask != 0) & (self.pred_mask != 0))
        self.__FP = np.sum((self.true_mask == 0) & (self.pred_mask != 0))
        self.__TN = np.sum((self.true_mask == 0) & (self.pred_mask == 0))
        self.__FN = np.sum((self.true_mask != 0) & (self.pred_mask == 0))

    def iou(self, ) -> float:
        return round(self.__TP / (self.__TP + self.__FP + self.__FN), 3)

    def dsc(self, ) -> float:
        return round(2 * self.__TP / (2 * self.__TP + self.__FP + self.__FN), 3)
    
    def precision(self, ) -> float:
        return round(self.__TP / (self.__TP + self.__FP), 3)

    def senstivity(self, ) -> float: # Recall
        return round(self.__TP / (self.__TP + self.__FN), 3)

    def specificity(self, ) -> float:        
        return round(self.__TN / (self.__FP + self.__TN), 3)

    def randindex(self, ) -> float:
        return round((self.__TP + self.__TN) / (self.__TP + self.__TN + self.__FN + self.__FP), 3)

    def cohenkappa(self, ) -> float:
        num = 2 * (self.__TP * self.__TN - self.__FN * self.__FP)
        den = (self.__TP + self.__FP) * (self.__FP + self.__TN) + (self.__TP + self.__FN) * (self.__FN + self.__TN)
        return round(num/ den, 3)
    

# Testing the functions
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
