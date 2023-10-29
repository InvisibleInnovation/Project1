import numpy as np

class Metrics:
    """
    A class for evaluating segmentation performance using various metrics.

    Parameters:
    true_mask (np.ndarray): Ground truth binary mask.
    pred_mask (np.ndarray): Predicted binary mask.

    Attributes:
    true_mask (np.ndarray): Ground truth binary mask.
    pred_mask (np.ndarray): Predicted binary mask.
    __TP (int): True Positive count.
    __FP (int): False Positive count.
    __TN (int): True Negative count.
    __FN (int): False Negative count.

    Methods:
    Intersection over union (IOU)
    Dice Similarity Coefficient (DSC)
    Precision
    Senstivity
    Specificity
    Pixel Accuracy/ Rand index
    Cohen Kappa
    """

    def __init__(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        self.__true_mask = true_mask
        self.__pred_mask = pred_mask

        self.__total_pixels = true_mask.size
        self.__intersection = np.logical_and(self.__true_mask, self.__pred_mask)

        self.__TP = np.sum(self.__intersection)
        self.__FP = np.sum(self.__pred_mask) - self.__TP
        self.__FN = np.sum(self.__true_mask) - self.__TP
        self.__TN = self.__total_pixels - (self.__TP + self.__FP + self.__FN)

        # self.__TP = np.sum((self.__true_mask != 0) & (self.__pred_mask != 0))
        # self.__FP = np.sum((self.__true_mask == 0) & (self.__pred_mask != 0))
        # self.__TN = np.sum((self.__true_mask == 0) & (self.__pred_mask == 0))
        # self.__FN = np.sum((self.__true_mask != 0) & (self.__pred_mask == 0))

    def iou(self, ) -> float:
        """
        iou() -> float:
        Calculates Intersection over Union (IoU) score.
        Returns:
            float: IoU score rounded to 3 decimal places.
        """
        return round(self.__TP / (self.__TP + self.__FP + self.__FN), 3)

    def dsc(self, ) -> float:
        """
        Calculates Dice Similarity Coefficient (DSC).
        Returns:
            float: DSC rounded to 3 decimal places.
        """
        return round(2 * self.__TP / (2 * self.__TP + self.__FP + self.__FN), 3)
    
    def precision(self, ) -> float:
        """
        precision() -> float:
        Calculates Precision.
        Returns:
            float: Precision rounded to 3 decimal places.
        """
        return round(self.__TP / (self.__TP + self.__FP), 3)

    def senstivity(self, ) -> float:
        """
        senstivity() -> float:
        Calculates Sensitivity/ Recall (Recall).
        Returns:
            float: Sensitivity/ Recall rounded to 3 decimal places.
        """
        return round(self.__TP / (self.__TP + self.__FN), 3)

    def specificity(self, ) -> float:  
        """
        specificity() -> float:
        Calculates Specificity.
        Returns:
            float: Specificity rounded to 3 decimal places.
        """     
        return round(self.__TN / (self.__FP + self.__TN), 3)

    def randindex(self, ) -> float:
        """
        randindex() -> float:
        Calculates Rand Index.
        Returns:
            float: Rand Index rounded to 3 decimal places.
        """
        return round((self.__TP + self.__TN) / (self.__TP + self.__TN + self.__FN + self.__FP), 3)

    def cohenkappa(self, ) -> float:
        """
        cohenkappa() -> float:
        Calculates Cohen's Kappa coefficient.
        Returns:
            float: Cohen's Kappa coefficient rounded to 3 decimal places.
        """
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
