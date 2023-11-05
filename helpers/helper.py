import tensorflow as tf

class Metrics:
    def iou(self, true_mask, pred_mask) -> float:
        intersect = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        union = tf.reduce_sum(pred_mask) + tf.reduce_sum(true_mask) - intersect
        iou = intersect / union
        return iou
    
    def iou_loss(self, true_mask, pred_mask):
        iou_score = self.iou(true_mask, pred_mask)
        return 1 - iou_score

    def dsc(self, true_mask, pred_mask) -> float:
        dsc_intersect = 2 * tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        dsc_union = 2 * tf.reduce_sum(pred_mask) + tf.reduce_sum(true_mask) - dsc_intersect
        return dsc_intersect / dsc_union
    
    def dsc_loss(self, true_mask, pred_mask) -> float:
        return 1 - self.dsc(true_mask, pred_mask)

    def precision(self, true_mask, pred_mask) -> float:
        intersect = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        total_pixel_pred = tf.reduce_sum(pred_mask)
        return intersect / total_pixel_pred

    def senstivity(self, true_mask, pred_mask) -> float:
        intersect = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        total_pixel_truth = tf.reduce_sum(true_mask)
        return intersect / total_pixel_truth
    
    def specificity(self, true_mask, pred_mask) -> float:  
        intersect = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        common = tf.reduce_sum(tf.cast(tf.equal(true_mask, pred_mask), tf.float32))
        total_pixel_pred = tf.reduce_sum(pred_mask)
        spec = (common - intersect) / (common + total_pixel_pred - 2 * intersect)
        return spec

    def randindex(self, true_mask, pred_mask) -> float:
        intersect = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        common = tf.reduce_sum(tf.cast(tf.equal(true_mask, pred_mask), tf.float32))
        total_pixel_pred = tf.reduce_sum(pred_mask)
        total_pixel_truth = tf.reduce_sum(true_mask)
        acc = common / (total_pixel_pred + total_pixel_truth + common - 2 * intersect)
        return acc

    def cohenkappa(self, true_mask, pred_mask) -> float:
        TP = tf.reduce_sum(tf.multiply(pred_mask, true_mask))
        TN = tf.reduce_sum(tf.cast(tf.equal(true_mask, pred_mask), tf.float32) - tf.reduce_sum(true_mask))
        FP = tf.reduce_sum(pred_mask) - TP
        FN = tf.reduce_sum(true_mask) - TP
        num = 2 * (TP * TN - FN * FP)
        den = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
        return num/ den

