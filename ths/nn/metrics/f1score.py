from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def tprate(y_true, y_pred):
    return recall(y_true, y_pred)

def fprate2(y_true, y_pred):
    #invert true and negative so negative is 1  and true is 1.
    y_true = 1 - y_true
    #invert predictions so that we get 1 for the predictions originally set to 0
    y_pred = 1 - y_pred
    return recall(y_true, y_pred)

def fprate(y_true, y_pred):
    # true positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = true_positives + false_positives
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # false_positives
    false_positive = predicted_positives - true_positives
    # Now work on negatives
    y_false = 1 - y_true
    possible_negatives = K.sum(K.round(K.clip(y_false, 0, 1)))
    fprate = false_positive / (possible_negatives  + K.epsilon())
    return fprate