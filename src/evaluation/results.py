import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

'''
Adapted from https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
'''

def plot_confusion_matrix(y_true, y_pred, class_names, target_classes=[]):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       y_true (iterables): Iterable of True labels
       y_pred (iterables): Interable of Predicted labels
       class_names (array, shape = [n]): String names of the integer classes
    """
    assert len(y_true) == len(y_pred)
    
    if target_classes:
        target_indices = [class_names.index(clx) for clx in target_classes]
        y_true[np.in1d(y_true, target_indices)] = -1
        y_true[y_true >= 0] = 1
        y_true[y_true < 0] = 0
        y_pred[np.in1d(y_pred, target_indices)] = -1
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = 0
        class_names = [str(target_classes), 'others']

    cm = confusion_matrix(y_true, y_pred,labels=range(len(class_names)))


    
    figure = plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix (Recall rate)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, size=16)
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure




