import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries(data,title, display_mode=False ,save_name=None):
    # data is a numpy array with shape (T,N)
    N = data.shape[1]
    fig, axes = plt.subplots(N, sharex=True)
    if N<2:
        axes.plot(data)
    else:
        for j in range(N):
            axes[j].plot(data[:,j])
    plt.title(title)

    if save_name is not None:
        fig.savefig(save_name)

    if display_mode:
        plt.show()
        plt.close()
    

def plot_losses(data, display_mode=False ,save_name=None):
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(data)

    if save_name is not None:
        plt.savefig(save_name)

    if display_mode:
        plt.show()
        plt.close()



def plot_recovered_graph(W_est, W, title=None, display_mode=False ,save_name=None):

    """
     Args:
        W: ground truth graph, W[i,j] means i->j.
        W_est: predicted graph, W_est[i,j] means i->j.

    """
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)

    ax1.set_title('recovered_graph')
    ax1.set_xlabel('Effects')
    ax1.set_ylabel('Causes')
    map1 = ax1.imshow(W_est, cmap='Blues', interpolation='none')
    # fig.colorbar(map1, ax=ax1)

    ax2.set_title('true_graph')
    ax2.set_xlabel('Effects')
    ax2.set_ylabel('Causes')
    map2 = ax2.imshow(W, cmap='Blues', interpolation='none')
    # fig.colorbar(map2, ax=ax2)

    
    fig.subplots_adjust(wspace=0.3)
    if title is not None:
        fig.suptitle(title)


    if save_name is not None:
        fig.savefig(save_name)
        
    if display_mode:
        plt.show()
        plt.close()


def F1(A_est, A_true, threshold):
    from sklearn.metrics import confusion_matrix
    A_est = (abs(A_est)> threshold).astype(int)

    confusion = confusion_matrix(A_true.reshape(1,-1)[0], A_est.reshape(1,-1)[0])
    
    
    tp = confusion[1,1] # W_ij=1 is positive, which is slightly different than common classification task.
    fp = confusion[0,1] 
    fn = confusion[1,0]

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    # f1 = (2*precision*recall)/(precision+recall)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



    










def count_accuracy(W_true, W_est, W_und=None):
    """
    Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        W_true: ground truth graph, W_true[i,j] means i->j.
        W_est: predicted graph, W_est[i,j] means i->j.
        W_und: predicted undirected edges in CPDAG, asymmetric

    Returns in dict:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    Referred from:
    - https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    B_true = W_true != 0
    B = W_est != 0
    B_und = None if W_und is None else W_und
    d = B.shape[0]

    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {
        'fdr': fdr,
        'tpr': tpr,
        'fpr': fpr,
        'shd': shd,
        'pred_size': pred_size
    }


