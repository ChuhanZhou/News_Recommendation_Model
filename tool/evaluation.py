from sklearn.metrics import roc_auc_score

def auc_score(true_list,true_scores_list):
    auc = roc_auc_score(true_list, true_scores_list)
    return auc

def true_positive_rate(true_list):
    return sum(true_list)/len(true_list)