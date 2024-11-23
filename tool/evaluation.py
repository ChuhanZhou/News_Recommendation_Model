from sklearn.metrics import roc_auc_score

def auc_score(true_list,scores_list):
    auc = roc_auc_score(true_list, scores_list)
    return auc

def list_auc_score(true_list,scores_list):
    auc = 0
    for i in range(len(true_list)):
        t_list = true_list[i]
        s_list = scores_list[i]
        auc += auc_score(t_list,s_list)
    auc = auc/len(true_list)
    return auc

def true_positive_rate(true_list):
    return sum(true_list)/len(true_list)