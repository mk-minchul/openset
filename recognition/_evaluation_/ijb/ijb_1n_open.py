import numpy as np
import torch


def inner_product(x1, x2):
    if isinstance(x1[0], torch.Tensor):
        x1, x2 = torch.stack(x1, dim=0), torch.stack(x2, dim=0)
        if x1.ndim == 3:
            x1, x2 = x1[:,:,0], x2[:,:,0]
        return torch.matmul(x1, x2.T).cpu().numpy()
    else:
        x1, x2 = np.array(x1), np.array(x2)
        if x1.ndim == 3:
            x1, x2 = x1[:,:,0], x2[:,:,0]
        return np.dot(x1, x2.T)


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_retrievals=False):
    ''' Closed/Open-set Identification. 
        A general case of Cummulative Match Characteristic (CMC) 
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_retrievals:       not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks, 
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape==label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    mate_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[mate_indices,:]
    label_mat_m = label_mat[mate_indices,:]
    score_mat_nm = score_mat[np.logical_not(mate_indices),:]
    label_mat_nm = label_mat[np.logical_not(mate_indices),:]
    mate_indices = np.argwhere(mate_indices).flatten()

    # print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as thrnp.vstack((eshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)[:,::-1]
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    sorted_score_mat_m = score_mat_m.copy()
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]
        sorted_score_mat_m[row,:] = score_mat_m[row, sort_idx]
        
    # Calculate DIRs for different FARs and ranks
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    success = np.ndarray((len(FARs), len(ranks)), dtype=np.object_)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:,0:rank].any(axis=1)
            DIRs[i,j] = (score_rank & retrieval_rank).astype(np.float32).mean()
            if get_retrievals:
                success[i,j] = (score_rank & retrieval_rank)
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()
        success = success.flatten()

    if get_retrievals:
        return DIRs, FARs, thresholds, mate_indices, success, sort_idx_mat_m, sorted_score_mat_m

    return DIRs, FARs, thresholds


def identification_v2(feature_probe, label_probe, 
                      feature_gallery1, label_gallery1, 
                      feature_gallery2, label_gallery2,
                      feature_gallery1_per_media, feature_gallery2_per_media,
                      k=5, use_knn=True):


    scores1 = inner_product(feature_probe, feature_gallery1)
    scores2 = inner_product(feature_probe, feature_gallery2)

    if use_knn:
        scores1_per_media = inner_product(feature_probe, feature_gallery1_per_media)
        scores2_per_media = inner_product(feature_probe, feature_gallery2_per_media)
        
        knn_scores1 = np.sort(scores1_per_media, axis=1)[:, -k]
        knn_scores2 = np.sort(scores2_per_media, axis=1)[:, -k]

        # add KNN scores
        top_k_indices1 = np.argsort(scores1, axis=1)[:, -1:]
        top_k_indices2 = np.argsort(scores2, axis=1)[:, -1:]
        for i in range(len(feature_probe)):
            scores1[i, top_k_indices1[i]] += knn_scores1[i]
            scores2[i, top_k_indices2[i]] += knn_scores2[i]

    # Close-set
    label_mat = (label_probe==np.vstack((label_gallery1, label_gallery2))).T
    DIRs_closeset, _, _ = DIR_FAR(np.hstack((scores1, scores2)), label_mat, ranks=[1, 5, 10])
    
    # Open-set
    DIRs_openset1, _, _ = DIR_FAR(scores1, 
        (label_probe == label_gallery1).T, FARs=[0.01, 0.05])
    DIRs_openset2, _, _ = DIR_FAR(scores2,
        (label_probe == label_gallery2).T, FARs=[0.01, 0.05])
    DIRs_openset = (DIRs_openset1 + DIRs_openset2) / 2.0
    
    return DIRs_closeset, DIRs_openset
    


def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        epsilon = 1e-5
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = (num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds
