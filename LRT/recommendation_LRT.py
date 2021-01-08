import time
import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from lib.TimeAwareMF import TimeAwareMF
from lib.metrics import precisionk, recallk, ndcgk


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_tuples = set()
    visited_lids = defaultdict(set)
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid, = int(uid), int(lid)
        training_tuples.add((uid, lid))
        visited_lids[uid].add(lid)

    check_in_data = open(check_in_file, 'r').readlines()
    training_tuples_with_time = defaultdict(int)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if (uid, lid) in training_tuples:
            hour = time.gmtime(ctime).tm_hour
            training_tuples_with_time[(hour, uid, lid)] += 1.0

    # Default setting: time is partitioned to 24 hours.
    sparse_training_matrices = [sparse.dok_matrix((user_num, poi_num)) for _ in range(24)]
    for (hour, uid, lid), freq in training_tuples_with_time.items():
        sparse_training_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)
    return sparse_training_matrices, training_tuples, visited_lids


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    sparse_training_matrices, training_tuples, visited_lids = read_training_data()
    ground_truth = read_ground_truth()

    # max_train=30
    TAMF.train(sparse_training_matrices, max_iters=1, load_sigma=False)
    TAMF.save_model("./tmp/")
    # TAMF.load_model("./tmp/")


    all_uids = range(user_num)
    all_lids = range(poi_num)
    np.random.shuffle(all_uids)

    # Change name of file where you want to save the results here:
    rec_list = open("../datasets/result/reclist_top_" + str(top_k) + ".txt", 'w')
    result_10 = open("../datasets/result/result_top_" + str(10) + ".txt", 'w')
    result_20 = open("../datasets/result/result_top_" + str(20) + ".txt", 'w')

    precision10, recall10, ndcg10= 0,0,0
    precision20, recall20, ndcg20= 0,0,0
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            overall_scores = [TAMF.predict(uid, lid)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]
            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            precision10 = precisionk(actual, predicted[:10])
            recall10 = recallk(actual, predicted[:10])
            ndcg10 = ndcgk(actual, predicted[:10])

            precision20 = precisionk(actual, predicted[:20])
            recall20 = recallk(actual, predicted[:20])
            ndcg20 = ndcgk(actual, predicted[:20])

            rec_list.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')

            # write the different ks
            result_10.write('\t'.join([str(cnt), str(uid), str(precision10), str(recall10), str(ndcg10)]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(precision20), str(recall20), str(ndcg20)]) + '\n')

        # print(cnt, uid, "pre@10:", np.mean(precision10), "rec@10:", np.mean(recall10))
        #     result_out.write('\t'.join([
        #         str(cnt),
        #         str(uid),
        #         ','.join([str(lid) for lid in predicted])
        #     ]) + '\n')


if __name__ == '__main__':
    data_dir = "../datasets/gowalla_u5628/"

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)

    main()
