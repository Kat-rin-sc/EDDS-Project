import numpy as np
import scipy.stats as stats

# False for averages, True for statistical tests
pvalues_notavg = True


if pvalues_notavg:
    filenames10 = ['10_lrt', '10_noctx', '10_notc']
    filenames20 = ['20_100_lrt', '20_100_noctx', '20_100_notc']

    a = np.loadtxt('../datasets/result/result_top_' + '10_stacp' + '.txt')

    for filename in filenames10:
        test = np.loadtxt('../datasets/result/result_top_' + filename + '.txt')
        print(filename)
        print('%.e' % stats.ttest_rel(a[:, 2], test[:, 2])[1])
        print('%.e' % stats.ttest_rel(a[:, 3], test[:, 3])[1])
        print('%.e' % stats.ttest_rel(a[:, 4], test[:, 4])[1])
        print

    a1 = np.loadtxt('../datasets/result/result_top_' + '20_100' + '.txt')

    for filename in filenames20:
        test = np.loadtxt('../datasets/result/result_top_' + filename + '.txt')
        print(filename + ':')
        print('%.e' % stats.ttest_rel(a1[:, 2], test[:, 2])[1])
        print('%.e' % stats.ttest_rel(a1[:, 3], test[:, 3])[1])
        print('%.e' % stats.ttest_rel(a1[:, 4], test[:, 4])[1])
        print
else:
    filenames = ['10_lrt', '10_noctx', '10_notc', '10_stacp', '20_100_lrt', '20_100_noctx', '20_100_notc', '20_100']
    for filename in filenames:
        test = np.loadtxt('../datasets/result/result_top_' + filename + '.txt')
        print(filename + ':')
        print('precision: ', test[:, 2].mean().round(4))
        print('recall: ', test[:, 3].mean().round(4))
        print('ndcg: ', test[:, 4].mean().round(4))
        print