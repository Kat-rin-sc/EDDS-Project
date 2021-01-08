import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

# load result files for further processing

Path = "../datasets/result/"
# results for full STACP algorithm:
df_100 = pd.read_csv(Path+"result_top_20_100.txt", sep = "\t", header = None)
df_100.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_80 = pd.read_csv(Path+"result_top_20_80.txt", sep = "\t", header = None)
df_80.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_60 = pd.read_csv(Path+"result_top_20_60.txt", sep = "\t", header = None)
df_60.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_40 = pd.read_csv(Path+"result_top_20_40.txt", sep = "\t", header = None)
df_40.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]

# results for STACP algorithm with no Context information:
df_100_noctx = pd.read_csv(Path+"result_top_20_100_noctx.txt", sep = "\t", header = None)
df_100_noctx.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_80_noctx = pd.read_csv(Path+"result_top_20_80_noctx.txt", sep = "\t", header = None)
df_80_noctx.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_60_noctx = pd.read_csv(Path+"result_top_20_60_noctx.txt", sep = "\t", header = None)
df_60_noctx.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_40_noctx = pd.read_csv(Path+"result_top_20_40_noctx.txt", sep = "\t", header = None)
df_40_noctx.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]

# results for STACP algorithm with no temporal information:
df_100_noTC = pd.read_csv(Path+"result_top_20_100_noTC.txt", sep = "\t", header = None)
df_100_noTC.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_80_noTC = pd.read_csv(Path+"result_top_20_80_noTC.txt", sep = "\t", header = None)
df_80_noTC.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_60_noTC = pd.read_csv(Path+"result_top_20_60_noTC.txt", sep = "\t", header = None)
df_60_noTC.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_40_noTC = pd.read_csv(Path+"result_top_20_40_noTC.txt", sep = "\t", header = None)
df_40_noTC.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]

# results for LRT algorithm:
df_100_lrt = pd.read_csv(Path+"lrt_top_20_100.txt", sep = "\t", header = None)
df_100_lrt.columns = ["cnt", "uid", "prec", "rec", "ndcg"]
df_80_lrt = pd.read_csv(Path+"lrt_top_20_80.txt", sep = "\t", header = None)
df_80_lrt.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_60_lrt = pd.read_csv(Path+"lrt_top_20_60.txt", sep = "\t", header = None)
df_60_lrt.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]
df_40_lrt = pd.read_csv(Path+"lrt_top_20_40.txt", sep = "\t", header = None)
df_40_lrt.columns = ["cnt", "uid", "prec", "rec", "ndcg", "map"]

# create talbe with mean values of performance metrics
ex2 = pd.DataFrame(columns = ["visited","STACP-nDCG", "STACP-map", "STACP-rec", "STACP-prec","STACP-noTC-nDCG", "STACP-noTC-map", "STACP-noTC-rec", "STACP-noTC-prec","STACP-noCTX-nDCG", "STACP-noCTX-map", "STACP-noCTX-rec", "STACP-noCTX-prec","LRT-nDCG", "LRT-map", "LRT-rec", "LRT-prec"]) 
ex2 = ex2.append({'visited':40,'STACP-nDCG':df_40["ndcg"].mean(),  'STACP-map':df_40["map"].mean(),  'STACP-rec':df_40["rec"].mean(),  'STACP-prec':df_40["prec"].mean(),'STACP-noCTX-nDCG':df_40_noctx["ndcg"].mean(),  'STACP-noCTX-map':df_40["map"].mean(),  'STACP-noCTX-rec':df_40["rec"].mean(),  'STACP-noCTX-prec':df_40["prec"].mean(),'STACP-noTC-nDCG':df_40_noTC["ndcg"].mean(),  'STACP-noTC-map':df_40["map"].mean(),  'STACP-noTC-rec':df_40["rec"].mean(),  'STACP-noTC-prec':df_40["prec"].mean(),'LRT-nDCG':df_40_lrt["ndcg"].mean(),  'LRT-map':df_40_lrt["map"].mean(),  'LRT-rec':df_40_lrt["rec"].mean(),  'LRT-prec':df_40_lrt["prec"].mean()}, ignore_index=True)
ex2 = ex2.append({'visited':60,'STACP-nDCG':df_60["ndcg"].mean(),  'STACP-map':df_60["map"].mean(),  'STACP-rec':df_60["rec"].mean(),  'STACP-prec':df_60["prec"].mean(),'STACP-noCTX-nDCG':df_60_noctx["ndcg"].mean(),  'STACP-noCTX-map':df_60["map"].mean(),  'STACP-noCTX-rec':df_60["rec"].mean(),  'STACP-noCTX-prec':df_60["prec"].mean(),'STACP-noTC-nDCG':df_60_noTC["ndcg"].mean(),  'STACP-noTC-map':df_60["map"].mean(),  'STACP-noTC-rec':df_60["rec"].mean(),  'STACP-noTC-prec':df_60["prec"].mean(),'LRT-nDCG':df_60_lrt["ndcg"].mean(),  'LRT-map':df_60_lrt["map"].mean(),  'LRT-rec':df_60_lrt["rec"].mean(),  'LRT-prec':df_60_lrt["prec"].mean()}, ignore_index=True)
ex2 = ex2.append({'visited':80,'STACP-nDCG':df_80["ndcg"].mean(),  'STACP-map':df_80["map"].mean(),  'STACP-rec':df_80["rec"].mean(),  'STACP-prec':df_80["prec"].mean(),'STACP-noCTX-nDCG':df_80_noctx["ndcg"].mean(),  'STACP-noCTX-map':df_80["map"].mean(),  'STACP-noCTX-rec':df_80["rec"].mean(),  'STACP-noCTX-prec':df_80["prec"].mean(),'STACP-noTC-nDCG':df_80_noTC["ndcg"].mean(),  'STACP-noTC-map':df_80["map"].mean(),  'STACP-noTC-rec':df_80["rec"].mean(),  'STACP-noTC-prec':df_80["prec"].mean(),'LRT-nDCG':df_80_lrt["ndcg"].mean(),  'LRT-map':df_80_lrt["map"].mean(),  'LRT-rec':df_80_lrt["rec"].mean(),  'LRT-prec':df_80_lrt["prec"].mean()}, ignore_index=True)
ex2 = ex2.append({'visited':100,'STACP-nDCG':df_100["ndcg"].mean(),  'STACP-map':df_100["map"].mean(),  'STACP-rec':df_100["rec"].mean(),  'STACP-prec':df_100["prec"].mean(),'STACP-noCTX-nDCG':df_100_noctx["ndcg"].mean(),  'STACP-noCTX-rec':df_100["rec"].mean(),'STACP-noCTX-map':df_100["map"].mean(),   'STACP-noCTX-prec':df_100["prec"].mean(),'LRT-nDCG':df_100_lrt["ndcg"].mean(), 'LRT-rec':df_100_lrt["rec"].mean(),  'LRT-prec':df_100_lrt["prec"].mean(),'STACP-noTC-nDCG':df_100_noTC["ndcg"].mean(),  'STACP-noTC-map':df_100["map"].mean(),  'STACP-noTC-rec':df_100["rec"].mean(),  'STACP-noTC-prec':df_100["prec"].mean()}, ignore_index=True)

# create data 
x = np.arange(4) 
y1 = ex2["STACP-nDCG"]
y2 = ex2["STACP-noTC-nDCG"]
y3 = ex2["STACP-noCTX-nDCG"]
y4 = ex2["LRT-nDCG"]
width = 0.1
  
# plot data in grouped manner of bar type 
plt.figure(figsize=(7,5));
plt.bar(x-0.15, y4, width, color='red') 
plt.bar(x+0.05, y2, width, color='orange') 
plt.bar(x-0.05, y3, width, color='green') 
plt.bar(x+0.15, y1, width, color='cyan') 

#plt.bar(x+0.15, y4, width, color='red') 

plt.xticks(x, ['40', '60','80', '100']) 
plt.legend(["LRT", "STACP-noTC", "STACP-noCTX", "STACP"], bbox_to_anchor=(1, 1.26),borderaxespad=0) 
plt.xlabel("Percentage of POIs each user has visited") 
plt.ylabel("nDCG@20") 
plt.grid(axis = 'y')
plt.title("nDCG of different algorithms on sparcity levels", loc = "left")
plt.show() 