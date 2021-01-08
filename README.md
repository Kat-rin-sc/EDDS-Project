# EDDS-Project
## Reproduction of paper results - Joint Geographicaland Temporal Modeling Based on MatrixFactorization for Point-of-Interest Recommendation
### Katrin Schreiberhuber, Mariusz Karpowicz, Jakub Ciemięga

##### Contact details
e12005481@student.tuwien.ac.at (Jakub Ciemięga)
e1503964@tuwien.ac.at (Katrin Schreiberhuber

##### Environment settings
as a custom library was implemented in Python 2, please use Python 2.7 to execute this code.
- Python version: '2.7'
- Install the required libraries

##### To run the code for the STACP and create the result files:

- Open and run STACP/recommendation_stacp.py
- When the execution is finished, the results of the run will be saved in the folder STACP/datasets/result/xxx.txt 
- You have to change the name you want to save the results as in the recommendation_stacp.py file in line 104-110 accordingly!
- To change to STACP_NoTC go to line 196 and change lambda = 1
- To change to STACP_NoCTX go to line ....

##### To run the code for the LRT algorithm and create the result files:
- Open and run  LRT/recommendation_LRT.py
- When the execution is finished, the results of the run will be saved in the folder STACP/datasets/result/xxx.txt 
- You have to change the name you want to save the results as in the recommendation_stacp.py file in line 104-110 accordingly!

##### To calculate the metrics  of experiment 1 and perform the two-tailed paired t-tests:
- all result files have to be in datasets/result/
- run STACP/STACP/tests.py

##### To run Experiment 2, comparing performance on data sparsity levels:
- create new training data by executing /Analysis/generate_new_train_data.py (you can skip this step as the datasets are already included in the right folder)
- the new data will be saved to /datasets/gowalla_u5628/
- run the jupyter notebook in /Analysis/Experiment_2.ipynb to generate the results for experiment 2.


##### and so on...
