# EDDS-Project
## Reproduction of paper results - Joint Geographicaland Temporal Modeling Based on MatrixFactorization for Point-of-Interest Recommendation
### Katrin Schreiberhuber, Mariusz Karpowicz, Jakub Ciemięga

#### Contact details
e12005481@student.tuwien.ac.at (Jakub Ciemięga)

e1503964@tuwien.ac.at (Katrin Schreiberhuber)

e12006880@student.tuwien.ac.at (Mariusz Karpowicz)

#### Environment settings
- Python version: '2.7'
- Install the required libraries

#### To run the code for the STACP and create the result files:

- Open and run STACP/recommendation_stacp.py
- When the execution is finished, the results of the run will be saved in the folder STACP/datasets/result/xxx.txt 
- **Changes to be considered before running the code:**
    - which training file to use for training the model (check line 182)
    - which version of the algorithm to use
        - To change to STACP_NoTC go to line 196 and change lambda = 1
        - To change to STACP_NoCTX go to line 128 and comment out part of the equation according to a comment above
    - under which name to save the result files of the run (check lines 104-110)

#### To run the code for the LRT algorithm and create the result files:
- Open and run  LRT/recommendation_LRT.py
- When the execution is finished, the results of the run will be saved in the folder STACP/datasets/result/xxx.txt 
- You have to change the name you want to save the results as in the recommendation_stacp.py file in line 104-110 accordingly!
- change the train_file used in the algorithm if this is what you want to analyse.

#### To calculate the metrics  of experiment 1 and perform the two-tailed paired t-tests:
- all result files have to be in datasets/result/
- run Analysis/tests.py

#### To run Experiment 2, comparing performance on data sparsity levels:
- create new training data by executing /Analysis/generate_new_train_data.py (you can skip this step as the datasets are already included in the right folder)
- the new data will be saved to /datasets/gowalla_u5628/
- run the jupyter notebook in /Analysis/Experiment_2.ipynb to generate the results for experiment 2.


##### To analyze Experiment 3 parameters:
- To generate plots of recall and precision from parameters alpha, d, lambda take data from /datasets/parameters.txt
- Run /Analysis/Experiment-3-parameters-calculation.ipynb to generate plots
- To generate results for parameters.txt for different parameters, they need to be changed in stacp/recommendation_stacp.py in lines 191-195
