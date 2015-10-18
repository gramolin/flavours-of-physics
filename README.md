# Kaggle's Flavours of Physics: the second-ranked solution

This is a solution ranked second on the [Private Leaderboard](https://www.kaggle.com/c/flavours-of-physics/leaderboard) of the Kaggle ["Flavours of Physics: Finding τ → μμμ"](https://www.kaggle.com/c/flavours-of-physics) competition. The model is based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and implemented in Python with the help of the [XGBoost](https://github.com/dmlc/xgboost) library. It is simply a combination of two XGBoost classifiers (boosters) trained on different sets of features. The first booster is an ensemble of 200 [decision trees](https://en.wikipedia.org/wiki/Decision_tree) targeting mostly geometric features (such as impact parameters and track isolation variables). The second booster consists of 100 trees trained on purely kinematic features. Final prediction is a weighted average of the probabilities predicted by the individual classifiers (with a weight of 0.78 assigned to the first booster). Combining two independent classifiers allows us to easily pass the [correlation test](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test). To pass the [agreement test](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test), the only thing needed is to exclude *SPDhits* from the features used in the training process.

## Dependencies

* The [XGBoost](https://github.com/dmlc/xgboost) library should be installed
* The standard Python packages **numpy**, **pandas**, and **csv** are required
* The training and test datasets (the files **training.csv** and **test.csv**) can be downloaded from [here](https://www.kaggle.com/c/flavours-of-physics/data)

## How to generate the solution

 1. Put the data files **training.csv** and **test.csv** in the **data** directory.
 2. To train the XGBoost classifiers, run **python train.py**. The trained boosters will be saved in the files **bst1.model** and **bst2.model**, so you can make predictions on new datasets without re-training the model.
 3. To make a prediction, run **python predict.py**. Results will be written to **submission.csv**.

## Feature engineering

Some new features were designed in addition to the original ones. The original feature *SPDhits* was not used since it prevents passing the agreement test. Lists of the features used to train each booster are provided below.

### Features for the first booster

* **Original features:** *FlightDistance*, *FlightDistanceError*, *LifeTime*, *IP*, *IPSig*, *VertexChi2*, *dira*, *pt*, *DOCAone*, *DOCAtwo*, *DOCAthree*, *IP_p0p2*, *IP_p1p2*, *isolationa*, *isolationb*, *isolationc*, *isolationd*, *isolatione*, *isolationf*, *iso*, *CDF1*, *CDF2*, *CDF3*, *ISO_SumBDT*, *p0_IsoBDT*, *p1_IsoBDT*, *p2_IsoBDT*, *p0_track_Chi2Dof*, *p1_track_Chi2Dof*, *p2_track_Chi2Dof*, *p0_IP*, *p0_IPSig*, *p1_IP*, *p1_IPSig*, *p2_IP*, *p2_IPSig*.

* **New features:**
  * __*E*__ is the full energy of the mother particle calculated assuming that the final-state particles p0, p1, and p2 are muons (*E* = *E0* + *E1* + *E2*).
  * __*FlightDistanceSig*__ is the ratio (*FlightDistance* / *FlightDistanceError*).
  * __*DOCA_sum*__ is the sum (*DOCAone* + *DOCAtwo* + *DOCAthree*).
  * __*isolation_sum*__ is the sum (*isolationa* + *isolationb* + *isolationc* + *isolationd* + *isolatione* + *isolationf*).
  * __*IsoBDT_sum*__ is the sum (*p0_IsoBDT* + *p1_IsoBDT* + *p2_IsoBDT*).
  * __*track_Chi2Dof*__ is calculated as sqrt[(*p0_track_Chi2Dof* – 1)^2 + (*p1_track_Chi2Dof* – 1)^2 + (*p2_track_Chi2Dof* – 1)^2].
  * __*IP_sum*__ is the sum (*p0_IP* + *p1_IP* + *p2_IP*).
  * __*IPSig_sum*__ is the sum (*p0_IPSig* + *p1_IPSig* + *p2_IPSig*).
  * __*CDF_sum*__ is the sum (*CDF1* + *CDF2* + *CDF3*).

### Features for the second booster

* **Original features:** *dira*, *pt*, *p0_pt*, *p0_p*, *p0_eta*, *p1_pt*, *p1_p*, *p1_eta*, *p2_pt*, *p2_p*, *p2_eta*.

* **New features:**
  * __*E*__ is the full energy of the mother particle calculated assuming that the final-state particles p0, p1, and p2 are muons (*E* = *E0* + *E1* + *E2*).
  * __*pz*__ is the longitudinal momentum of the mother particle.
  * __*beta*__ is the relativistic beta of the mother particle (beta = v / c).
  * __*gamma*__ is the relativistic gamma of the mother particle (gamma = 1 / sqrt(1 – beta^2)).
  * __*beta_gamma*__ is *beta*×*gamma* calculated as *FlightDistance* / (*LifeTime*×*c*), where *c* is the speed of light.
  * __*Delta_E*__ is the difference between energies of the mother particle calculated in two different ways.
  * __*Delta_M*__ is the difference between masses of the mother particle calculated in two different ways.
  * __*flag_M*__ equals to 1 if the mass of the mother particle is close to the tau mass; equals to 0 otherwise. 
  * __*E0*__ is the full energy of the particle p0 calculated as *E0* = sqrt[(*m_mu*)^2 + (*p0_p*)^2], where *m_mu* is the muon mass.
  * __*E1*__ is the full energy of the particle p1 calculated as *E1* = sqrt[(*m_mu*)^2 + (*p1_p*)^2], where *m_mu* is the muon mass.
  * __*E2*__ is the full energy of the particle p2 calculated as *E2* = sqrt[(*m_mu*)^2 + (*p2_p*)^2], where *m_mu* is the muon mass.
  * __*E0_ratio*__ is the ratio (*E0* / *E*).
  * __*E1_ratio*__ is the ratio (*E1* / *E*).
  * __*E2_ratio*__ is the ratio (*E2* / *E*).
  * __*p0_pt_ratio*__ is the ratio (*p0_pt* / *pt*).
  * __*p1_pt_ratio*__ is the ratio (*p1_pt* / *pt*).
  * __*p2_pt_ratio*__ is the ratio (*p2_pt* / *pt*).
  * __*eta_01*__ is the difference (*p0_eta* – *p1_eta*).
  * __*eta_02*__ is the difference (*p0_eta* – *p2_eta*).
  * __*eta_12*__ is the difference (*p1_eta* – *p2_eta*).
  * __*t_coll*__ is calculated as (*p0_pt* + *p1_pt* + *p2_pt*) / *pt* (this equals to unity if the final-state particles p0, p1, and p2 are collinear in the transverse plane).

