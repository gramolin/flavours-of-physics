# Kaggle's Flavours of Physics: the second-ranked solution

This is a solution ranked second on the [Private Leaderboard](https://www.kaggle.com/c/flavours-of-physics/leaderboard) of the Kaggle ["Flavours of Physics: Finding τ → μμμ"](https://www.kaggle.com/c/flavours-of-physics) competition. The model is based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and implemented in Python with the help of the [XGBoost](https://github.com/dmlc/xgboost) library. It is simply a blend of two XGBoost classifiers (boosters) trained on different sets of features. The first booster is an ensemble of 200 [decision trees](https://en.wikipedia.org/wiki/Decision_tree) targeting mostly geometric features (such as impact parameters and track isolation variables). The second booster consists of 100 trees trained on purely kinematic features. Blending two independent classifiers together allows us to easily pass the [correlation test](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test). To pass the [agreement test](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test), the only thing needed is to exclude SPDhits from the features used in the training process.

## Dependencies
* [XGBoost](https://github.com/dmlc/xgboost) library should be installed
* The standard Python packages **numpy**, **pandas**, and **csv** are required
* The training and test datasets (the files **training.csv** and **test.csv**) can be downloaded from [here](https://www.kaggle.com/c/flavours-of-physics/data)

## How to generate the solution
 1. Put the data files **training.csv** and **test.csv** in the **data** directory.
 2. To train the XGBoost classifiers, run **python train.py**. The trained boosters will be saved in the files **bst1.model** and **bst2.model**, so you can make predictions on new datasets without re-training the model.
 3. To make a prediction, run **python predict.py**. Results will be written to **submission.csv**.

