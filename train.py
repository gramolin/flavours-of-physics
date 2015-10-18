# https://github.com/gramolin/flavours-of-physics

import csv
import pandas as pd
import xgboost as xgb

import features
import parameters

# Read the training set:
train = pd.read_csv('data/training.csv', index_col='id')
train = train[train['min_ANNmuon'] > 0.4]

# Add extra features:
train = features.add_features(train)

# Train the first XGBoost booster:
bst1 = xgb.train(parameters.params1, xgb.DMatrix(train[features.list1], train['signal']), parameters.num_trees1)
bst1.save_model('bst1.model')

# Train the second XGBoost booster:
bst2 = xgb.train(parameters.params2, xgb.DMatrix(train[features.list2], train['signal']), parameters.num_trees2)
bst2.save_model('bst2.model')
