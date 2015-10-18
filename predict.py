# https://github.com/gramolin/flavours-of-physics

import csv
import pandas as pd
import xgboost as xgb

import features
import parameters

# Load the first booster:
bst1 = xgb.Booster()
bst1.load_model("bst1.model")

# Load the second booster:
bst2 = xgb.Booster()
bst2.load_model("bst2.model")

# Create a submission file:
with open('submission.csv', 'w') as csvfile:
  csv.writer(csvfile, delimiter=',').writerow(['id', 'prediction'])

# Prediction and output:
for chunk in pd.read_csv("data/test.csv", index_col='id', chunksize=100000):
  # Add extra features:
  chunk = features.add_features(chunk)
  
  # Predict probabilities:
  probs1 = bst1.predict(xgb.DMatrix(chunk[features.list1])) # Booster 1
  probs2 = bst2.predict(xgb.DMatrix(chunk[features.list2])) # Booster 2
  
  # Weighted average of the predictions:
  result = pd.DataFrame({'id': chunk.index})
  result['prediction'] = 0.5*(parameters.w1*probs1 + (1 - parameters.w1)*probs2)
  
  # Write to the submission file:
  result.to_csv('submission.csv', index=False, header=False, sep=',', mode='a')
