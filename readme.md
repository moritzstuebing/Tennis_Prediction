# Tennis Prediction

## Summary

This project predicts the outcome of Novak Djokovic's matches across his career. We ask whether engineered features such as form, surface, age, series and more can add any additional predictive power beyond what the odds themselves provide. They don't: across logistic regression, neural networks and XGBoost, odds-only models achieve the same or better binary cross-entropy than models built on the full dataset. The best model overall was XGBoost on odds alone (BCE 0.3656). We theorise this is due to odds already using the same publicly-available information that our engineered features are based on. Thus, adding in these features does not introduce new signal.

All models are evaluated on the same chronological test set (the final ~20% of matches, n = 248). Each cell shows **test Binary Cross-Entropy (BCE) / test accuracy**; lower BCE is better.

| Model | Odds only | No odds | All features |
|---|---|---|---|
| Logistic regression | 0.3870 / 85.48% | 0.4264 / 82.66% | 0.4102 / 84.68% |
| Neural network (1×32 hidden, ReLU) | 0.3846 / 85.48% | 0.4062 / 83.87% | 0.4073 / 85.08% |
| Random forest | 0.6821 / 82.66% | 0.4638 / 81.45% | 0.4268 / 84.27% |
| XGBoost | **0.3656** / 84.68% | 0.4113 / 83.47% | 0.3733 / 85.08% |
| *Base-rate baseline* | *0.4419 / 83.87%* | *0.4419 / 83.87%* | *0.4419 / 83.87%* |

The base-rate baseline row shows the results that would be achieved by a model predicting the training set win rate on the test set. We can see that almost all models managed to beat this baseline, indicating that they are truly learning some patterns in the data to extract signal, rather than giving these simple predictions. 

## Datasets
We use the ATP dataset found at https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull. The matches in this dataset are in chronological order. Also note that the dataset is frequently updated with new matches. The dataset we used had a date range from 3 January 2000 until 12 July 2026, concluding with the Sinner vs Zverev Wimbledon 2026 final. We use this dataset to form our Djokovic-specific datasets, which after cleaning contained 1250 matches.

Many features are already included in the dataset. We wrangled these features into forms that a model can interpret, such as creating a binary feature indicating whether Djokovic won or not. We also engineered some new features such as a 5-match win rate and quadratic age term.

The full dataset includes the original ATP data (modified for our purposes) as well as the newly-engineered features. The odds-only dataset includes only Djokovic's and the opponent's odds before the match. The no odds dataset includes all of the features in the ATP dataset apart from the odds data, as well as our self-engineered features.

## Methodology Choices

#### Time-Series
This is time-series data, so careful considerations had to be made to prevent temporal leakage and to ensure consistency with how such models would be used in production. Firstly, we opted for chronological train-test splits. Temporal datasets have similar patterns and entries between nearby datapoints. A randomised split would mean patterns from all eras are available in the train and test set. Thus, the model would have encountered similar patterns to those in the test set before, which would be a source of temporal data leakage. In particular, there would have been cases where one match was in the test set, but both matches on either side of it were in the train set. This would mean the model could directly interpolate between these datapoints and easily predict correctly. Both of these issues would have made the test performance artificially good. Furthermore, a chronological split mimics how such models would be used in production, where they would be trained on existing match data to predict future outcomes. For similar reasons, we use a expanding window cross-validation for any hyperparameter tuning.

#### Evaluation Metrics
Accuracy is misleading in this context. In this dataset, Djokovic wins about 84% of the matches. This class imbalance means that a model predicting a win for every match, which has clearly not learned to recognise any actual signal, would thus score approximately 84% accuracy. This seems decent, but is clearly undesirable since such a model has made no attempt to learn the patterns. 

Thus, although we do check accuracy scores for all models, comparing them to the baseline accuracy, we are more interested in the binary cross-entropy, which combines the accuracy of the model with calibration of the prediction probabilities.

#### Splits
We used the exact same splits across all models, to ensure fair comparison between models.

## Limitations

#### Some Circularity
We recognise that odds are essentially market-produced probabilities of the player winning. Thus, odds-only models will learn to reproduce the odds. Our interest here is whether adding extra features can improve on the odds, which will help us understand whether publicly-available information is truly used in generation of the odds.

#### Loss Signal

Djokovic has had a very successful career, and most of his matches result in wins. Many of his losses have been upsets where he has been the favourite, and the features would also point towards him winning the match. These upsets are largely unpredictable from our features, and thus trying to extract loss signal from datapoints that look like wins is very difficult. As a result, the loss recall was low for all models (for example 0.1750 for the XGBoost full dataset model). To effectively extract loss signal, it may be better to focus on a player with more frequent losses where they genuinely weren't the favourite.

## Repository Structure

```
Tennis_Prediction/
├── Data/
│   ├── atp_tennis.csv               # Full ATP dataset (Kaggle)
│   ├── djokovic_data.csv            # Djokovic's matches, cleaned (00_cleaning)
│   ├── full_data.csv                # All features (01_feature_engineering)
│   ├── full_data_enc.csv            # All features, one-hot encoded
│   ├── odds_data.csv                # Odds-only features
│   ├── odds_data_enc.csv            # Odds-only features, one-hot encoded
│   ├── nodds_data.csv               # All features except odds
│   └── nodds_data_enc.csv           # All features except odds, one-hot encoded
├── Notebooks/
│   ├── 00_cleaning.ipynb
│   ├── 01_feature_engineering.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_linear_odds.ipynb
│   ├── 04_log_reg.ipynb
│   ├── 05_neural_network.ipynb
│   ├── 06_random_forests.ipynb
│   ├── 07_xgboost.ipynb
│   └── utils.py                     # Shared helpers (chronological splits, folds, metrics)
├── requirements.txt
└── README.md
```

## Future Work

#### Paired Bootstrap
The test set is relatively small (248 matches). We want to understand if the models actually have significantly different performances or whether a different test set would change the results. For example, in the XGBoost notebook, we found the full model has a binary cross-entropy of 0.3733 while the odds only model has a binary cross-entropy of 0.3656. This is a small difference, and a different test set may give a different result. A paired bootstrap would help us understand if we can truly tell the models apart.

#### Betting Simulation
Use an odds-only model and a chosen betting strategy to simulate the outcome of placing bets based on model outcomes.

## Tools
Python, Jupyter, NumPy, pandas, matplotlib, scikit-learn, PyTorch, XGBoost

## How To Run
`pip install -r requirements.txt` then run the notebooks in order.
