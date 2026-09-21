# Tennis Prediction

## Summary

This repository explores the relationship between various features and performance of predictive models in classifying matches into win/loss, and predicting bookmaker odds. We take a focus on Novak Djokovic's matches. Bookmaker-generated odds are known to already be very good classification predictors for win/loss. We evaluate whether adding additional features can improve the performance of models over using just odds data. Most models do not substantially improve when adding in these features, which the notebooks show.

## Dataset
Full ATP dataset found at https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull. We use this dataset to form our Djokovic-specific datasets. Note that the matches are in chronological order here.

## Notebooks Roadmap

#### 00_cleaning
Extracts Djokovic's matches, and removes missing/incomplete data and non-useful features in the original dataset.

#### 01_feature_engineering
We engineer some new features, and convert our original features into Djokovic-specific variables that we can use for our purposes.

#### 02_eda
Performs exploratory data analysis to understand the relationships of our chosen features with our target variable, and amongst themselves.

#### 03_linear_odds
Linear regression on the bookmaker odds using our chosen features. Least squares, ridge and LASSO.

#### 04_log_reg
Logistic regression on win/loss classification. Performed on the three datasets (full, odds only, no odds).

#### 05_neural_network
Fully-connected MLP with pytorch for classification on each of the three datasets.

#### 06_random_forests
Random forest ensemble models with scikit-learn for classification on all three dataset.

## Results

All models are evaluated on the same chronological test set (the final 20% of matches, n = 248). Each cell shows **test BCE / test accuracy**; lower BCE is better.

| Model | Odds only | No odds | All features |
|---|---|---|---|
| Logistic regression | 0.3870 / 85.48% | 0.4264 / 82.66% | 0.4102 / 84.68% |
| Neural network (1×32 hidden, ReLU) | **0.3846** / 85.48% | 0.4062 / 83.87% | 0.4073 / 85.08% |
| Random forest | 0.6821 / 82.66% | 0.4638 / 81.45% | 0.4268 / 84.27% |
| *Base-rate baseline* | *0.4419 / 83.87%* | *0.4419 / 83.87%* | *0.4419 / 83.87%* |

The base-rate baseline row indicates the test BCE of a model predicting the win rate of the training set for every match, and the test accuracy of a model predicting win for every single match. Apart from the random forests models, all models do better than these baselines, indicating effective learning of signal.

The poor performance of the random forest models (worse than the baseline for both the odds only and no odds dataset) shows how random forests models are not the best choice of model for our task. In contrast to the other models, the odds-only forest performs worse than the other two models. We explain in the notebook that this is due to the small number of features meaning the trees are very similar, pushing predicted probabilities towards $0$ and $1$, which can lead to overconfidently wrong predictions and poor BCE. When more features are added, there is more capacity for the trees to disagree, and the predicted probabilities are less clustered around the extreme points, leading to better BCEs.

For the logistic regression and neural network, both models perform best on the odds-only dataset. This shows how the odds contain more explanatory power than all of our chosen features combined. In particular, using only the odds is actually also better than using the full dataset. We chalk this up to added features increasing model complexity, leading to overfitting and introducing further noise and variation, confusing the models. Increasing model complexity can decrease the bias, but only if the added features provide information that the existing features do not already encode. In our case, the odds probably already encode all of the information that our chosen features do.

To summarise, the odds themselves are the best predictor for Djokovic match classification, and the features we have chosen do not add any additional information, and also cannot recover the information that the odds themselves encode.