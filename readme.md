# Tennis Prediction

## Summary

This repository explores the relationship between various features and performance of predictive models in classifying matches into win/loss, and predicting bookmaker odds. Bookmaker-generated odds are known to already be very good classification predictors for win/loss. We evaluate whether adding additional features can improve the performance of models over using just odds data. Most models do not substantially improve when adding in these features, and we show this. We use the ATP tennis dataset that can be found on kaggle, extracting only Djokovic's matches

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

## Discussion

The main focus of our prediction modelling was to compare model performance when trained on odds-only data, no-odds data, and a full dataset with odds and all of our chosen features. The models we trained all performed slightly better when trained only on odds. Using models trained on the full dataset (chosen features as well as odds) perform better than models trained only on our additional features, however they do not performs better than those trained only on odds, indicating some model confusion introduced by the additional features.

This indicates that the odds are by far the most important feature in our dataset for classification, since they give a bookmaker-algorithm calculated probability of Djokovic winning the match. These are already well-calibrated and will predict most of the signal in the data, including features that are difficult to engineer like current news (e.g "he tweaked something in practice"). Hence, trying to classify while using the odds as a feature already uses the best possible classification predictor. When adding further features, we are trying to predict the signal that the bookmakers missed, which is a difficult task. Furthermore, the model learns that odds are a very good indicator for our classification and contain the most signal by far, so just learns to predict using this. Any of the other features, like age, rank, 5SMA, are given much less importance. The accuracies our model achieve are essentially just a reflection of how good the bookmaker odds are at capturing the signal.

This is also an artefact of the player we have chosen. Djokovic wins most of his matches, and we have seen that some models default to just guessing "win" for every match to attain a relatively good accuracy. The losses he does have are often upsets, and there is not much signal to predict these more random events. Hence, when choosing to pursure hard classification, the models often just pursue the win rate by predicting a win every time.

In summary, odds are the best predictors for win/loss classification in our dataset, and the addition of our chosen features does not significantly improve the performance of classic models.