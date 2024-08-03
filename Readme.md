We try to build machine learning models to help predict the number of goals that a soccer team had scored based on based on the match statistics. The aim of the project is about inferring the key stats that result in more goals. The model does not aim to predict goals (and the winning team) for a unplayed or unknown matches. This kind of predictive models would require to learn based on teams recent past performances which is beyond the scope of this project.

We divide the project in two parts, data collection & cleaning, and model building.
# Data Collection
We first collected data from https://www.kaggle.com/datasets/pablohfreitas/all-premier-league-matches-20102021 which contains detailed stats of every soccer match played in EPL from 2010 to 2021. 
However we realized that this dataset doesn't contain Expected Goals (XG) stat which we thought would be crucial for the model to learn (in hindsight this is not required in our final model as XG is highly correleated with 'Shots on Target' which is already present in this dataset. In future we would like to go back to this dataset and train the models without XG.)

To get XG data we used https://www.kaggle.com/datasets/yusukefromjpn/england-premier-league-2018-to-2019-stats?select=england-premier-league-matches-2018-to-2019-stats.csv dataset. But this is only for 2018-2019 season and has 380 matches.

We combined the relevant stats from these two datasets and created a cleaned training dataset in notebook 'data_cleaning_epl'. We store the data as 'data/home_data_18_19.csv' and 'data/away_data_18_19.csv'

In **future** we would like to use all features of the first dataset and do a systematic feature selection, such as, PCA.


# Predicting Expected Goals (XG)
In notebook 'expected_goals_predictor' we try to infer the metric 'Expected Goals (XG)' from other features. XG can be defined as the expected number of goals that the team shuold have scored based on historical data. This includes various factors like location of shot, position of shooting player, position of goalkeeper, etc. 
We find very strong linear relationship between XG and shots_on_target. We did detailed linear regression analysis and verified the accuracy of our inference based on various statistical tests.

# Predicting number of goals a team scored
In notebook 'goal_predictor' we predict the number of goals that a team scored in a given match based on various features such as shots on target, whether the team won or drew, goals by opposition. We wanted to infer the features responsible for scoring higher goals. For this we primarily adopted Posisson Regression since the number of goals is a positive integer response and normal regression would not be good.
In addition, we also viewed predicting goals as a classification problem and fit various classification algorithms such as Logistic Regression, KNN classifier, Linear Discriminant Analysis (LDA), Gaussian Naive Bayes. We find that KNN overfits the data and performance of Poisson Regression and LDA are similar. Though Poisson Regression is preferred due to the possibilities of making better inferences.


## Error metrics
Below is detail about the metric we used to evaluate the models.
1. Error in goal prediction: Each model either gives a float value for predicted goals (eg Poisson Regression) or probability for different goals. For the latter we can calculate expexted goals using function mean_goals(). Let us call this mean by $\mu$. The error is then defined as $\epsilon = \frac{1}{N}\sum_{i=1}^N (y_i - \mu_i )^2.$
2. F-score for classification: For Poisson Regression we round the mean goal to nearest interger. This would be the prediction of the goal. For classification models the prediction is the most likely class. We then calculate the standard F1-score using sklearn library.

## Few comments on the model selection
1. Since we have around 10 features and ~700 samples, we are not in the regime of high-dimensional data. Hence we used ordinary least sqaure methods and did not care too much about issues commonly associated with high-dimensionality.
2. We did not use Lasso or penalized/regularized models. This is fine since our dimension of features is much small than number of samples. 
3. We only worked and looked at features with low level of multi-colinearity. We used VIF to throw/modify some features. 
4. We also used our knowledge of soccer to determine the features absolutely necessary for goals. E.g we know for sure that higher number of shots should lead to higher expected number of goals (this was also verified in 'expected_goals_predictor' notebook).

## Things that can be done and imporoved
#### Use a systematic feature selection. 
   1. Use Principal Component Analysis (PCA) instead of direct features.
   2. Use Lasso on Poisson regression. But this is not standard method in any package I know. This should be easy to implement though. 
   3. For KNN a systematic search for optimal neighbors can be done.
#### Infer the winning team
We can use the features to predict the winning team and learn the key stats that are responsible for better chances of winning. A preliminary analysis is done in notebook 'win_predictor'


## Testing on independent dataset