# ML BGG
## Machine Learning Project with BoardGames
![head](resources/img/head.png)  


### Table of Contents  
[Intro](#Intro)  
[Exploratory Data Analysis](#Exploratory-Data-Analysis)  
[Feature Engineering](#Feature-Engineering)  
[Machine Learning](#Machine-Learning)  
[Results](#results-of-the-training)  
[Final Thoughts](#Final-Thoughts)  
[Sources](#Sources)

### Intro
-------------
The objective of the project is to try to predict the average rating of boardgames if we provide the model with enough data such as the minimum age to play, the number of players or what kind of mechanics exist within the game.  

### Exploratory Data Analysis
-------------
The dataset was pretty clean and we had a comfortable number of columns to investigate. Initially, we had a dataset with 15909 rows and 33 columns. In the early stages of the project, we used dummies for all the categorical columns and it raised the number of columns to 200, so we discarded the idea.  

We can observe in the images below that dices and resource management maintain their positions as the most popular type of games. The second picture is a heatmap of the variables in the dataset.  
![mechanics](resources/img/mechanicslong.png)

![heatmap](resources/img/heatmap.png)

### Feature Engineering
-------------
Getting rid of the NaN values were the utmost priority. Since there was only a few rows with more NaN than data, we just got rid of them (unknown games that lacked a lot of information). We also noticed that the Domains column had a lot of missings, so we assigned them the Unknown categorical to start and divided with dummies.  

### Machine Learning
-------------
This is a small sample of models we worked with: [Linear](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Polynomial](https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2), RandomForest, XGB Regressor, [BayesRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) or [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).  

### Results of the training
-------------  
While training the models, the feature importances showed that the Bayesian average and interaction with the game (visits to the board game page, comments about them, adding them to their wishlists...) helped greatly when trying to predict their rating.  

**Last iteration**  
| Model | R2 Score | R2 Score with new data |
| :--- | :---: | ---:|
| [XGB Regressor](https://xgboost.readthedocs.io/en/stable/parameter.html) | 0.899273 | 0.889522 |
| [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | 0.826174 | 0.792521 |
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | 0.800226 | 0.647375 |

### Final Thoughts
-------------
The winner was the XGB Regressor, and the feature importances were:  
![XGBFinal](resources/img/XGBFinal.png)

The project was very revealing and we can see the bright possibilities for the board game industry in the near future. Even though the model is not very good with newer games with few interactions(number of players wishlisting the game, talking about the game in social media, etc.), it's robust when given enough data with a high probability of making the correct prediction.  

For future work, we need to filter the features and work on the variables to do a new and improved iteration. Neural Networks are also very promising.  

### Sources
-------------
The initial dataset was from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/board-games).
* Python: 3.7.4
* Python libraries such as pandas, seaborn or sklearn.