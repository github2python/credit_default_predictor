# Predicting Credit Default Using Machine Learning 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) [![matplotlib](https://img.shields.io/badge/matplotlib-006400.svg?&style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com) [![XGBoost](https://img.shields.io/badge/XGBoost-F2C037.svg?&style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)


## Introduction 

In the world of banking, lending isn't just a transaction—it's a strategic move that can make or break a financial institution's success. Banks face the critical challenge of determining how much to lend and to whom, based on a multitude of factors that can impact both their profitability and their risk exposure.

Imagine a scenario where a bank must decide whether to grant a substantial loan to a potential borrower. The decision hinges not only on the borrower’s financial history but also on predicting their ability to repay. Banks are constantly balancing the scales between offering loans that meet customer needs and minimizing the risk of defaults.

With each loan, banks are essentially making a bet on the borrower’s future. A well-calibrated lending strategy can enhance customer satisfaction and drive business growth, while poor decisions can lead to financial losses and damaged reputations. As such, understanding customer behavior and leveraging advanced predictive models becomes crucial for making informed lending decisions.

By analyzing key factors and historical data, banks can refine their lending practices, ensuring they support the right customers while safeguarding their financial health. This intricate dance of risk and reward is what makes loan management both challenging and fascinating.



![](Payday-Loan.gif)

## Use of Machine Learning

Companies can harness the power of machine learning to uncover key features and insights that drive their lending decisions. By leveraging predictive models, they can more accurately assess whether to approve or deny a loan application.

Imagine if, based on a set of specific features—such as financial history, income level, and credit score—a machine learning model could forecast the likelihood of a customer defaulting on a loan. This capability enables companies to make more informed lending decisions, reducing risk and improving financial outcomes.

With machine learning, businesses can analyze vast amounts of data to identify patterns and trends, ultimately predicting customer behavior with greater precision. This not only enhances their decision-making process but also supports more responsible and strategic lending practices.

Incorporating machine learning into loan assessment processes can transform how companies evaluate risk, offering a smarter, data-driven approach to managing financial uncertainty.

## Metrics

The output variable in our case is __discrete__, making this a classification problem. To evaluate the performance of our model in predicting whether a person will default on a loan, we should use classification metrics. Below are some key metrics to consider:

* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1 Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__Area Under Curve (AUC)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)
* [__Classification Report__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Visualizations

In this section, we would be primarily focusing on the visualizations from the analysis and the __ML model prediction__ matrices to determine the best model for deployment. 

After taking a look at a few rows and columns in the dataset, there are features such as whether the loan applicant has a type of loan, and most importantly whether they have defaulted on a loan or not, age, montly income, debt ratio (important). 

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/data.jpg  "/>

There were many missing field in many columns mostly in dependents column. Therefore, various imputation methods can be used. In addition, features that do not give a lot of predictive information can be removed. Since they are numerical, methods such as mean imputation, median imputation, and mode imputation could be used in this process of filling in the missing values. I used column mean to fill them

### Model Performance

#### Analysis

Since this is a __binary classification task__, metrics such as precision, recall, f1-score, and accuracy can be taken into consideration. Various plots that indicate the performance of the model can be plotted such as confusion matrix plots and AUC curves. Let us look at how the models are performing in the test data. 

First I have used three models as shown below and compared their performance on various parameters as stated above.

[__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 

[__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[__XG Boost Classifier__](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)

## Machine Learning Models

We know that there are __millions of records__ in our data. Hence, it is important to use the most appropriate machine learning model that deals with __high-dimensional data__ well. Below are the results for various machine learning models used for predicting whether a person would default on a __loan or not__. 

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/table.jpg  "/>

__Confusion Matrix__ -

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/confusion_before_tuning.png  "/>

__Models Performance__

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/models_performance.png  "/>

__ROC Curve__

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/ROC.png  "/>

Now as you can see models are performing well but we can improve the performance by hyperparameter tuning.
I have used randomized search cv for tuning the models

[__RandomizedSearchCV__](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

Here are the results for all the models after tuning - 

__Performance Table__

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/table_compare.jpg  "/>

__Confusion Matrix__ -

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/Confusion_matrix.png  "/>

__Models Performance__

<img src = "https://github.com/github2python/credit_default_predictor/blob/main/images/Models_tuning_after.png  "/>

We can see that there has been significant jump in the performance of __Decision Tree Classifier__ after tuning and for other two models also performance has slightly improved.

##Conclusion

Random Forest Classifier and XG Boost have nearly same performances, I have used random forest because it ran fast on my local machine as compared to XG Boost as it is a bit heavy because of optimizing various features.

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git that could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link of the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(18).png" width = "600" />

5. The link of the repository can be found when you click on "Code" (Green button) and then, there would be a html link just below. Therefore, the command to download a particular repository should be "Git clone html" where the html is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(20).png" width = "600" />

8. Later, open the jupyter notebook by writing "jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 
