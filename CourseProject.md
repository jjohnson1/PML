Predicting Activity Quality
========================================================
## Jeff Johnson
#### *June 21, 2015*

This document details my analysis for the course project in the Practical Machine Learning class. The goal is to create a machine learning algorithm to predict activiy quality from variable observations taken by fitness tracking devices while indidvuals were performing a weightlifting exercise. Below I describe my process of loading and cleaning the data, training an algorithm, using cross-validation, and estimating out of sample error. 



### Reading and Preparing Data
The data for this project come from the Weight Lifting Exercise Dataset described at http://groupware.les.inf.puc-rio.br/har. The training set can be found at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv, and the test set at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv. The data contain observations of 160 variables collected by exercise tracking devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* while subjects performed a simple weightlifting exercise. There are a number of missing observations throughout the dataset, and many variables contain only a small number of observations. For simplicity, I have elected to remove those varaibles prior to training a predictive model. 

The code below loads both the training and testing datasets (already downloaded from the URLs above) and removes unwanted variables. This leaves me with appropriate training and testing datasets to use to create and validate a predictive model.


```r
train <- read.csv("trainingdata.csv")
test <- read.csv("testdata.csv")
keep <- is.na(train[1,]) == FALSE & train[1,] != "" # remove unwanted variables (NAs and mostly empty variables)
train <- train[,keep]
train <- train[, -c(1,2,3,4,5)] # remove name, id, and timestamp information
test <- test[,keep]
test <- test[, -c(1,2,3,4,5)]
```

### Model Training
Next, I train a model using Linear Discriminant Analysis (LDA) with 10-fold cross-validation repeated 10 times. I chose the LDA method after attempting to use others (such as gradient boosting and random forest) that took too long to run on my computer with this dataset. LDA proved to run quickly enough to allow appropriate experimentation while providing reasonable accuracy (>70%). There are likely methods that provide better accuracy, though finding the most accurate model is not the focus of this project.

#### Cross-Validation
As mentioned above, I build the algorithm using 10-fold cross validation, repeated 10 times. This creates 10 subsets of the training set, trains a model based on 9 of those, and then estimates accuracy on the remaining subset. This process is repeated so that each subset is left out once, and then 10 new subets are created and the entire process repeats 10 times. The metrics reported after running the code below are the average values obtained from all samplings.

The code below trains an LDA model using repeated cross-validation as described above.


```r
set.seed(12345)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
fit <- train(classe ~ ., data = train, method = "lda", trControl = fitControl)
fit
```

```
## Linear Discriminant Analysis 
## 
## 19622 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 17660, 17659, 17659, 17660, 17661, 17661, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.7127451  0.6365404  0.009413663  0.01188489
## 
## 
```

### Training Set Accuracy
As a quick check, I investigate how well the model predicts the data used in the training process. The table below shows the number of obserations predicted to be a certain class (row) based on their actual class (column). The diagonal indicates accurate predictions, while off-diagonal entries are misclassifications.


```
##     
## pred    A    B    C    D    E
##    A 4615  520  332  173  144
##    B  164 2493  320  132  528
##    C  366  473 2271  395  321
##    D  415  154  401 2401  338
##    E   20  157   98  115 2276
```

This indicates reasonable performance of the model on the training set. The model accurately predicts 71.63% of the observations, which is right in line with expectations from the summary presented above. 

### Out of Sample Error
From the summary output of the model above, we see that the accuracy of the model is 71.27% when trained using 10-fold cross-validation repeated 10 times. As a result, **I expect the out of sample error rate (taken to be the misclassification rate in this case) to be 28.73%.** I calculate this by simply subtracting the reported accuracy from 1. 
