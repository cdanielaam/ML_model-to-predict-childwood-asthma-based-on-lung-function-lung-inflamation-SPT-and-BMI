#Install Packages:

install.packages("caret")
install.packages("ellipse")
install.packages("kernlab")
install.packages("randomForest")
install.packages("mlbench")
install.packages("MASS")                 
install.packages("XQuartz")

#Load packages:
library(tidyverse)
library(readr)
library(dplyr)
library(ggpubr)
library(car)
library(reshape2)
library(hms)
library(BlandAltmanLeh)
library(ggstatsplot)
library(caret)
library(ellipse)
library(kernlab)
library(randomForest)
library(mlbench)
library(MASS)
library(XQuartz)


#Import dataset:
ML_Asthma <- read_csv("~/Desktop/ML model in R _ Asthma /ML_Asthma.csv")

#Comvert GOLD_Dx as "factor":
ML_Asthma$GOLD_Dx <- as.factor(ML_Asthma$GOLD_Dx)
class(ML_Asthma$GOLD_Dx)

#Create a split model(80/20) for trainning/validadion:
validation_index <- createDataPartition(ML_Asthma$GOLD_Dx, p=0.80, list=FALSE)

#20% of the data for validation:
validation <- ML_Asthma[-validation_index,]

#80% of data for training and testing the models:
ML_Asthma <- ML_Asthma[validation_index,]

#Dataset dimensions:
dim(ML_Asthma)

#List types for each attribute:
sapply(ML_Asthma, class)

#List the levels for the class we want to predict:
levels(ML_Asthma$GOLD_Dx)

#Class Distribution:
percentage <- prop.table(table(ML_Asthma$GOLD_Dx)) * 100
cbind(freq=table(ML_Asthma$GOLD_Dx), percentage=percentage)

#Statistical Summary:
summary(ML_Asthma)

#Univariate Plots for attributes:
  #Split input(predictors) and output(predicted) attributes:
x <- ML_Asthma[,1:19]
y <- ML_Asthma[,20]

  #Boxplot for each imput attribut:
boxplot(ML_Asthma$FeNO)
boxplot(ML_Asthma$FVC_pre)
boxplot(ML_Asthma$FEV1_pre)
boxplot(ML_Asthma$FEF2575_pre)
boxplot(ML_Asthma$FEV1FVC_pre)
boxplot(ML_Asthma$BMI)
boxplot(ML_Asthma$REV_ML)

  #Plot output attribut:
plot(y)

#Multivariate Plots:
featurePlot(ML_Asthma$FeNO, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$FVC_pre, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$REV_ML, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$BMI, ML_Asthma$GOLD_Dx)

#Density plots for each attribute by class value:
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(ML_Asthma$FeNO, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FVC_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FEV1FVC_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FEV1_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$REV_ML, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$BMI, ML_Asthma$GOLD_Dx, plot="density", scales=scales)

#Test Harness:
  #This will split our dataset into 10 parts, train in 9 and test on 1 and 
  #release for all combinations of train-test splits. We will also repeat 
  #the process 3 times for each algorithm with different splits of the data 
  #into 10 groups, in an effort to get a more accurate estimate. We are using 
  #the metric of “Accuracy” to evaluate models. This is a ratio of the number 
  #of correctly predicted instances in divided by the total number of instances 
  #in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). 
  #We will be using the metric variable when we run build and evaluate each 
  #model next.

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Build 5 Models:
  #Linear Discriminant Analysis (LDA):
set.seed(7)
fit.lda <- train(GOLD_Dx~., data=ML_Asthma, method="lda", metric=metric, trControl=control)

  #Classification and Regression Trees (CART):
set.seed(7)
fit.cart <- train(GOLD_Dx~., data=ML_Asthma, method="rpart", metric=metric, trControl=control)

  #k-Nearest Neighbors (kNN):
set.seed(7)
fit.knn <- train(GOLD_Dx~., data=ML_Asthma, method="knn", metric=metric, trControl=control)

  #Support Vector Machines (SVM) with a linear kernel:
set.seed(7)
fit.svm <- train(GOLD_Dx~., data=ML_Asthma, method="svmRadial", metric=metric, trControl=control)

  #Random Forest (RF):
set.seed(7)
fit.rf <- train(GOLD_Dx~., data=ML_Asthma, method="rf", metric=metric, trControl=control)

#Select Best Model:
  #Summarize accuracy of models:
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

  #Compare accuracy of models:
dotplot(results)

  #Density plots of accuracy:
scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|")


  #Parallel plots to compare models
parallelplot(results)

  #Pair-wise scatterplots of predictions to compare models:
splom(results)

  #xyplot plots to compare 2 best models
xyplot(results, models=c("rf", "cart"))

  #Summarize Best Model (RF):
print(fit.rf)

#Making Predictions:
  #Estimating skill of RF on the validation dataset:
predictions <- predict(fit.rf, validation)
confusionMatrix(predictions, validation$GOLD_Dx)

view(predictions)
predictions <- predictions

#Table with predictions and data:
tbl = table(predictions, validation$GOLD_Dx)
tbl
chisq.test(tbl)

#Adding RF predictions to validadion dataset:
validation$RF_predictions <- predictions
view(validation)


