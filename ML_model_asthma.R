#Install Packages:

install.packages("caret")
install.packages("ellipse")
install.packages("kernlab")
install.packages("randomForest")
install.packages("mlbench")
install.packages("MASS")

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


#Import dataset:
ML_Asthma <- read_csv("~/Desktop/ML model in R _ Asthma /ML_Asthma.csv")
View(ML_Asthma)

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
x <- ML_Asthma[,1:18]
y <- ML_Asthma[,19]

  #Boxplot for each imput attribut:
boxplot(ML_Asthma$FeNO)
boxplot(ML_Asthma$FVC_pre)
boxplot(ML_Asthma$FEV1_pre)
boxplot(ML_Asthma$FEF2575_pre)
boxplot(ML_Asthma$FEV1FVC_pre)
boxplot(ML_Asthma$BMI)
boxplot(ML_Asthma$REV_ml)

  #Plot output attribut:
plot(y)

#Multivariate Plots:
featurePlot(ML_Asthma$FeNO, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$FVC_pre, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$REV_ml, ML_Asthma$GOLD_Dx)
featurePlot(ML_Asthma$BMI, ML_Asthma$GOLD_Dx)

#Density plots for each attribute by class value:
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(ML_Asthma$FeNO, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FVC_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FEV1FVC_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$FEV1_pre, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
featurePlot(ML_Asthma$REV_ml, ML_Asthma$GOLD_Dx, plot="density", scales=scales)
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

#Adding RF predictions to validadion dataset (remove # to run):
#validation$RF_predictions <- predictions
#view(validation)

#Deeper Analysis of Ramdon Forest Model:
set.seed(222)
  #(GOLD_Dx~. where "." means all variables are include. It is possible to specify the
  #the variables we want to include)
rf <- randomForest(GOLD_Dx~., data = ML_Asthma)
print(rf)
  #(Attributes allows to see the attributes of model, so we can choose which attributes
  #to further check.)
attributes(rf)
rf$call
rf$type
rf$predicted
rf$votes
rf$err.rate
rf$confusion
rf$oob.times
rf$classes
rf$importance
rf$importanceSD
rf$localImportance
rf$proximity
rf$ntree
rf$mtry
rf$forest
rf$y
rf$test
rf$inbag
rf$terms

#Prediction and Confusion Matrix ML_Asthma:
p1 <- predict(rf, ML_Asthma)
  #Compare initial results from RF model and real data:
head(p1)
head(ML_Asthma$GOLD_Dx)
  #Confusion Matrix:
confusionMatrix(p1, ML_Asthma$GOLD_Dx)

#RF model evaluation with validation data:
p2 <- predict(rf, validation)
head(p2)
head(validation$GOLD_Dx)
confusionMatrix(p2, validation$GOLD_Dx)

#Error rate of RF model:
plot(rf)  
  #(it is possible to see that the RF model does not improve much after 150-200 trees)

#Number of nodes for the trees:
hist(treesize(rf),
     main = "No. of Nodes for the Tree",
     col = "blue")

#Variable importance (how worse the model plays if a variable is removed):
varImp(rf)
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "TOP 10 - Variable Importance")
#How often a variable appears in the RF model:
varUsed(rf)

#Graphical visualization of important variables:
#By Age:
ggplot(data = ML_Asthma) +
  aes(x = Age, fill = GOLD_Dx) +
  geom_histogram() +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By Age",
       caption = "Carla Martins")

#By Gender:
ggplot(data = ML_Asthma) +
  aes(x = Gender, fill = GOLD_Dx) +
  geom_histogram(bins = 3) +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By Gender",
       caption = "Author: Carla Martins")

#By SPT:
ggplot(data = ML_Asthma) +
  aes(x = SPT, fill = GOLD_Dx) +
  geom_histogram(bins = 3) +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By SPT",
       caption = "Carla Martins")

#By Father's Age:
ggplot(data = ML_Asthma) +
  aes(x = F_age, fill = GOLD_Dx) +
  geom_histogram(colour = "#1380A1") +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By Father's Age",
       caption = "Author: Carla Martins")

#By Mother's Age:
ggplot(data = ML_Asthma) +
  aes(x = M_age, fill = GOLD_Dx) +
  geom_histogram(colour = "#1380A1") +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By Mother's Age",
       caption = "Author: Carla Martins")

#By REV_ml:
ggplot(data = ML_Asthma) +
  aes(x = REV_ml, fill = GOLD_Dx) +
  geom_histogram(bins = 20, colour = "#1380A1") +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By Salbutamol Reversibility",
       caption = "Author: Carla Martins")

#By FEF2575_pre:
ggplot(data = ML_Asthma) +
  aes(x = FEF2575_pre, fill = GOLD_Dx) +
  geom_histogram(colour = "#1380A1") +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By FEF2575_pre",
       caption = "Author: Carla Martins")

#By FEV1_pre:
ggplot(data = ML_Asthma) +
  aes(x = FEV1_pre, fill = GOLD_Dx) +
  geom_histogram(colour = "#1380A1") +
  labs(title = "Asthma diagnosis rate",
       y = "Asthma",
       subtitle = "Distribution By FEV1_pre",
       caption = "Author: Carla Martins")

#Children reversibility according to FEV1 pre and SPT results:
ML_Asthma %>%
  filter %>%
  ggplot(mapping = aes(x = FEV1_pre, y = REV_ml)) +
  geom_point(aes(colour = GOLD_Dx)) +
  geom_smooth(se = TRUE)+
  facet_grid(Gender~SPT, scales = "free") +
  labs(title = "Pattern of reversibility by FEV1_pre",
       x = "FEV1 pre (ml)",
       y = "Salbutamol Reversibility (ml)",
       subtitle = "Children reversibility according to FEV1 pre and SPT results", 
       
       caption = "Author: Carla Martins") +
  theme(
    plot.subtitle = element_text(colour = "#17c5c9",
                                 size=14))

#Salbutamol Reversibility by GOLD_Dx
ML_Asthma %>%
  ggplot(mapping =  aes(x = GOLD_Dx, y = REV_ml)) +
  geom_point(colour = "#1380A1", size = 1) +
  geom_jitter(aes(colour = SPT))+ 
  geom_boxplot(alpha = 0.7, outlier.colour = NA)+
  labs(title = "Salbutamol Reversibility Distribution by Asthma Diagnosis and Age",
       x = "Asthma Diagnosis",
       y = "Salbutamol Reversibility (ml)",
       subtitle = "How Inhaled Salbutamol Reversibility distribution affects Asthma Diagnosis",
       caption = "Author: Carla Martins") 

#FEF2575_pre by GOLD_Dx
ML_Asthma %>%
  ggplot(mapping =  aes(x = GOLD_Dx, y = FEF2575_pre)) +
  geom_point(colour = "#1380A1", size = 1) +
  geom_jitter(aes(colour = SPT))+ 
  geom_boxplot(alpha = 0.7, outlier.colour = NA)+
  labs(title = "FEF2575 pre Distribution by Asthma Diagnosis and Age",
       x = "Asthma Diagnosis",
       y = "FEF2575 (ml/s)",
       subtitle = "How FEF2575 distribution affects Asthma Diagnosis",
       caption = "Author: Carla Martins") +
  coord_flip()







