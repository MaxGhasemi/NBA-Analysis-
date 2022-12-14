#### BSAN 750 Project Contract Regression

nba<-read.csv("final_nba.csv")

#Look at the data
head(nba)


#View New data
head(nba1)

#Load Libraries
library(caret) 	
library(xgboost)
library(rpart.plot)
library(tidyverse)

nba<-read.csv("final_nba.csv")
#Log Salary
nba$logsalary<-log(nba$Salary,10)
#Exclude all Categorical Variables, and Contributions, Per Game Stats only
nba1<-nba[,-c(1,2,3,7:12,14,15,17,18,21,22,24,25,26,27,28,29,30,31,32,51,52,53)]


  

#Create training and testing data 
parts<-createDataPartition(nba1$logsalary, p = .8, list = F)
train<-nba1[parts, ]
test<-nba1[-parts, ]

#define predictor and response variables in training set
train_x <- data.matrix(train[, -28])
train_y <- train[,28]

#define predictor and response variables in testing set
test_x <- data.matrix(test[, -28])
test_y <- test[, 28]

#define final training and testing sets
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
watchlist <- list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
xgb.m1 <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)


#define final model
xgb.m2 <- xgboost(data = xgb_train, max.depth = 3, nrounds = 31, verbose = 0)
summary(xgb.m2)


# Compute feature importance matrix
importance_matrix <- xgb.importance(colnames(xgb_train), model = xgb.m2)
importance_matrix

# Variable Importance graph
xgb.plot.importance(importance_matrix)


# a plot with all the trees
xgb.plot.tree(model =  fit.xgboost.50, trees = 2)



#Make new model without 2k rating 


### Make a model to predict all star and no all star

# Get 17/18 and 18/19 data 

#Classification 

nba<-read.csv("final_nba.csv")
#Log Salary
nba$logsalary<-log(nba$Salary,10)
#Exclude all Categorical Variables, and Contributions, Per Game Stats only
nba1<-nba[,-c(1,2,3,7:12,14,15,17,18,21,22,24,25,26,27,28,29,30,31,32,51,52,53)]

#Training and Testing
index <- sample(nrow(nba1),nrow(nba1)*0.70)
nba.train <- nba1[index,]
nba.test <- nba1[-index,]

library(xgboost)
fit.xgboost.reg<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                          nthread = 1, nrounds = 100, objective = "reg:squarederror", verbose = 0)

#Cross Validation 
xgboost.cv<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                    nfold=10, nthread = 4, nrounds = 200, objective = "reg:squarederror", verbose = 0)
plot(xgboost.cv$evaluation_log$iter, xgboost.cv$evaluation_log$test_rmse_mean, type='l')


pred.xgboost<- predict(fit.xgboost.reg, newdata = as.matrix(nba.test[,-28]))
mspe1<-mean((nba.test$logsalary-pred.xgboost)^2)


# 200 Depth
fit.xgboost.200<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                          nthread = 1, nrounds = 200, objective = "reg:squarederror", verbose = 0)

#Cross Validation 
xgboost.cv200<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                    nfold=10, nthread = 4, nrounds = 400, objective = "reg:squarederror", verbose = 0)
plot(xgboost.cv$evaluation_log$iter, xgboost.cv$evaluation_log$test_rmse_mean, type='l')


pred.xgboost200<- predict(fit.xgboost.200, newdata = as.matrix(nba.test[,-28]))
mspe2<-mean((nba.test$logsalary-pred.xgboost200)^2)

# 100 round, 10 depth
fit.xgboost.10<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 10, eta = 0.1, 
                          nthread = 1, nrounds = 100, objective = "reg:squarederror", verbose = 0)

#Cross Validation 
xgboost.cv10<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 10, eta = 0.1, 
                       nfold=10, nthread = 4, nrounds = 200, objective = "reg:squarederror", verbose = 0)
plot(xgboost.cv10$evaluation_log$iter, xgboost.cv10$evaluation_log$test_rmse_mean, type='l')


pred.xgboost10<- predict(fit.xgboost.10, newdata = as.matrix(nba.test[,-28]))
mspe3<-mean((nba.test$logsalary-pred.xgboost10)^2)

# 50 rounds depth
fit.xgboost.50<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                         nthread = 1, nrounds = 75, objective = "reg:squarederror", verbose = 0)

#Cross Validation 
xgboost.cv50<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                      nfold=10, nthread = 4, nrounds = 75, objective = "reg:squarederror", verbose = 0)
plot(xgboost.cv50$evaluation_log$iter, xgboost.cv50$evaluation_log$test_rmse_mean, type='l')


pred.xgboost50<- predict(fit.xgboost.50, newdata = as.matrix(nba.test[,-28]))
mspe4<-mean((nba.test$logsalary-pred.xgboost50)^2)


# 50 rounds depth
  fit.xgboost.50<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
nthread = 1, nrounds = 50, objective = "reg:squarederror", verbose = 0)

   #Cross Validation 
  xgboost.cv50<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
 nfold=10, nthread = 4, nrounds = 50, objective = "reg:squarederror", verbose = 0)
 plot(xgboost.cv50$evaluation_log$iter, xgboost.cv50$evaluation_log$test_rmse_mean, type='l')
 
   
 pred.xgboost50<- predict(fit.xgboost.50, newdata = as.matrix(nba.test[,-28]))
 mspe4<-mean((nba.test$logsalary-pred.xgboost50)^2)
 mspe4



 
 # Testing Data with Kawhi Leonard
 

 
kawhi.df<-data.frame(kawhi.col,kl.data)

kawhi.col<-colnames(nba.test)

kl.data<-c(79,93,25,.485,.380,.529,.541,.880,33.4,8.6,17.7,2.0,5.2,6.6,12.5,6.3,7.2,1.1,4.7,5.8,3.5,1.8,.7,2.1,1.6,25.5,1,7.246451)





whi.df2<-t(kawhi.df)
kawhi[1,]<-kl.data
kl
head(kl)

kawhi.df3<-as.data.frame(kawhi.df2)

kawhi.df3<-!rownames(kawhi.df3)

pred.xgboost.kawhi<- predict(fit.xgboost.50, newdata = as.matrix(kawhi.df3[,-28]))
mspe.kl<-mean((nba.test$logsalary-pred.xgboost.kawhi)^2)
mspe.kl



#### Creating a Matrix 
#Creating a Matrix  
MatrixA <- matrix(data = kl.data, nrow = 1, ncol = 28)  
colnames(MatrixA)<-colnames(nba.test)


pred.xgboost.kawhi<- predict(fit.xgboost.50, newdata = as.matrix(MatrixA[,-28]))
mspe.kl<-mean((7.246451-pred.xgboost.kawhi)^2)

pe.kl<-7.246451-pred.xgboost.kawhi
pe.kl



# 50 rounds depth
fit.xgboost.50<- xgboost(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                         nthread = 1, nrounds = 50, objective = "reg:squarederror", verbose = 0)

#Cross Validation 
xgboost.cv50<- xgb.cv(data = as.matrix(nba.train[,-28]), label = nba.train[,28], max.depth = 4, eta = 0.1, 
                      nfold=10, nthread = 4, nrounds = 50, objective = "reg:squarederror", verbose = 0)
plot(xgboost.cv50$evaluation_log$iter, xgboost.cv50$evaluation_log$test_rmse_mean, type='l')


pred.xgboost50<- predict(fit.xgboost.50, newdata = as.matrix(nba.test[,-28]))
mspe4<-mean((nba.test$logsalary-pred.xgboost50)^2)
mspe4




plot(nba$X2K.Rating,nba$Salary)

d<-nba%>%
  group_by(Player)%>%
  filter(X2K.Rating>95)
  

d%>%
  group_by(Player)%>%
  filter(Salary<10000000)
