# loading required packages

library(glmnet)
library(pROC)
library(caret)

# setup
N <- 200 # number of observations
n.sim <- 30 # number of simulations
n.cov <- 1000 # number of covariates

############ four examples of high dimensional data sets ##########

# container matrix  (for misclassifications errors)

full.model.me <- matrix(NA,nrow = n.sim , ncol = 4)
lasso.me <- matrix(NA,nrow = n.sim , ncol = 4)
ridge.me <- matrix(NA,nrow = n.sim , ncol = 4)
elnet.me <- matrix(NA,nrow = n.sim , ncol = 4)

#container matrix (for AUC values)

full.model.auc <- matrix(NA,nrow = n.sim , ncol = 4)
lasso.auc <- matrix(NA,nrow = n.sim , ncol = 4)
ridge.auc <- matrix(NA,nrow = n.sim , ncol = 4)
elnet.auc <- matrix(NA,nrow = n.sim , ncol = 4)

#container matrix for the number of non-zero beta-coeff estimated

full.model.nb <- matrix(n.cov,nrow = n.sim , ncol = 4)
lasso.nb <- matrix(NA,nrow = n.sim , ncol = 4)
ridge.nb <- matrix(NA,nrow = n.sim , ncol = 4)
elnet.nb <- matrix(NA,nrow = n.sim , ncol = 4)

# 100 lambda values for ridge and lasso
grid <- 10^seq(2,-2,length=100) 
# set up (alpha,lambda)grid to search for pair that minimizes CV error
alp.grid <-seq(0.05,0.9,length=10) ; lam.grid <- 10^seq(2,-2,length=20)

set.seed(123)
 #### we loop over simulations and record the ME and AUC value each time #####

###############################################################################
############################### EXAMPLE 1 #####################################

 for(i in 1:n.sim) {
   
   # variance-covariance matrix fill with correlation
   Sigma <- matrix(NA,n.cov,n.cov) 
   for(j in 1:n.cov){
     for(k in 1:n.cov) Sigma[j,k] <- 0.5^abs(j-k)
   }
  diag(Sigma) <-1  # set diagonal to 1
  # N (200) random draws of 1000 covariates with mean 0 and variance Sigma
  X <- MASS::mvrnorm(N,rep(0,n.cov),Sigma)
  dim(X) # p >> n 
  # beta-coefficients
  beta <- c(runif(122,2,5),rep(0,878))
  # response variable Y simulation
  y <- apply(X,MARGIN = 1,FUN = function(x) rbinom(1,1,1/(1+exp(-t(x)%*%beta))))
  
  # split into training and test data
  train <- sample(c(1:N),size = 120)
   
  ################### Ridge
  
  # 10-fold cross-validation on ridge to find best of 100 lambda value
  
  cv.ridge <-cv.glmnet(X[train,],y[train],alpha=0,lambda=grid,
                          nfolds = 10,family="binomial")
  ridge.model <- glmnet(X[train,],y[train],alpha=0,
                      lambda=cv.ridge$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  r.pred.prob <- predict(ridge.model,newx=X[-train,],type = "response" )
  r.pred.classes <- ifelse(r.pred.prob > 0.5, 1,0)
  
 ## Model accuracy :
  #Misclassification error rate (ME)
obs.classes <- y[-train]
ridge.me[i,1] <- mean(r.pred.classes != obs.classes)
#AUC value
ridge.auc[i,1] <-auc(y[-train],r.pred.prob )
# number of non-zero beta-coefficients for ridge
ridge.nb[i,1]<- length(as.matrix(coef(ridge.model))[which(coef(ridge.model)[-1]!=0),1])
 
######################Lasso

# 10-fold cross-validation on Lasso to find best of 100 lambda values

cv.lasso <-cv.glmnet(X[train,],y[train],alpha=1,lambda=grid,
                     nfolds = 10,family="binomial")
lasso.model <- glmnet(X[train,],y[train],alpha=1,
                             lambda=cv.lasso$lambda.min,family="binomial")

# predict outcome using the model with the best lambda

l.pred.prob <- predict(lasso.model,newx=X[-train,],type = "response")
l.pred.classes <- ifelse(l.pred.prob > 0.5, 1,0)

## Model accuracy :
# Misclassification error rate (ME)
obs.classes <- y[-train]
lasso.me[i,1]<- mean(l.pred.classes!= obs.classes)
#AUC value
lasso.auc[i,1]<- auc(y[-train],l.pred.prob )

# number of non-zero  beta-coefficients for Lasso regression

lasso.nb[i,1]<- length(as.matrix(coef(lasso.model))[which(coef(lasso.model)[-1]!=0),1])


####################### Elastic-Net

y1<-as.factor(y)
data<-as.data.frame(cbind(y1,X))
test.data <- data[-train,]
train.data<-data[train,]

# set up cross validation method for train function
control<-trainControl(method = "cv",number = 10)
#set up search grid for alpha and lambda parameters
srchgrid<-expand.grid(alpha=alp.grid,lambda=lam.grid)
#Training Elastic Net regression:perform CV forecasting y level based on all features
cv.elnet<-train(y1~.,data=train.data,method="glmnet",trControl=control,
tuneGrid=srchgrid)

# Elastic net regression  model 
op.alp<-cv.elnet$bestTune$alpha
op.lam<-cv.elnet$bestTune$lambda
elnet.model<-glmnet(X[train,],y[train],alpha=op.alp,lambda=op.lam,family="binomial")
# predict outcome using the model
eln.pred.prob<-predict(elnet.model,newx=X[-train,],type = "response")
eln.pred.classes<- ifelse(eln.pred.prob > 0.5, 1,0)

## Model accuracy :
# Misclassification error rate (ME)
elnet.me[i,1]<- mean(eln.pred.classes!=obs.classes)
# AUC value
elnet.auc[i,1]<-auc(y[-train],eln.pred.prob)
# number of non-zero beta coefficients for Elastic-net
elnet.nb[i,1]<- length(as.matrix(coef(elnet.model))[which(coef(elnet.model)[-1]!=0),1])


###################### Full logistic model

data<-as.data.frame(cbind(y,X))
test.data <- data[-train,-1]
train.data<-data[train,]
full.model<-glm(y~.,data=train.data,family="binomial")


# predict outcome using the model 
full.pred.prob <- predict(full.model,newdata=test.data,type = "response")
full.pred.classes <- ifelse(full.pred.prob > 0.5, 1,0)

 ## Model accuracy :
# Misclassification error rate (ME)
obs.classes <- y[-train]
full.model.me[i,1]<- mean(full.pred.classes!=obs.classes)
# AUC value
full.model.auc[i,1]<-auc(y[-train],full.pred.prob )
 }

###############################################################################
################################ EXAMPLE 2 ####################################
for(i in 1:n.sim) {
  
  # variance-covariance matrix fill with correlation
  Sigma <- matrix(NA,n.cov,n.cov) 
  for(j in 1:n.cov){
    for(k in 1:n.cov) Sigma[j,k] <- 0.5^abs(j-k)
  }
  diag(Sigma) <-1  # set diagonal to 1
  # N (200) random draws of 1000 covariates with mean 0 and variance Sigma
  X <- MASS::mvrnorm(N,rep(0,n.cov),Sigma)
  dim(X) # p >> n 
  # beta-coefficients
  beta <- c(rep(0.8,n.cov))
  # response variable Y simulation
  y <- apply(X,MARGIN = 1,FUN = function(x) rbinom(1,1,1/(1+exp(-t(x)%*%beta))))
  
  # slit into training and test data
  train <- sample(c(1:N),size = 120)
  
  ################### Ridge
  
  # 10-fold cross-validation on ridge to find best of 100 lambda value
  
  cv.ridge <-cv.glmnet(X[train,],y[train],alpha=0,lambda=grid,
                       nfolds = 10,family="binomial")
  ridge.model <- glmnet(X[train,],y[train],alpha=0,
                        lambda=cv.ridge$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  r.pred.prob <- predict(ridge.model,newx=X[-train,],type = "response" )
  r.pred.classes <- ifelse(r.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  #Misclassification error rate (ME)
  obs.classes <- y[-train]
  ridge.me[i,2] <- mean(r.pred.classes != obs.classes)
  #AUC value
  ridge.auc[i,2] <-auc(y[-train],r.pred.prob )
  # number of non-zero beta-coefficients for ridge
  ridge.nb[i,2]<- length(as.matrix(coef(ridge.model))[which(coef(ridge.model)[-1]!=0),1])
  
  
  ######################Lasso
  
  # 10-fold cross-validation on Lasso to find best of 100 lambda values
  
  cv.lasso <-cv.glmnet(X[train,],y[train],alpha=1,lambda=grid,
                       nfolds = 10,family="binomial")
  lasso.model <- glmnet(X[train,],y[train],alpha=1,
                        lambda=cv.lasso$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  l.pred.prob <- predict(lasso.model,newx=X[-train,],type = "response")
  l.pred.classes <- ifelse(l.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  lasso.me[i,2]<- mean(l.pred.classes!= obs.classes)
  #AUC value
  lasso.auc[i,2]<- auc(y[-train],l.pred.prob )
  #number  of non-zero beta-coefficients for lasso
  lasso.nb[i,2]<- length(as.matrix(coef(lasso.model))[which(coef(lasso.model)[-1]!=0),1])
  
  ####################### Elastic-Net
  
  y1<-as.factor(y)
  data<-as.data.frame(cbind(y1,X))
  test.data <- data[-train,]
  train.data<-data[train,]
  
  # set up cross validation method for train function
  control<-trainControl(method = "cv",number = 10)
  #set up search grid for alpha and lambda parameters
  srchgrid<-expand.grid(alpha=alp.grid,lambda=lam.grid)
  #Training Elastic Net regression:perform CV forecasting y level based on all features
  cv.elnet<-train(y1~.,data=train.data,method="glmnet",trControl=control,
                  tuneGrid=srchgrid)
  
  # Elastic net regression  model 
  op.alp<-cv.elnet$bestTune$alpha
  op.lam<-cv.elnet$bestTune$lambda
  elnet.model<-glmnet(X[train,],y[train],alpha=op.alp,lambda=op.lam,family="binomial")
  # predict outcome using the model
  eln.pred.prob<-predict(elnet.model,newx=X[-train,],type = "response")
  eln.pred.classes<- ifelse(eln.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  elnet.me[i,2]<- mean(eln.pred.classes!=obs.classes)
  # AUC value
  elnet.auc[i,2]<-auc(y[-train],eln.pred.prob)
  #number of non-zero beta coefficients for Elastic-net
  elnet.nb[i,2]<- length(as.matrix(coef(elnet.model))[which(coef(elnet.model)[-1]!=0),1])
  
  
  ###################### Full logistic model
  
  data<-as.data.frame(cbind(y,X))
  test.data <- data[-train,-1]
  train.data<-data[train,]
  full.model<-glm(y~.,data=train.data,family="binomial")
  
  
  # predict outcome using the model 
  full.pred.prob <- predict(full.model,newdata=test.data,type = "response")
  full.pred.classes <- ifelse(full.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  full.model.me[i,2]<- mean(full.pred.classes!=obs.classes)
  # AUC value
  full.model.auc[i,2]<-auc(y[-train],full.pred.prob )
}


###############################################################################
################################ EXAMPLE 3 ####################################

for(i in 1:n.sim) {
  
  # variance-covariance matrix fill with correlation
  Sigma <- matrix(NA,n.cov,n.cov) 
  for(j in 1:n.cov){
    for(k in 1:n.cov) Sigma[j,k] <- 0.9^abs(j-k)
  }
  diag(Sigma) <-1  # set diagonal to 1
  # N (200) random draws of 1000 covariates with mean 0 and variance Sigma
  X <- MASS::mvrnorm(N,rep(0,n.cov),Sigma)
  dim(X) # p >> n 
  # beta-coefficients
  beta <-rep(c(rep(2,125),rep(0,125)),4)
  # response variable Y simulation
  y <- apply(X,MARGIN = 1,FUN = function(x) rbinom(1,1,1/(1+exp(-t(x)%*%beta))))
  
  # slit into training and test data
  train <- sample(c(1:N),size = 120)
  
  ################### Ridge
  
  # 10-fold cross-validation on ridge to find best of 100 lambda value
  
  cv.ridge <-cv.glmnet(X[train,],y[train],alpha=0,lambda=grid,
                       nfolds = 10,family="binomial")
  ridge.model <- glmnet(X[train,],y[train],alpha=0,
                        lambda=cv.ridge$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  r.pred.prob <- predict(ridge.model,newx=X[-train,],type = "response" )
  r.pred.classes <- ifelse(r.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  #Misclassification error rate (ME)
  obs.classes <- y[-train]
  ridge.me[i,3] <- mean(r.pred.classes != obs.classes)
  #AUC value
  ridge.auc[i,3] <-auc(y[-train],r.pred.prob )
  # number of non-zero beta-coefficients for ridge
  ridge.nb[i,3]<- length(as.matrix(coef(ridge.model))[which(coef(ridge.model)[-1]!=0),1])
  
  
  ######################Lasso
  
  # 10-fold cross-validation on Lasso to find best of 100 lambda values
  
  cv.lasso <-cv.glmnet(X[train,],y[train],alpha=1,lambda=grid,
                       nfolds = 10,family="binomial")
  lasso.model <- glmnet(X[train,],y[train],alpha=1,
                        lambda=cv.lasso$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  l.pred.prob <- predict(lasso.model,newx=X[-train,],type = "response")
  l.pred.classes <- ifelse(l.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  lasso.me[i,3]<- mean(l.pred.classes!= obs.classes)
  #AUC value
  lasso.auc[i,3]<- auc(y[-train],l.pred.prob )
  #number  of non-zero beta-coefficients for lasso
  lasso.nb[i,3]<- length(as.matrix(coef(lasso.model))[which(coef(lasso.model)[-1]!=0),1])
  
  ####################### Elastic-Net
  
  y1<-as.factor(y)
  data<-as.data.frame(cbind(y1,X))
  test.data <- data[-train,]
  train.data<-data[train,]
  
  # set up cross validation method for train function
  control<-trainControl(method = "cv",number = 10)
  #set up search grid for alpha and lambda parameters
  srchgrid<-expand.grid(alpha=alp.grid,lambda=lam.grid)
  #Training Elastic Net regression:perform CV forecasting y level based on all features
  cv.elnet<-train(y1~.,data=train.data,method="glmnet",trControl=control,
                  tuneGrid=srchgrid)
  
  # Elastic net regression  model 
  op.alp<-cv.elnet$bestTune$alpha
  op.lam<-cv.elnet$bestTune$lambda
  elnet.model<-glmnet(X[train,],y[train],alpha=op.alp,lambda=op.lam,family="binomial")
  # predict outcome using the model
  eln.pred.prob<-predict(elnet.model,newx=X[-train,],type = "response")
  eln.pred.classes<- ifelse(eln.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  elnet.me[i,3]<- mean(eln.pred.classes!=obs.classes)
  # AUC value
  elnet.auc[i,3]<-auc(y[-train],eln.pred.prob)
  #number of non-zero beta coefficients for Elastic-net
  elnet.nb[i,3]<- length(as.matrix(coef(elnet.model))[which(coef(elnet.model)[-1]!=0),1])
  
  
  ###################### Full logistic model
  
  data<-as.data.frame(cbind(y,X))
  test.data <- data[-train,-1]
  train.data<-data[train,]
  full.model<-glm(y~.,data=train.data,family="binomial")
  
  
  # predict outcome using the model 
  full.pred.prob <- predict(full.model,newdata=test.data,type = "response")
  full.pred.classes <- ifelse(full.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  full.model.me[i,3]<- mean(full.pred.classes!=obs.classes)
  # AUC value
  full.model.auc[i,3]<-auc(y[-train],full.pred.prob )
}


###############################################################################
################################ EXAMPLE 4 ####################################

for(i in 1:n.sim) {
  
  # variance-covariance matrix fill with correlation
  Sigma <- matrix(0,n.cov,n.cov) 
  for(j in 1:n.cov/2){
    for(k in 1:n.cov/2) Sigma[j,k] <- 0.5^abs(j-k)
  }
  diag(Sigma) <-1  # set diagonal to 1
  # N (200) random draws of 1000 covariates with mean 0 and variance Sigma
  X <- MASS::mvrnorm(N,rep(0,n.cov),Sigma)
  dim(X) # p >> n 
  # beta-coefficients
  beta <-c(rep(3,500),rep(0,500))
  # response variable Y simulation
  y <- apply(X,MARGIN = 1,FUN = function(x) rbinom(1,1,1/(1+exp(-t(x)%*%beta))))
  
  # slit into training and test data
  train <- sample(c(1:N),size = 120)
  
  ################### Ridge
  
  # 10-fold cross-validation on ridge to find best of 100 lambda value
  
  cv.ridge <-cv.glmnet(X[train,],y[train],alpha=0,lambda=grid,
                       nfolds = 10,family="binomial")
  ridge.model <- glmnet(X[train,],y[train],alpha=0,
                        lambda=cv.ridge$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  r.pred.prob <- predict(ridge.model,newx=X[-train,],type = "response" )
  r.pred.classes <- ifelse(r.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  #Misclassification error rate (ME)
  obs.classes <- y[-train]
  ridge.me[i,4] <- mean(r.pred.classes != obs.classes)
  #AUC value
  ridge.auc[i,4] <-auc(y[-train],r.pred.prob )
  # number of non-zero beta-coefficients for ridge
  ridge.nb[i,4]<- length(as.matrix(coef(ridge.model))[which(coef(ridge.model)[-1]!=0),1])
  
  
  ######################Lasso
  
  # 10-fold cross-validation on Lasso to find best of 100 lambda values
  
  cv.lasso <-cv.glmnet(X[train,],y[train],alpha=1,lambda=grid,
                       nfolds = 10,family="binomial")
  lasso.model <- glmnet(X[train,],y[train],alpha=1,
                        lambda=cv.lasso$lambda.min,family="binomial")
  
  # predict outcome using the model with the best lambda
  
  l.pred.prob <- predict(lasso.model,newx=X[-train,],type = "response")
  l.pred.classes <- ifelse(l.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  lasso.me[i,4]<- mean(l.pred.classes!= obs.classes)
  #AUC value
  lasso.auc[i,4]<- auc(y[-train],l.pred.prob )
  #number  of non-zero beta-coefficients for lasso
  lasso.nb[i,4]<- length(as.matrix(coef(lasso.model))[which(coef(lasso.model)[-1]!=0),1])
  
  
  ####################### Elastic-Net
  
  y1<-as.factor(y)
  data<-as.data.frame(cbind(y1,X))
  test.data <- data[-train,]
  train.data<-data[train,]
  
  # set up cross validation method for train function
  control<-trainControl(method = "cv",number = 10)
  #set up search grid for alpha and lambda parameters
  srchgrid<-expand.grid(alpha=alp.grid,lambda=lam.grid)
  #Training Elastic Net regression:perform CV forecasting y level based on all features
  cv.elnet<-train(y1~.,data=train.data,method="glmnet",trControl=control,
                  tuneGrid=srchgrid)
  
  # Elastic net regression  model 
  op.alp<-cv.elnet$bestTune$alpha
  op.lam<-cv.elnet$bestTune$lambda
  elnet.model<-glmnet(X[train,],y[train],alpha=op.alp,lambda=op.lam,family="binomial")
  # predict outcome using the model
  eln.pred.prob<-predict(elnet.model,newx=X[-train,],type = "response")
  eln.pred.classes<- ifelse(eln.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  elnet.me[i,4]<- mean(eln.pred.classes!=obs.classes)
  # AUC value
  elnet.auc[i,4]<-auc(y[-train],eln.pred.prob)
  #number of non-zero beta coefficients for Elastic-net
  elnet.nb[i,4]<- length(as.matrix(coef(elnet.model))[which(coef(elnet.model)[-1]!=0),1])
  
  
  ###################### Full logistic model
  
  data<-as.data.frame(cbind(y,X))
  test.data <- data[-train,-1]
  train.data<-data[train,]
  full.model<-glm(y~.,data=train.data,family="binomial")
  
  
  # predict outcome using the model 
  full.pred.prob <- predict(full.model,newdata=test.data,type = "response")
  full.pred.classes <- ifelse(full.pred.prob > 0.5, 1,0)
  
  ## Model accuracy :
  # Misclassification error rate (ME)
  obs.classes <- y[-train]
  full.model.me[i,4]<- mean(full.pred.classes!=obs.classes)
  # AUC value
  full.model.auc[i,4]<-auc(y[-train],full.pred.prob )
}

################################################################################
######## result Matrix of ME and AUC values for each simulations ##########

full.model.me 
lasso.me 
ridge.me 
elnet.me 

full.model.auc
lasso.auc
ridge.auc 
elnet.auc 

full.model.nb
lasso.nb 
ridge.nb 
elnet.nb 
 ## we take column mean to get the Average ME over n.sim simulations and
#create an outcome object where the rows contain ridge,lasso,el-net
#and full logistic results respectively

Ave.ME.results<-rbind(apply(ridge.me ,2,mean),apply(lasso.me ,2,mean),
            apply(elnet.me ,2,mean), apply(full.model.me ,2,mean)  )
 rownames(Ave.ME.results)<-c("Ridge","Lasso","Elastic Net"," full logistic model")
 colnames(Ave.ME.results)<-c("ME.ave_Exp1","ME.ave_Exp2","ME.ave_Exp3","ME.ave_Exp4")

 Ave.ME.results
 ### Now proceed the same way and store Standard deviation ME over n.sim
 Sd.ME.results<-rbind(apply(ridge.me ,2,sd),apply(lasso.me ,2,sd),
                       apply(elnet.me ,2,sd), apply(full.model.me ,2,sd))
 rownames(Sd.ME.results)<-c("Ridge","Lasso","Elastic Net"," full logistic model")
 colnames(Sd.ME.results)<-c("ME.sd_Exp1","ME.sd_Exp2","ME.sd_Exp3","ME.sd_Exp4")
 
 Sd.ME.results
 ### outcome object containing results Average AUC over n.sim simulations
 Ave.AUC.results<-rbind(apply(ridge.auc ,2,mean),apply(lasso.auc ,2,mean),
                       apply(elnet.auc ,2,mean), apply(full.model.auc ,2,mean)  )
 rownames(Ave.AUC.results)<-c("Ridge","Lasso","Elastic Net"," full logistic model")
 colnames(Ave.AUC.results)<-c("AUC.ave_Exp1","AUC.ave_Exp2","AUC.ave_Exp3","AUC.ave_Exp4")
 
 Ave.AUC.results
 ### outcome object containing results standard deviation AUC over n.sim simulations
 Sd.AUC.results<-rbind(apply(ridge.auc ,2,sd),apply(lasso.auc ,2,sd),
                      apply(elnet.auc ,2,sd), apply(full.model.auc ,2,sd))
 rownames(Sd.AUC.results)<-c("Ridge","Lasso","Elastic Net"," full logistic model")
 colnames(Sd.AUC.results)<-c("AUC.sd_Exp1","AUC.sd_Exp2","AUC.sd_Exp3","AUC.sd_Exp4")
 
 Sd.AUC.results
 ### Outcome object containing results Average of non-zero beta coefficients estimated 
 # over n.sim simulations
 Ave.nbc.results<-rbind(apply(ridge.nb ,2,mean),apply(lasso.nb ,2,mean),
                        apply(elnet.nb,2,mean), apply(full.model.nb,2,mean) )
 rownames(Ave.nbc.results)<-c("Ridge","Lasso","Elastic Net"," full logistic model")
 colnames(Ave.nbc.results)<-c("nbc.ave_Exp1","nbc.ave_Exp2","nbc.ave_Exp3","nbc.ave_Exp4")
 
 Ave.nbc.results
 
 ############################# RESUME ###########################
 
 Simulation.Results<-list(Ave.ME=Ave.ME.results,Sd.ME=Sd.ME.results,
Ave.AUC=Ave.AUC.results,Sd.AUC= Sd.AUC.results,Ave.Num.coef=Ave.nbc.results)
 
 Simulation.Results
 
 # Graphics
 barplot(Ave.ME.results,beside = TRUE,legend.text = T,col = c("seagreen","violet","grey","red1"),
  border =T,main = "Barplot Average Misclassification \n error for each example.")
 dev.new()
 barplot(Ave.AUC.results,beside = TRUE,legend.text = T,col = c("yellow2","cyan","grey","red3"),
         border =T,main = "Barplot Average\nAUC for each example.",width = 0.5)
 
 ## Export the results
 
library(readr)
write.csv2(Ave.ME.results,file = "Ave.ME.csv")
write.csv2(Sd.ME.results,file = "Sd.ME.csv")
write.csv2(Ave.AUC.results,file = "Ave.AUC.csv")
write.csv2(Sd.AUC.results,file = "Sd.AUC.csv")
write.csv2(Ave.nbc.results,file = "Ave.Num.coef.csv")
