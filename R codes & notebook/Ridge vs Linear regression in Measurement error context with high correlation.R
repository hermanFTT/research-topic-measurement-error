# loading require package.
library(glmnet)

n=500  #number of samples
p=100  #number of covariates
n.sim=100 # number of Monte Carlo simulation
grid <- 10^seq(2,-2,length=100) # 100 lambda values for Ridge

### container matrix ###

# container matrix for mean square error(MSE) and prediction error for 
# "true linear model (lm), naive linear model, corrected lm, true ridge
#, naive ridge and corrected ridge regression.
lm.res= matrix(NA,nrow = n.sim , ncol = 3)
ridge.res=matrix(NA,nrow = n.sim , ncol = 3)
colnames(lm.res)<-c("t.lm.mse","n.lm.mse","cor.lm.mse")
colnames(ridge.res)<-c("t.ridge.mse","n.ridge.mse","cor.ridge.mse")

######### Loop over simulation and record MSE and PE value each time #########
for(i in 1:n.sim){
  ###### model setup
  
  # variance-covariance matrix of true unobserved values with high correlation
  Sigma_X <- matrix(NA,p,p) 
  for(j in 1:p){
    for(k in 1:p) Sigma_X[j,k] <- 0.9^abs(j-k)}
  #true variables
  X <- MASS::mvrnorm(n,rep(0,p),Sigma_X)
  #measurement error covariance matrix (assume here to be known)
  Sigma_U<-diag(x=0.75 ,nrow = p, ncol = p)
  U<- MASS::mvrnorm(n,rep(0,p),Sigma_U)
  #measurement matrix ( observed values)
  W <- X + U
  #coefficient
  beta<-runif(p,1,4)
  # Response
  y <- X %*% beta + rnorm(n, sd = 1)
  
  ###### fit true Linear model
  t.lm.fit<-lm(y~X)
  # estimated coefficient
  hat.beta<-coef(t.lm.fit)[-1]
  # MSE 
  lm.res[i,1]<-mean((hat.beta-beta)^2)
  
  ###### fit naive linear model
  
  n.lm.fit<-lm(y~W)
  # estimated coefficient
  hat.beta<-coef(n.lm.fit)[-1]
  # MSE 
  lm.res[i,2]<-mean((hat.beta-beta)^2)
  
  ###### correct for measurement error in the model
  
  #  reliability matrix "K" estimate
  hat.K<-solve(t(W)%*%W)%*%(t(W)%*%W-n*Sigma_U)
  # estimate coefficient under measurement error
  hat.beta.me<-solve(hat.K)%*%hat.beta
  ## MSE
  lm.res[i,3]<-mean((hat.beta.me-beta)^2)
  
  ####### fit true Ridge regression model
  
  # 10-folds cross validation to find the optimal "lambda"
  cv.t.ridge <-cv.glmnet(X,y,alpha=0,lambda=grid,nfolds = 10)
  t.ridge.fit<- glmnet(X,y,alpha=0,lambda=cv.t.ridge$lambda.min)
  hat.beta_R<-coef(t.ridge.fit)[-1]
  ## MSE
  ridge.res[i,1]<-mean((hat.beta_R-beta)^2)
  
  ###### fit naive Ridge
  
  # 10-folds cross validation to find the optimal "lambda"
  cv.n.ridge <-cv.glmnet(W,y,alpha=0,lambda=grid,nfolds = 10)
  n.ridge.fit<- glmnet(W,y,alpha=0,lambda=cv.n.ridge$lambda.min)
  hat.beta_nR<-coef(n.ridge.fit)[-1]
  ## MSE
  ridge.res[i,2]<-mean((hat.beta_nR-beta)^2)
  
  ##### correct for measurement error in ridge regression
  # use estimated reliability matrix "hat.K" 
  # perform regular ridge regression of "y" on "W%*%hat.K "
  # 10-folds cross validation to find the optimal "lambda"
  cv.cor.ridge <-cv.glmnet(W%*%hat.K,y,alpha=0,lambda=grid,nfolds = 10)
  cor.ridge.fit<- glmnet(W%*%hat.K,y,alpha=0,lambda=cv.cor.ridge$lambda.min)
  hat.beta_corR<-coef(cor.ridge.fit)[-1]
  ## MSE
  ridge.res[i,3]<-mean((hat.beta_corR-beta)^2)

}

######################## Result Matrix of MSE #############################
lm.res ; ridge.res
lm.MSE.res<-rbind(apply(lm.res ,2,mean),apply(lm.res ,2,sd) )
R.MSE.res<-rbind(apply(ridge.res ,2,mean),apply(ridge.res ,2,sd) )

#outcome object containing result average and standard deviation for each method
simulation.res<-cbind(lm.MSE.res,R.MSE.res)
rownames(simulation.res)<-c("Ave","Sd")

simulation.res

