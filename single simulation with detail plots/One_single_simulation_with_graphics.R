
################################################################################

" This is one single example of simulation with all details plot describing the models"

################################################################################


install.packages("devtools")
devtools::install_github("celiaescribe/BDcocolasso")
install.packages("hdme")
install.packages("glmnet")
install.packages("readr")

# loading required packages.
library(hdme); library(glmnet); library(BDcocolasso); library(readr)

n=200  #number of samples
p=3000   #number of covariates
n.sim=1 # number of monte carlo simulation
grid <- 10^seq(2,-2,length=100) # range of lambda values for ridge and lasso


l=0
### loop over simulations and record the MSE, PE, l1-error and number of coefs selected each time ####
# s= number of non zero coefficients
for(s in c(5)){ 
  l=l+1
  for(i in 1:n.sim){
    ###### model setup
    
    # variance-covariance matrix of true unobserved values fill with correlation
    Sigma_X <- matrix(NA,p,p) 
    for(j in 1:p){
      for(k in 1:p) Sigma_X[j,k] <- 0.5^abs(j-k)}
    
    #true variables
    X <- MASS::mvrnorm(n,rep(0,p),Sigma_X)
    
    #measurement error covariance matrix (assume here to be known)
    Sigma_U<-diag(x=0.75 ,nrow = p, ncol = p)
    U<- MASS::mvrnorm(n,rep(0,p),Sigma_U)
    
    #measurement matrix ( observed values)
    W <- X + U
    
    #coefficient
    beta<-rep(0,p); beta[sample(1:p,size = s)]=2
    
    # Response
    y <- X %*% beta + rnorm(n, sd = 0.05/1.96)
    
    
    ########## Fit true Lasso model
    
    # 10-fold cross-validation on Lasso to find best of 100 lambda values
    
    cv.t.lasso <-cv.glmnet(X,y,alpha=1,lambda=grid,
                           nfolds = 10,family="gaussian")
    t.lasso.fit <- glmnet(X,y,alpha=1,family="gaussian")
    hat.beta<-coef(t.lasso.fit)[-1,]
  
    ######## Fit Dantzig selector  on true model
    
    # 10-folds cross-validation (DS)
    ds.fit<-gds(X,y,family = "gaussian")
    hat.beta<-coef(ds.fit,all=T)[,2]
 
    ######## Naive lasso fit
    
    # 10-fold cross-validation on Lasso to find best of 100 lambda values
    cv.n.lasso <-cv.glmnet(W,y,alpha=1,lambda=grid,
                           nfolds = 10,family="gaussian")
    n.lasso.fit <- glmnet(W,y,alpha=1,family="gaussian")
    hat.beta<-coef(n.lasso.fit)[-1,]
    ## model accuracy :
    
    
    ######## fit corrected Lasso  ( Non-convex lasso or NCL)
    
    # select optimal R using naive lasso in a range [R_max/500,2*R_max]
    cv.fit<-cv_corrected_lasso(W,y,Sigma_U,no_radii=100,n_folds =10)
    R<-cv.fit$radius_min
    # corrected lasso with optimal R selected
    cor.lasso.fit<-corrected_lasso(W,y,Sigma_U,family = "gaussian")
    hat.beta<-rep(0,p)
    hat.beta[coef(cor.lasso.fit)$coefficient]<-coef(cor.lasso.fit)$estimate
    
  
    
    ########### fit convex conditional lasso (CoColasso) model
    
    coco.lasso.fit<-coco(Z=W,y=y,n=n,p=p,center.Z = FALSE, scale.Z = FALSE, step=100,K=10,mu=10,tau=0.75,etol = 1e-4,noise = "additive", block=FALSE, penalty="lasso",center.y = F,scale.y = F)
    hat.beta<-coco.lasso.fit$beta.opt
    
   
    ########## Matrix uncertainty selector 
    
    # the optimal lambda is selected by the cross-validation of the naive lasso
    
    # selected "delta" according to the "elbow" rule
    
    # use the selected delta to compute the final estimate
    mus.fit<-mus(W,y)
    # coefficient estimated
    hat.beta<-coef(mus.fit,all=TRUE)$estimate
    
  
  }
}



############################################################################
# Note 

 "Run the code from here, no need to retrain all the models, it could take
 2h-3h hours according to your avalaible computation power."
" Rather load the models already train provide with this code. You just need to 
specify the path to the '.rds' file in the  'readRDS()' function. "
################################################################################
############################# Load the model ###################################
library(caret)

################### true lasso model

# Coefficients path of the true model 
True_lasso_path=readRDS("/home/herman/Desktop/thesis_implement/one implementation/true.lasso.path.rds")
plot(True_lasso_path,xvar="lambda",main=" coefficient paths of the true Lasso model")
dev.new()

# Cross validation of the true model
cv=readRDS("/home/herman/Desktop/thesis_implement/one implementation/true.lasso.cv.rds")
plot(cv,main=" Cross-validation graphic of the true model")
dev.new()
# coefficients estimated true model
coef=readRDS("/home/herman/Desktop/thesis_implement/one implementation/true.lasso.model.rds")
plot(seq(1,3000),coef(coef)[-1,],main=" coefficients estimated of true lasso model" , col="blue")

dev.new()



######################## Naive Lasso


# Cross validation of the true model
cv=readRDS("/home/herman/Desktop/thesis_implement/one implementation/naive.lasso.cv.rds")
plot(cv,main=" Cross-validation graphic of the naive model")
dev.new()  

naive_lasso_path=readRDS("/home/herman/Desktop/thesis_implement/one implementation/naive.lasso.path.rds")
plot(naive_lasso_path,xvar="lambda",main=" coefficient paths of the naive Lasso model")
abline(v=log(cv$lambda.min),lty="dashed")
dev.new()

# coefficients estimated
coef=readRDS("/home/herman/Desktop/thesis_implement/one implementation/naive.lasso.model.rds")
plot(seq(1,3000),coef(coef)[-1,],main=" coefficients estimated of naive lasso model" , col="red" )

dev.new()

####################### Corrected Lasso (Non Convex Lasso )

## Cross-validation to find the optimal raduis
cv=readRDS("/home/herman/Desktop/thesis_implement/one implementation/corrected.lasso.cv.rds")
plot(cv)

dev.new()  

## Evolution of Nonzero coefficients with respect to  Radius ( R )
cor_lasso_path=readRDS("/home/herman/Desktop/thesis_implement/one implementation/corrected.lasso.path.rds")
plot(cor_lasso_path)
dev.new()

## coefficient paths corrected Lasso (NCL)
cor_lasso_path=readRDS("/home/herman/Desktop/thesis_implement/one implementation/corrected.lasso.path.rds")
plot(cor_lasso_path,type = 'path')
dev.new()

# coefficients estimated Corrected Lasso (NCL)
coef=readRDS("/home/herman/Desktop/thesis_implement/one implementation/corrected.lasso.model.rds")
plot(coef,main=" coefficients estimated")

dev.new()


################### CocoLasso 

Coco_lasso=readRDS("/home/herman/Desktop/thesis_implement/one implementation/CoCo.lasso.model.rds")
# plot coefficients paths CoCoLasso 
BDcocolasso::plotCoef(Coco_lasso)

# plot mean square error  CoCoLasso ( Convex Conditional Lasso)
BDcocolasso::plotError(Coco_lasso)


################### Matrix uncertainty selector (MUS)

## Number of Nonzero coefficients elong the grid of "delta"  ( to apply the Elbow rule)
dev.new()
mus1=readRDS("/home/herman/Desktop/thesis_implement/one implementation/MUS.model.rds")
plot(mus1)
dev.new()

## coefficients estimated MUS 

mus2=readRDS("/home/herman/Desktop/thesis_implement/MUS.model.rds")
plot(mus2)

