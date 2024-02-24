########################## REAL DATA EXAMPLE ############################
# We will be working with Human DNA methylation data from "flow-sorted 
#-blood samples; A "GenomicRatioSet":"Bioconductor" data object derived
#              from the "SummrizedExperiment" class                  #

#Loading required library
library(minfi)
library(here)
library(readr)
library(SummarizedExperiment)
library(caret)
library(glmnet)

################## let's read in the data

DNA.methylation.data<-readRDS(here("datasets/methylation.rds"))
DNA.methylation.data # we see that this object has "dim()=5000*37" p>>n
# Extract the matrix of methylation M-values
methyl.matrix<-assay(DNA.methylation.data)
# transpose to have features as column and samples as rows
methyl.matrix<-t(assay(DNA.methylation.data))
# view dimension of methylation matrix
methyl.matrix
dim(methyl.matrix)
# examine the metadata,phenotypes and grouping relating to this data
head(pData(DNA.methylation.data)) # for the first 6 samples

######### We will focus on the association between age and methylation ########
Age<-DNA.methylation.data$Age

#### let us check out what happens if we try fit a linear model to the data ###
# R will run a multivariate regression model in which each of the column in 
# methyl.matrix is used as predictor.

linear_model_fit <- lm(Age~methyl.matrix)
summary(linear_model_fit)
# singularities
XtX <- t(methyl.matrix)%*%methyl.matrix
det(XtX)  # we can't fit standard linear model to this high-dimensional data.

###### Now we'll work with set of features known to be associated with Age from
#                        a paper by "Horvath et al."                         #
# read in the data
coefhorvath<-readRDS(here("datasets/coefHorvath.rds"))
dim(coefhorvath); class(coefhorvath)
features<-coefhorvath[1:20,]$CpGmarker
horv.matrix<-methyl.matrix[,features]
dim(horv.matrix)   # not technically high-dimensional data 

# Generate an index to split the data into train and test set
set.seed(50)
train<-sample(nrow(methyl.matrix),27)
train.horv.matrix<-horv.matrix[train,] ; train.Age<-Age[train]
test.horv.matrix<-horv.matrix[-train,] ; test.Age<-Age[-train]

#############################################################################
########### OLS regression Vs Ridge regression on "horv.matrix" data ########

## multilinear regression fit
horv.lm.fit<-lm(train.Age~.,data = as.data.frame(train.horv.matrix))
summary(horv.lm.fit)
# Check mean squared error on the model.
horv.lm.mse<-mean(residuals(horv.lm.fit)^2)
horv.lm.mse
# examine the MSE on the test Data
pred.lm<-predict(horv.lm.fit,newdata = as.data.frame(test.horv.matrix))
 MSE.lm<-mean((test.Age-pred.lm)^2)
MSE.lm

## Ridge regression fit

# 100 lambda values for ridge and lasso
grid <- 10^seq(2,-2,length=100) 
# performing Leave One Out CV to search for the best lambda
cv.ridge<- cv.glmnet(x=train.horv.matrix,y=train.Age,nfolds =27,alpha=0,lambda =grid  )
ridge.fit<-glmnet(x=train.horv.matrix,y=train.Age,lambda = cv.ridge$lambda.min,alpha=0)
# plot of test MSE's vs lambda values
#plot showing how estimated coefficients change as we increase the penalty, "lambda"
ridg.fit<-glmnet(x=train.horv.matrix,y=train.Age,alpha=0)
dev.new() 
plot(cv.ridge)
dev.new() 
plot(ridg.fit,xvar="lambda")
abline(v=log(cv.ridge$lambda.min),lty="dashed")
abline(h=0,lty="dashed")
# examine MSE on test data
pred.ridge<-predict(ridge.fit,newx=test.horv.matrix)
MSE.ridge<-mean((test.Age-pred.ridge)^2)
MSE.ridge
####### Which performs better, Ridge or OLS ?
min(c(MSE.ridge,MSE.lm))
# plot predicted Ages for both method against the true Ages
lim<-range(c(pred.lm,test.Age,pred.ridge))
dev.new();par(mfrow=1:2)
plot(test.Age,pred.lm,xlim=lim,ylim=lim,pch=19) ;abline(0:1,lty="dashed")
plot(test.Age,pred.ridge,xlim=lim,ylim=lim,pch=19) ;abline(0:1,lty="dashed")

## display the coefficient estimated for both method
coeff<-cbind(coef(horv.lm.fit)[-1],coef(ridge.fit)[-1])
colnames(coeff)<-c("lm coefs","Ridge coefs") ; coeff 
write.csv2(coeff,file = "coef_lm_Ridge.csv")

###########################################################################
###########   LASSO regression on DNA.methylation.data ####################

# perform 10-folds CV to find the best lambda value
cv.lasso<-cv.glmnet(methyl.matrix,Age,alpha=1,lambda = grid ,nfolds=10)
lasso.fit<-glmnet(methyl.matrix,Age,alpha=1,lambda = cv.lasso$lambda.min)
# plot of test MSE's vs lambda values
#plot showing how estimated coefficients change as we increase the penalty, "lambda"
lass.fit<-glmnet(x=train.horv.matrix,y=train.Age,alpha=1)
dev.new() 
plot(cv.lasso)
dev.new() 
plot(lass.fit,xvar="lambda")
abline(v=log(cv.lasso$lambda.min),lty="dashed")

# view coefficients of the model
lasso_coefficients <- coef(lasso.fit)[-1]
lasso_coefficients
# view selected variables performed by Lasso regression

selected_coefs <- as.matrix(lasso_coefficients)[which(lasso_coefficients !=0),1]
selected_features<-names(selected_coefs)
selected_features
length(selected_features)
## compare features selected with Horvath signature
intersect(selected_features,coefhorvath$CpGmarker) # we selected some of the same feature
length(intersect(selected_features,coefhorvath$CpGmarker))

## Lasso Vs Ridge coefficients paths
dev.new() 
par(mfrow=c(2,1))
plot(ridg.fit,xvar="lambda",main="ridge case")
plot(lass.fit,xvar="lambda",main="lasso case")

###############################################################################
############### Ridge regression on DNA.methylation.data #####################

# perform 10-folds CV to find the best lambda value
cv.r<-cv.glmnet(methyl.matrix,Age,alpha=0,lambda = grid ,nfolds=10)
Ridge.fit<-glmnet(methyl.matrix,Age,alpha=0,lambda = cv.r$lambda.min)
# plot of test MSE's vs lambda values
#plot showing how estimated coefficients change as we increase the penalty, "lambda"
Ridg.fit<-glmnet(x=train.horv.matrix,y=train.Age,alpha=0)
dev.new() 
plot(cv.r)
dev.new() 
plot(Ridg.fit,xvar="lambda")
abline(v=log(cv.r$lambda.min),lty="dashed")

# view coefficients of the model
ridge_coefficients <- coef(Ridge.fit)[-1]
ridge_coefficients
##############################################################################
######### Blending Ridge regression and the LASSO : Elastic-Net #############

# set up (alpha,lambda)grid to search for pair that minimizes CV error 
# using "caret package"
alp.grid <-seq(0.05,0.9,length=10) ; lam.grid <- 10^seq(2,-2,length=20)
data<-as.data.frame(cbind(Age=Age,methyl.matrix))
# set up cross validation method for train function
control<-trainControl(method = "cv",number = 10)
#set up search grid for alpha and lambda parameters
srchgrid<-expand.grid(alpha=alp.grid,lambda=lam.grid)
#Training Elastic Net regression:perform CV forecasting y level based on all features
cv.elnet<-train(Age~.,data=data,method="glmnet",trControl=control,tuneGrid=srchgrid)
cv.elnet
# plot CV performance
dev.new()
plot(cv.elnet)
# Elastic net regression  model 
op.alp<-cv.elnet$bestTune$alpha
op.lam<-cv.elnet$bestTune$lambda
elnet.model<-glmnet(methyl.matrix,Age,alpha=op.alp,lambda=op.lam)

### Lasso Vs Elastic coefficients paths (setting "alpha=op.alp") 
eln.model<-glmnet(methyl.matrix,Age,alpha=op.alp)
dev.new()
par(mfrow=c(2,1))
plot(eln.model,main="Elastic Net case")
plot(lass.fit,main="lasso case")
### compare the coefficients with the LASSO model
elnet_coefs<-coef(elnet.model)[-1]
sum(elnet_coefs[,1]==0)     #number of coefficients set to zero for "elnet"
sum(lasso_coefficients[,1]==0) # number of coefficients set to zero for LASSO
 # plot Lasso coefficients against Elastic Net coefficients
dev.new()
plot(lasso_coefficients[,1],elnet_coefs[,1],pch=19,xlab="Lasso coefficients",
     ylab="Elastic net coefficients"); abline(0:1,lty="dashed",col="blue")
# compare features remaining in the model with Horvath signature
elnet.rm.features<-names(as.matrix(elnet_coefs)[which(elnet_coefs!=0),1])
elnet.rm.features; length(elnet.rm.features)
intersect(elnet.rm.features,coefhorvath$CpGmarker)
length(intersect(elnet.rm.features,coefhorvath$CpGmarker))

## display coefficients estimated the method: Lasso , Ridge and Elastic-Net

est.coeff<-cbind(lasso_coefficients,ridge_coefficients,elnet_coefs)
colnames(est.coeff)<-c("Lasso_coefs","Ridge_coef","Eln_coef")
est.coeff
write.csv2(est.coeff,file = "coef_DNA_methyl.csv")




