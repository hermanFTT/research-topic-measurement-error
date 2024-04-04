##############################################################################
"100 simulations in a loop.

estimated ending time: 3 days."
##############################################################################

# install required packages

#install.packages("devtools")
#devtools::install_github("celiaescribe/BDcocolasso")
#install.packages("hdme")
#install.packages("glmnet")
install.packages("readr")

# loading required packages.
library(hdme); library(glmnet); library(BDcocolasso); library(readr)

n=200  #number of samples
p=5000   #number of covariates
n.sim=100# number of monte carlo simulation
grid <- 10^seq(2,-2,length=50) # range of lambda values for ridge and lasso

 ### container matrix when the measurement error covariance matrix is  Known ###

# container matrix for mean square error(MSE) and l1-error
t.lasso.mse= matrix(NA,nrow = n.sim , ncol = 3); t.lasso.l1=matrix(NA,nrow = n.sim , ncol = 3)
t.dantzig.mse=matrix(NA,nrow = n.sim , ncol = 3); t.dantzig.l1=matrix(NA,nrow = n.sim , ncol = 3)
n.lasso.mse=matrix(NA,nrow = n.sim , ncol = 3) ; n.lasso.l1=matrix(NA,nrow = n.sim , ncol = 3)
cor.lasso.mse=matrix(NA,nrow = n.sim , ncol = 3); cor.lasso.l1=matrix(NA,nrow = n.sim , ncol = 3)
coco.lasso.mse=matrix(NA,nrow = n.sim , ncol = 3); coco.lasso.l1=matrix(NA,nrow = n.sim , ncol = 3)
MUS.mse=matrix(NA,nrow = n.sim , ncol = 3); MUS.l1=matrix(NA,nrow = n.sim , ncol = 3)

#container matrix for prediction error (PE) and number of correct variables selected
t.lasso.pe= matrix(NA,nrow = n.sim , ncol = 3);t.lasso.ncc=matrix(NA,nrow = n.sim , ncol = 3)
t.dantzig.pe=matrix(NA,nrow = n.sim , ncol = 3);t.dantzig.ncc=matrix(NA,nrow = n.sim , ncol = 3)
n.lasso.pe=matrix(NA,nrow = n.sim , ncol = 3) ;n.lasso.ncc=matrix(NA,nrow = n.sim , ncol = 3)
cor.lasso.pe=matrix(NA,nrow = n.sim , ncol = 3);cor.lasso.ncc=matrix(NA,nrow = n.sim , ncol = 3)
coco.lasso.pe=matrix(NA,nrow = n.sim , ncol = 3);coco.lasso.ncc=matrix(NA,nrow = n.sim , ncol = 3)
MUS.pe=matrix(NA,nrow = n.sim , ncol = 3); MUS.ncc=matrix(NA,nrow = n.sim , ncol = 3)

#container matrix for the number of variables selected
t.lasso.nsv=matrix(NA,nrow = n.sim , ncol = 3) ;t.dantzig.nsv=matrix(NA,nrow = n.sim , ncol = 3)
n.lasso.nsv=matrix(NA,nrow = n.sim , ncol = 3);cor.lasso.nsv=matrix(NA,nrow = n.sim , ncol = 3)
coco.lasso.nsv=matrix(NA,nrow = n.sim , ncol = 3);MUS.nsv=matrix(NA,nrow = n.sim , ncol = 3)

#### container matrix when measurement error covariance matrix is unknown ###
# for mean square error(MSE) and l1-error
cor.lasso.mse2=matrix(NA,nrow = n.sim , ncol = 3);cor.lasso.l1_2=matrix(NA,nrow = n.sim , ncol = 3)
coco.lasso.mse2=matrix(NA,nrow = n.sim , ncol = 3);coco.lasso.l1_2=matrix(NA,nrow = n.sim , ncol = 3)
# for prediction error (PE) and number of correct variables selected
cor.lasso.pe2=matrix(NA,nrow = n.sim , ncol = 3);cor.lasso.ncc2=matrix(NA,nrow = n.sim , ncol = 3)
coco.lasso.pe2=matrix(NA,nrow = n.sim , ncol = 3);coco.lasso.ncc2=matrix(NA,nrow = n.sim , ncol = 3)
#for number of variables selected
cor.lasso.nsv2=matrix(NA,nrow = n.sim , ncol = 3);coco.lasso.nsv2=matrix(NA,nrow = n.sim , ncol = 3)

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
    Sigma_U<-diag(x=0.90 ,nrow = p, ncol = p)
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
                         nfolds = 5,family="gaussian")
    t.lasso.fit <- glmnet(X,y,alpha=1,lambda=cv.t.lasso$lambda.min,family="gaussian")
    hat.beta<-coef(t.lasso.fit)[-1,]
    ## model accuracy :
    m=i
    #l1-error, MSE , and prediction error
    t.lasso.mse[m,l]<-sum((hat.beta-beta)^2)/n  
    t.lasso.l1[m,l]<-sum(abs(hat.beta-beta))
    t.lasso.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
   # nber of selected and nber of correct variables selected
    t.lasso.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
    t.lasso.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
    
    ######## Fit Dantzig selector  on true model
    
    # 10-folds cross-validation (DS)
    ds.fit<-gds(X,y,lambda =cv.t.lasso$lambda.min,family = "gaussian")
    hat.beta<-coef(ds.fit,all=T)[,2]
    ## model accuracy :
    #l1-error, MSE , and prediction error
    t.dantzig.mse[m,l]<-sum((hat.beta-beta)^2)/n  
    t.dantzig.l1[m,l]<-sum(abs(hat.beta-beta))
    t.dantzig.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
    # nber of selected and nber of correct variables selected
    t.dantzig.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
    t.dantzig.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
    
    ######## Naive lasso fit
    
    # 10-fold cross-validation on Lasso to find best of 100 lambda values
    cv.n.lasso <-cv.glmnet(W,y,alpha=1,lambda=grid,
                           nfolds = 5,family="gaussian")
    n.lasso.fit <- glmnet(W,y,alpha=1,lambda=cv.n.lasso$lambda.min,family="gaussian")
    hat.beta<-coef(n.lasso.fit)[-1,]
    ## model accuracy :
    #l1-error, MSE , and prediction error
    n.lasso.mse[m,l]<-sum((hat.beta-beta)^2)/n  
    n.lasso.l1[m,l]<-sum(abs(hat.beta-beta))
    n.lasso.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
    # nber of selected and nber of correct variables selected
    n.lasso.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
    n.lasso.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
    
    
    ######## fit corrected Lasso  ( Non-convex lasso or NCL)
    
    # select optimal R using naive lasso in a range [R_max/500,2*R_max]
    cv.fit<-cv_corrected_lasso(W,y,Sigma_U,no_radii=100,n_folds =5)
    R<-cv.fit$radius_min
    # corrected lasso with optimal R selected
    cor.lasso.fit<-corrected_lasso(W,y,Sigma_U,family = "gaussian",
                                   radii =R )
    hat.beta<-rep(0,p)
    hat.beta[coef(cor.lasso.fit)$coefficient]<-coef(cor.lasso.fit)$estimate
    
    ## model accuracy :
    #l1-error, MSE , and prediction error
    cor.lasso.mse[m,l]<-sum((hat.beta-beta)^2)/n 
    cor.lasso.l1[m,l]<-sum(abs(hat.beta-beta))
    cor.lasso.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
    # nber of selected and nber of correct variables selected
    cor.lasso.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
    cor.lasso.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
    
   
   ########### fit convex conditional lasso (CoColasso) model
    
    coco.lasso.fit<-coco(Z=W,y=y,n=n,p=p,center.Z = FALSE, scale.Z = FALSE, step=100,K=5,mu=10,tau=0.75,etol = 1e-3,noise = "additive", block=FALSE, penalty="lasso",center.y = F,scale.y = F)
    hat.beta<-coco.lasso.fit$beta.opt
    
    ## model accuracy :
    #l1-error, MSE , and prediction error
    coco.lasso.mse[m,l]<-sum((hat.beta-beta)^2)/n 
    coco.lasso.l1[m,l]<-sum(abs(hat.beta-beta))
    coco.lasso.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
    # nber of selected and nber of correct variables selected
    coco.lasso.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
    coco.lasso.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
    
   ########## Matrix uncertainty selector 
   
   # the optimal lambda is selected by the cross-validation of the naive lasso
  
   # selected "delta" according to the "elbow" rule
   
   # use the selected delta to compute the final estimate
   mus.fit<-mus(W,y,delta=0.1)
   # coefficient estimated
   hat.beta<-coef(mus.fit,all=TRUE)$estimate
   
   ## model accuracy :
   #l1-error, MSE , and prediction error
   MUS.mse[m,l]<-sum((hat.beta-beta)^2)/n  
   MUS.l1[m,l]<-sum(abs(hat.beta-beta))
   MUS.pe[m,l]<-sum((X%*%(hat.beta-beta))^2)/n
   # nber of selected and nber of correct variables selected
   MUS.nsv[m,l]<-length(hat.beta[which(hat.beta!=0)])
   MUS.ncc[m,l]<-sum(hat.beta[which(beta!=0)]!=0)
  
  }
}
    

################################################################################
# We got result matrix of MSE, PE, L1-Error and correct variable selected for each simulation
###############################################################################

## we take column mean to get the Average MSE over n.sim simulations and
#create an outcome object where the rows contain "t.lasso, t.dantzig, n.lasso,
# cor.lasso, cocolasso and MUS" results respectively

Ave.MSE.results<-rbind(apply(t.lasso.mse ,2,mean),apply(t.dantzig.mse ,2,mean),
apply(n.lasso.mse ,2,mean), apply(cor.lasso.mse ,2,mean),
apply(coco.lasso.mse ,2,mean), apply(MUS.mse ,2,mean) )
rownames(Ave.MSE.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                    "CoCoLasso","MUS")
colnames(Ave.MSE.results)<-c("s=5","s=10","s=20")
Ave.MSE.results
### Now proceed the same way and store Standard deviation ME over n.sim

Sd.MSE.results<-rbind(apply(t.lasso.mse ,2,sd),apply(t.dantzig.mse ,2,sd),
                       apply(n.lasso.mse ,2,sd), apply(cor.lasso.mse ,2,sd),
                       apply(coco.lasso.mse ,2,sd), apply(MUS.mse ,2,sd) )
rownames(Ave.MSE.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                             "CoCoLasso","MUS")
colnames(Ave.MSE.results)<-c("s=5","s=10","s=20")
Sd.MSE.results
### outcome object containing results Average L1-error over n.sim simulations
Ave.L1.results<-rbind(apply(t.lasso.l1 ,2,mean),apply(t.dantzig.l1 ,2,mean),
                       apply(n.lasso.l1 ,2,mean), apply(cor.lasso.l1 ,2,mean),
                       apply(coco.lasso.l1 ,2,mean), apply(MUS.l1 ,2,mean) )
rownames(Ave.L1.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                             "CoCoLasso","MUS")
colnames(Ave.L1.results)<-c("s=5","s=10","s=20")
Ave.L1.results
### object containing the results Standard deviation L1-error over n.sim simultations
Sd.L1.results<-rbind(apply(t.lasso.l1 ,2,sd),apply(t.dantzig.l1 ,2,sd),
                      apply(n.lasso.l1 ,2,sd), apply(cor.lasso.l1 ,2,sd),
                      apply(coco.lasso.l1 ,2,sd), apply(MUS.l1 ,2,sd) )
rownames(Sd.L1.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                            "CoCoLasso","MUS")
colnames(Sd.L1.results)<-c("s=5","s=10","s=20")
Ave.L1.results
### outcome object containing results Average Prediction Error over n.sim simulations
Ave.PE.results<-rbind(apply(t.lasso.pe ,2,mean),apply(t.dantzig.pe ,2,mean),
                      apply(n.lasso.pe ,2,mean), apply(cor.lasso.pe ,2,mean),
                      apply(coco.lasso.pe ,2,mean), apply(MUS.pe ,2,mean) )
rownames(Ave.PE.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                            "CoCoLasso","MUS")
colnames(Ave.PE.results)<-c("s=5","s=10","s=20")
Ave.PE.results
### outcome object containing results Sd of Prediction Error over n.sim simulations
Sd.PE.results<-rbind(apply(t.lasso.pe ,2,sd),apply(t.dantzig.pe ,2,sd),
                      apply(n.lasso.pe ,2,sd), apply(cor.lasso.pe ,2,sd),
                      apply(coco.lasso.pe ,2,sd), apply(MUS.pe ,2,sd) )
rownames(Sd.PE.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                            "CoCoLasso","MUS")
colnames(Sd.PE.results)<-c("s=5","s=10","s=20")
Sd.PE.results
### outcome object containing results Average selected variables over n.sim simulations
Ave.NSV.results<-rbind(apply(t.lasso.nsv ,2,mean),apply(t.dantzig.nsv ,2,mean),
                      apply(n.lasso.nsv,2,mean), apply(cor.lasso.nsv,2,mean),
                      apply(coco.lasso.nsv ,2,mean), apply(MUS.nsv,2,mean) )
rownames(Ave.NSV.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                            "CoCoLasso","MUS")
colnames(Ave.NSV.results)<-c("s=5","s=10","s=20")
Ave.NSV.results
### outcome object containing results Sd selected variables over n.sim simulations
Sd.NSV.results<-rbind(apply(t.lasso.nsv ,2,sd),apply(t.dantzig.nsv ,2,sd),
                       apply(n.lasso.nsv,2,sd), apply(cor.lasso.nsv,2,sd),
                       apply(coco.lasso.nsv ,2,sd), apply(MUS.nsv,2,sd) )
rownames(Sd.NSV.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                             "CoCoLasso","MUS")
colnames(Sd.NSV.results)<-c("s=5","s=10","s=20")
Sd.NSV.results
### outcome object containing results Average of correct variables
#  over n.sim simulations
Ave.NCC.results<-rbind(apply(t.lasso.ncc ,2,mean),apply(t.dantzig.ncc ,2,mean),
                       apply(n.lasso.ncc,2,mean), apply(cor.lasso.ncc,2,mean),
                       apply(coco.lasso.ncc ,2,mean), apply(MUS.ncc,2,mean) )
rownames(Ave.NCC.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                             "CoCoLasso","MUS")
colnames(Ave.NCC.results)<-c("s=5","s=10","s=20")
Ave.NCC.results
### outcome object containing results Sd of correct variables
#  over n.sim simulations
Sd.NCC.results<-rbind(apply(t.lasso.ncc,2,sd),apply(t.dantzig.ncc ,2,sd),
                       apply(n.lasso.ncc,2,sd), apply(cor.lasso.ncc,2,sd),
                       apply(coco.lasso.ncc ,2,sd), apply(MUS.ncc,2,sd) )
rownames(Sd.NCC.results)<-c("True lasso","DS","Naive lasso","Non convex lasso",
                             "CoCoLasso","MUS")
colnames(Sd.NCC.results)<-c("s=5","s=10","s=20")
Sd.NCC.results

############################## Resume3 #######################################
Simulation.Results<-list(Ave.MSE= Ave.MSE.results,Sd.MSE= Sd.MSE.results,
Ave.L1_Error=Ave.L1.results,Sd.L1_Error=Sd.L1.results ,
Ave.PE=Ave.PE.results ,Sd.PE=Sd.PE.results , Ave.NSV= Ave.NSV.results,
Sd.NSV=Sd.NSV.results , Ave.NCC= Ave.NCC.results,Sd.NCC=Sd.NCC.results)

Simulation.Results

# export all the container matrix
save(t.lasso.mse,t.lasso.l1,t.lasso.pe,t.lasso.nsv,t.lasso.ncc, file = "true_lasso_3.RData")
save(n.lasso.mse,n.lasso.l1,n.lasso.pe,n.lasso.nsv,n.lasso.ncc,file="naive_lasso_3.RData")
save(t.dantzig.mse,t.dantzig.l1,t.dantzig.pe,t.dantzig.pe,t.dantzig.nsv,t.dantzig.ncc,file = "dantzig_3.RData")
save(cor.lasso.mse,cor.lasso.l1,cor.lasso.pe,cor.lasso.nsv,cor.lasso.ncc,file = "corrected_lasso_3.RData")
save(coco.lasso.mse,coco.lasso.l1,coco.lasso.pe,coco.lasso.nsv,coco.lasso.ncc,file = "coco_lasso_3.RData")
save(MUS.mse,MUS.l1,MUS.pe,MUS.nsv,MUS.ncc,file = "mus_3.RData")


# Export the result ( Average and Standard deviation)
library(readr)
write.csv2(Ave.MSE.results,file = "Ave.MSE_3.csv")
write.csv2(Sd.MSE.results,file = "Sd.MSE_3.csv")
write.csv2(Ave.L1.results,file = "Ave.L1_Erro_3.csv")
write.csv2(Sd.L1.results,file = "Sd.L1_Error_3.csv")
write.csv2(Ave.PE.results,file = "Ave.PE_3.csv")
write.csv2(Sd.PE.results,file = "Sd.PE_3.csv")
write.csv2(Ave.NSV.results,file = "Ave.NSV_3.csv")
write.csv2(Sd.NSV.results,file = "Sd.NSV_3.csv")
write.csv2(Ave.NCC.results,file = "Ave.NCC_3.csv")
write.csv2(Sd.NCC.results,file = "Sd.NCC_3.csv")


############################################################################

################ Save the models

saveRDS(t.lasso.fit,file = "true.lasso.model.rds")
saveRDS(n.lasso.fit,file = "naive.lasso.model.rds")
saveRDS(ds.fit,file = "true.Dantzig.model.rds")
saveRDS( cor.lasso.fit,file = "corrected.lasso.model.rds")
saveRDS( coco.lasso.fit,file = "CoCo.lasso.model.rds")
saveRDS( mus.fit,file = "MUS.model.rds")
