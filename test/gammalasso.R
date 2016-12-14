# creates test data for gammalasso.jl

library(gamlr)

# automatic way only works with source("script.R"), not with RStudio
script.dir <- dirname(sys.frame(1)$ofile)
testdata.dir <- paste0(script.dir,"/data")

# uncomment to set manually
# testdata.dir <- "~/.julia/v0.4/Lasso/test/data"

setwd(testdata.dir)

### a low-D test (highly multi-collinear)

set.seed(13)
n <- 1000
p <- 3
xvar <- matrix(0.9, nrow=p,ncol=p)
diag(xvar) <- 1
x <- matrix(rnorm(p*n), nrow=n)%*%chol(xvar)
Ey <- 4 + 3*x[,1] + -1*x[,2]
y <- Ey + rnorm(n)

rbinom1 <- function(n,p) {rbinom(n,1,p)}
pop.var <- function(x) var(x) * (length(x)-1) / length(x)
pop.sd <- function(x) sqrt(pop.var(x))

families <- list(gaussian(link="identity"),binomial(link="logit"),poisson(link="log"))
randfuns <- list(rnorm,rbinom1,rpois)
gammas <- c(0, 2, 10)

for(f in 1:3) {
  # f <- 3
  # gamma <- 2
  
  family <- families[[f]]
  print(family)
  randfun <- randfuns[[f]]
  invlink <- family$linkinv
  yp <- randfun(n,invlink(Ey))
  if (family$family == "gaussian") {
    print("standardizing y for comparability with glmnet")
    yp <- yp / pop.sd(yp)
  }
  
  # export test data
  data = cbind(yp, x)
  familyfilename <- paste0("gamlr.",family$family)
  write.table(data,file = paste0(familyfilename,".data.csv"), sep=",",row.names=FALSE,col.names=FALSE)
  
  # export estimates
  for(gamma in gammas) {
    # AICc selection
    fit <- gamlr(x, yp, gamma=gamma, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE)
    fitname <- paste0("gamma",gamma)
    fitfilename <- paste0(familyfilename,".",fitname)
    
    fit$AICc <- AICc(fit)
    fit$logLik <- logLik(fit)
    fittable <- data.frame(fit$lambda,fit$df,fit$deviance,fit$alpha,fit$logLik,fit$AICc)
    write.table(fittable,file = paste0(fitfilename,".fit.csv") ,sep=",",row.names=FALSE)
    
    coefs <- as.matrix(fit$beta)
    write.table(coefs,file = paste0(fitfilename,".coefs.csv") ,sep=",",row.names=FALSE,col.names=FALSE)
    
    # now 10-fold cv
    cvfit <- cv.gamlr(x, yp, gamma=gamma, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE, nfold=10)

    coefs_cvmin <- as.matrix(coef(cvfit,select="min"))
    write.table(coefs_cvmin,file = paste0(fitfilename,".coefs.CVmin.csv") ,sep=",",row.names=FALSE,col.names=FALSE)
    
    coefs_cv1se <- as.matrix(coef(cvfit,select="1se"))
    write.table(coefs_cv1se,file = paste0(fitfilename,".coefs.CV1se.csv") ,sep=",",row.names=FALSE,col.names=FALSE)

    # both fit's params
    params <- data.frame(fit$gamma,fit$family,cvfit$seg.min,cvfit$seg.1se,cvfit$lambda.min,cvfit$lambda.1se)
    write.table(params,file = paste0(fitfilename,".params.csv") ,sep=",",row.names=FALSE)

    }
}
