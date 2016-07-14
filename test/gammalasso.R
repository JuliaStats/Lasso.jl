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

families <- list(gaussian(link="identity"),binomial(link="logit"),poisson(link="log"))
randfuns <- list(rnorm,rbinom1,rpois)
gammas <- c(0, 2, 10)

for(f in 1:3) {
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
    fit <- gamlr(x, yp, gamma=gamma, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE)
    fitname <- paste0("gamma",gamma)
    fitfilename <- paste0(familyfilename,".",fitname)
    
    fittable <- data.frame(fit$lambda,fit$df,fit$deviance,fit$alpha)
    write.table(fittable,file = paste0(fitfilename,".fit.csv") ,sep=",",row.names=FALSE)
    
    coefs <- as.matrix(fit$beta)
    write.table(coefs,file = paste0(fitfilename,".coefs.csv") ,sep=",",row.names=FALSE,col.names=FALSE)
    
    params <- data.frame(fit$gamma,fit$family)
    write.table(params,file = paste0(fitfilename,".params.csv") ,sep=",",row.names=FALSE)
  }
}
