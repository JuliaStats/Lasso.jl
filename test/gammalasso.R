# creates test data for gammalasso.jl

library(gamlr)

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

for(f in 1:3) {
  family <- families[[f]]
  randfun <- randfuns[[f]]
  invlink <- family$linkinv
  yp <- randfun(n,invlink(Ey))
  
  ## run models to extra small lambda 1e-3xlambda.start
  fitlasso <- gamlr(x, yp, gamma=0, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE) # lasso
  fitgl <- gamlr(x, yp, gamma=2, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE) # small gamma
  fitglbv <- gamlr(x, yp, gamma=10, lambda.min.ratio=1e-3, family=family$family, standardize = TRUE) # big gamma
  
#   par(mfrow=c(1,3))
#   ylim = range(c(fitglbv$beta@x))
#   plot(fitlasso, ylim=ylim, col="navy")
#   plot(fitgl, ylim=ylim, col="maroon")
#   plot(fitglbv, ylim=ylim, col="darkorange")
  
  # export test data
  data = cbind(yp, x)
  familyfilename <- paste0("~/.julia/v0.4/Lasso/test/data/gamlr.",family$family)
  write.table(data,file = paste0(familyfilename,".data.csv"), sep=",",row.names=FALSE,col.names=FALSE)
  
  # export estimates
  for(fitname in c("fitlasso", "fitgl", "fitglbv")) {
    fit <- eval(parse(text=fitname))
    fitfilename <- paste0(familyfilename,".",fitname)
    
    fittable <- data.frame(fit$lambda,fit$df,fit$deviance,fit$alpha)
    write.table(fittable,file = paste0(fitfilename,".fit.csv") ,sep=",",row.names=FALSE)
    
    coefs <- as.matrix(fit$beta)
    write.table(coefs,file = paste0(fitfilename,".coefs.csv") ,sep=",",row.names=FALSE,col.names=FALSE)
    
    params <- data.frame(fit$gamma,fit$family)
    write.table(params,file = paste0(fitfilename,".params.csv") ,sep=",",row.names=FALSE)
  }
}
