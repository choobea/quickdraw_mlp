## 
##
## -------------------------------------------
## LOAD PACKAGES
## -------------------------------------------
##
## 
## Load any R package required for your code
## to successfully run!

library(tidyverse)
library(SpatialExtremes)
## ;;
#setwd("")

## -------------------------------------------
## READING DATA
## -------------------------------------------
## ;;
#load("burlington.RData")
## ls()                     ## see contents of 
##                          ## burlington object


## --------------------------------------------------
## --------------------------------------------------

## ;;
## ---------------------------------------------
## Q1: -- add your code below 
## ---------------------------------------------
## ;;

## 1.1
testScores <- list(  lsat = c(576, 635, 558, 578, 666, 580, 555,
                            661, 651, 605, 653, 575, 545, 572, 594),
                     gpa =  c(3.39, 3.30, 2.81, 3.03, 3.55, 3.07, 3.00,
                              3.43, 3.36, 3.13, 3.12, 2.74, 2.76, 2.88, 2.96),
                     n = 15)

CI.var.cor <- function(x, y, R = 1000, alpha = 0.05) {
  ## Bootstrap Basic and Percentage confidence intervals for correlation
  ##
  ## x - first covariate
  ## y - second covariate
  ## R - number of bootstrap samples to produce
  ## alpha - confidence level
  ##
  ## Returns confidence intervals
  
  # data needs to be the same length
  n <- length(x)
  n2 <- length(y)
  stopifnot(n == n2)
  stopifnot(n > 1)
  
  # Sample estimate
  sampleCor <- cor(x, y)
  
  # Collect bootstrap estimates
  index <- 1:n
  corr <- array(dim = R)
  for (i in 1:R){
    sample_index <- sample(index, n, replace = TRUE)
    x_bootstrap <- x[sample_index]
    y_bootstrap <- y[sample_index]
    corr[i] <- cor(x_bootstrap, y_bootstrap)
  }
  
  # Interval Calculation
  quantiles <- c(alpha/2, 1 - alpha/2)
  bootstrapPercentage <- quantile(corr, quantiles)
  bootstrapBasic <- c(2*sampleCor - bootstrapPercentage[2],
                      2*sampleCor - bootstrapPercentage[1])
  
  output <- rbind(bootstrapBasic, bootstrapPercentage)
  colnames(output) <- c(alpha/2,1 - alpha/2)
  
  # Output
  output
}

out <- CI.var.cor(testScores$lsat, testScores$gpa, R = 10000, alpha = 0.05)

## 1.2

CI.var.ratio <- function(x, y, R = 1000, alpha = 0.05) {
  ## Bootstrap Basic and Percentage confidence intervals for variance ratio
  ##
  ## x - first covariate
  ## y - second covariate
  ## R - number of bootstrap samples to produce
  ## alpha - confidence level
  ##
  ## Returns confidence intervals
  
  # We need more than 2 observations for variance
  n_x <- length(x)
  n_y <- length(y)
  stopifnot(n_x > 1 || n_y > 1)
  
  # Sample estimate
  sampleRatio <- var(x)/var(y)
  
  # Collect bootstrap estimates
  index <- 1:n_x
  ratios <- array(dim = R)
  for (i in 1:R){
    sample_index <- sample(index, n_x, replace = TRUE)
    x_bootstrap <- x[sample_index]
    y_bootstrap <- y[sample_index]
    # x_bootstrap <- sample(x, n_x, replace = TRUE)
    # y_bootstrap <- sample(y, n_y, replace = TRUE)
    ratios[i] <- var(x_bootstrap)/var(y_bootstrap)
  }
  
  # Interval Calculation
  quantiles <- c(alpha/2, 1 - alpha/2)
  bootstrapPercentage <- quantile(ratios, quantiles)
  bootstrapBasic <- c(2*sampleRatio - bootstrapPercentage[2], 
                      2*sampleRatio - bootstrapPercentage[1])
  
  output <- rbind(bootstrapBasic, bootstrapPercentage)
  colnames(output) <- c(alpha/2,1 - alpha/2)
  
  # Output
  output
}

out <- CI.var.ratio(testScores$lsat, testScores$gpa, R = 1000, alpha = 0.05)


## ;;
## -------------------------------------------
## Q2: -- add your code below  
## -------------------------------------------
## ;;
fit.gpd <- function(x, thresh, tol.xi.limit=5e-2, ...)
{
  llik.gpd <- function(par, x, thresh, tol.xi.limit=5e-2)
  {
    y     <- x[x>thresh]
    sigma <- exp(par[1])
    xi    <- par[2]
    n     <- length(y)
    if(abs(xi)<=tol.xi.limit)
    {
      llik <- n*log(sigma) + sum((y-thresh)/sigma)
      return(llik)
    }
    par.log.max <- log( pmax( 1+xi*(y-thresh)/sigma, 0 ) )
    llik        <- -n*log( sigma )-(1+( 1/xi ))*sum( par.log.max )
    llik        <- -ifelse( llik > -Inf, llik, -1e40 )
    return(llik)
  }
  fit <- optim(par = c(0, 0), fn = llik.gpd,
               x=x, thresh=thresh,
               control=list( maxit=10000, ... ))
  sigmahat <- exp( fit$par[1] )
  xihat    <- fit$par[2]
  return(c(sigmahat, xihat))
}


## 2.1

parboot.gpd <- function(data, thresh, R = 1000) {
  ## Parametric Bootstrap for generalized Pareto
  ##
  ## data - vector of datapoints
  ## thresh - threshold value
  ## R - number of bootstrap samples to produce
  ##
  ## Returns confidence intervals
  
  # MLE estimates
  n <- length(data)
  pHat <- sum(data > thresh)/n
  theta <- fit.gpd(x = data, thresh = thresh)
  
  # Bootstrap generation
  thetaBoot <- array(dim = c(R,3))
  for (i in 1:R){
    # Estimates of p
    xBoot <- sample(data, n, replace = TRUE)
    thetaBoot[i,3] <- sum(xBoot > thresh)/n
    
    # Estimates of sigma and xi
    nI <- rbinom(1, n, pHat)
    excedences <- rgpd(nI, thresh, theta[1], theta[2])
    thetaBoot[i,1:2] <- fit.gpd(x = excedences, thresh = thresh)
  }
  
  # Estimation of bias and st. deviation
  sdev <- apply(thetaBoot, 2, sd)
  bias <- sweep(thetaBoot, 2, c(theta, pHat), '-') %>% apply(2, mean)
  
  # Output
  c(mle = list(c(theta, pHat)), bias = list(bias), 
                  sd = list(sdev), distn = list(thetaBoot))
}

## 2.2

npboot.gpd <- function(data, thresh, R = 1000) {
  ## Parametric Bootstrap for generalized Pareto
  ##
  ## data - vector of datapoints
  ## thresh - threshold value
  ## R - number of bootstrap samples to produce
  ##
  ## Returns confidence intervals
  
  # MLE estimates
  n <- length(data)
  pHat <- sum(data > thresh)/n
  theta <- fit.gpd(x = data, thresh = thresh)
  
  # Bootstrap generation
  thetaBoot <- array(dim = c(R,3))
  for (i in 1:R){
    # Estimates of p
    xBoot <- sample(data, n, replace = TRUE)
    thetaBoot[i,3] <- sum(xBoot > thresh)/n
    
    # Estimates of sigma and xi
    thetaBoot[i,1:2] <- fit.gpd(x = xBoot, thresh = thresh)
  }
  
  # Estimation of bias and st. deviation
  sdev <- apply(thetaBoot, 2, sd)
  bias <- sweep(thetaBoot, 2, c(theta, pHat), '-') %>% apply(2, mean)
  
  # Output
  c(mle = list(c(theta, pHat)), bias = list(bias), 
    sd = list(sdev), distn = list(thetaBoot))
}


## 2.3

# initalise
u = quantile(burlington$Precipitation, 0.9)
R = 1000

# Calculate bootstrap estimates
paramBoot <- parboot.gpd(burlington$Precipitation, u, R)
nonparamBoot <- npboot.gpd(burlington$Precipitation, u, R)

Tau <- c(100, 500, 1000, 10000)

# initialise
xParam <- array(dim = c(R, length(Tau)))
xNonparam <- array(dim = c(R, length(Tau)))

for (i in 1:length(Tau)){
  sigmahat <- paramBoot$distn[,1]
  xihat <- paramBoot$distn[,2]
  phat <- paramBoot$distn[,3]
  xParam[,i] <- u + sigmahat/xihat*((Tau[i]*phat)^xihat - 1)
  
  sigmahat <- nonparamBoot$distn[,1]
  xihat <- nonparamBoot$distn[,2]
  phat <- nonparamBoot$distn[,3]
  xNonparam[,i] <- u + sigmahat/xihat*((Tau[i]*phat)^xihat - 1)
}

# Calculate percentile Confidence intervals
quantiles <- c(0.025, 0.975)
paramCI <- apply(xParam, 2, function(x) quantile(x, quantiles))
nonparamCI <- apply(xNonparam, 2, function(x) quantile(x, quantiles))

## ;;
## -------------------------------------------
## Q3: -- add your code  below ;;
## -------------------------------------------
## ;;

data <- list(t = c(94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5),
              x = c(5, 1, 5, 14, 3, 19, 1, 1, 4, 22),
              n = 10)

## 3.1

# Answers are in the PDF

## 3.2
logAlphaLogDensity <- function(y, theta, beta, lambda) {
  ## log Density for log(alpha)
  ##
  ## y - log alpha variate
  ## theta  - 
  ## beta   - parameters
  ## lambda - 
  ##
  ## Value of log density
  
  n <- length(theta)
  A <- n*(exp(y)*log(beta) - log(gamma(exp(y))))
  B <- exp(y)*sum(log(theta))
  C <- -lambda*exp(y)
  D <- y
  
  return (A + B + C + D)
}

mcmc.pumps <- function(x, t, nburn = 100, ndraw = 10000, propV = 1,
                       alpha0 = 2, beta0 = 2, theta0 = rep(1, each = length(x)),
                       lambda0 = 1, gamma0 = 0.01, delta0 = 0.01) {
  ## MCMC algorith for simulating from the posterior distributions of the parameters
  ##
  ## x - initial vector of data
  ## t - initial vector of times
  ## nburn - number of initial draws to discard
  ## ndraw - number of draws to produce
  ## propV - variance of proposal
  ## alpha0   - 
  ## beta0    - 
  ## theta0   - initial
  ## lambda0  - values
  ## gamma0   -
  ## delta0   -
  ##
  ## Returns list: draws, acceptance rate
  
  n <- length(x)
  logAlpha0 <- log(alpha0)
  beta <- beta0
  theta <- theta0
  
  # Initialisation
  draws <- matrix( nrow = ndraw, ncol = 2 + n) 
  iter <- -nburn
  naccept <- 0
  while(iter < ndraw){
    iter <- iter + 1
    
    # MCMC for alpha
    # Could be done without transformation of variables, here transformation was
    # used for more consistency with general algorithm
    logAlpha <- logAlpha0 + rnorm(1, 0, propV)
    u <- runif(1)
    acceptP <- min(1, exp(logAlphaLogDensity(y = logAlpha, theta, beta, lambda0) 
                          - logAlphaLogDensity(logAlpha0, theta, beta, lambda0)))
    
    if (u < acceptP){
      naccept <- naccept + 1
      logAlpha0 <- logAlpha
    }
    
    alpha <- exp(logAlpha0)
    
    # Gibbs sampler for beta and thetha
    beta <- rgamma(1, n*alpha + gamma0, rate = (sum(theta) + delta0) )
    theta <- rgamma(n, x + alpha, rate = (t + beta))
    
    # Store values
    if(iter > 0)
    {
      draws[iter,1] <- alpha
      draws[iter,2] <- beta
      draws[iter,3:(n+2)] <- theta
    }
  }
  
  return(list(draws, naccept/(nburn+ndraw))) 
}

out <- mcmc.pumps(x = data$x, t = data$t, nburn = 0, ndraw = 100, propV = 1)

## 3.3

predictive.pumps <- function(xstar, i, tstar, x, t, ...) {
  ## Estimate the value of predictive distribution
  ##
  ## xstar - vector of x where evaluate distribution
  ## i - number of pump to generate values
  ## tstar - time period for values generation
  ## x - data for number of breakdowns
  ## t - data for time periods
  ## ... - additional arguments to pass to generator of posterior distributions
  ## of parameters of the model
  ##
  ## Returns vector of values of distribution

  # Generate sample from the posterior distribution of parameters
  params <- mcmc.pumps(x = x, t = t, ...)[[1]]
  thetaEst <- params[, i+2]
  
  dens <- array(dim = length(xstar))
  
  # Estimate posterior density
  for (i in 1:length(xstar)){
    dens[i] <- mean(dpois(xstar[i], thetaEst*tstar))
  }
  
  dens
}

out <- predictive.pumps(xstar = seq(0, 10),i = 10, tstar = 1, x = data$x, t = data$t)

# Solution for generation from the predictive distribution 
#predictive.pumps <- function(n, i, tstar, xstar0 = 0, x, t, nburnx = 0, propL = 4, ...) {
#   ## Generate a sample from the posterior predictive distirbution
#   ## using MCMC
#   ##
#   ## n - number of variates to generate
#   ## i - number of pump to generate values
#   ## tstar - time period for values generation
#   ## xstar0 - initia value for generation
#   ## x - data for number of breakdowns
#   ## t - data for time periods
#   ## nburnx - number of values to discard
#   ## propL - rate for the generator
#   ## ... - additional arguments to pass to generator of posterior distributions
#   ## of parameters of the model
#   ##
#   ## Returns list: draws, acceptance rate
#   
#   # Generate sample from th posterior distribution of parameters
#   params <- mcmc.pumps(x = x, t = t, ...)[[1]]
#   thetaEst <- params[, i+2]
#   
#   # Initialise
#   estimates <- array(dim = n)
#   iter <- -nburnx
#   naccept <- 0
#   xstar <- xstar0
#   
#   while (iter < n){
#     iter <- iter + 1
#     
#     # Propose from a symmetric around xstar integer distribution
#     power <- rbernoulli(1)
#     prop <- xstar + ((-1)^power)*rpois(1, lambda = propL)
#     
#     u <- runif(1)
#     
#     # Estimate posterior density
#     propDensity <- mean(dpois(prop, thetaEst*tstar))
#     xstarDensity <- mean(dpois(xstar, thetaEst*tstar))
#     
#     alpha <- min(1, propDensity/xstarDensity)
#     
#     if (u < alpha){
#       naccept <- naccept + 1
#       xstar <- prop
#     }
#     
#     if (iter > 0 ) estimates[iter] <- xstar
#   }
#   
#   list(estimates, naccept/(nburnx + n))
# }
# 
# out <- predictive.pumps(10000,i = 10, tstar = 1, x = data$x, t = data$t, nburnx = 100, propL = 3)

## ;;
## -------------------------------------------
## Q4: -- add your code below;;
## -------------------------------------------
## ;;

## 4.1

logMuPosterior <- function(mu, x, nu){
  ## log posterior density of mu
  ##
  ## mu - point where to evaluate log density
  ## x - initial vector of data
  ## nu - parameter of the model
  ##
  ## Returns value of log density
  
  stopifnot( nu > 0 )
  # likelihood of the data term
  llik <- (-(nu+1)/2)*log(1 + 1/nu*(x-mu)^2)
  
  # output
  sum(llik) + (-mu^2/2)
}


mcmc.t <- function(x, mu0 = 0, nu = 10, nburn = 100, ndraw = 10000, propV = 0.05) {
  ## MCMC algorith of simulating from the posterior
  ##
  ## x - initial vector of data
  ## mu0 - point where to initialise MC
  ## nu - parameter of the model
  ## nburn - number of initial draws to discard
  ## ndraw - number of draws to produce
  ## propV - variance of proposal 
  ##
  ## Returns list: draws, acceptance rate
  
  # Initalise
  draws <- array( dim = ndraw)
  iter <- -nburn
  naccept <- 0
  
  while (iter < ndraw){
    iter <- iter + 1
    # mu - proposal
    mu <- mu0 + rnorm(1, 0, propV)
    
    # acceptance probability
    alpha <- min(1, exp(logMuPosterior(mu, x, nu) - logMuPosterior(mu0, x, nu)))
    
    # accept/reject
    u <- runif(1)
    if (u < alpha){
      naccept <- naccept + 1
      mu0 <- mu
    }
    
    # store value
    if (iter > 0 ) draws[iter] <- mu0
  }
  
  # output
  list(draws, naccept/(nburn + ndraw))
}

out <- mcmc.t(x = rt(10000, 10, 10))

## 4.2

mcmc.gibbs <- function(x, mu0, z0, nu, nburn = 100, ndraw = 10000) {
  ## Simulating from the posterior by Gibbs sampler
  ##
  ## x - initial vector of data
  ## mu0 - point where to initialise MC
  ## z0 - point where to initialise MC
  ## nu - parameter of the model
  ## nburn - number of initial draws to discard
  ## ndraw - number of draws to produce
  ##
  ## Returns draws, acceptance rate
  
  # Initialisation
  n <- length(z0)
  mu <- mu0
  z <- z0
  draws <- matrix( nrow = ndraw, ncol = 1 + n) 
  iter <- -nburn
  
  while(iter < ndraw){
    iter <- iter + 1
    
    # simulate mu
    muMean <- sum(z*x)/(sum(z)+1)
    muVar <- 1/(sum(z)+1)
    mu <- rnorm(1, muMean, muVar) 
    
    # simulate z
    zShape <- (nu+1)/2
    zRate <- ((x-mu)^2 + nu)/2
    z <- rgamma(n, shape = zShape, rate = zRate)
    
    # store values
    if(iter > 0)
    {
      draws[iter,1] <- mu
      draws[iter,2:(n+1)] <- z 
    }
  }
  
  return(draws) 
}

out <- mcmc.gibbs(x = rt(10000, 10, 1), mu0 = 0, z0 = rep(1, each=10000), nu=10)

## 4.3

density.t <- function(x, mu, nu = 10){
  ## Calculate the value of density of a t-distribution
  ##
  ## x - initial vector of data
  ## mu - parameter of the distribution
  ## nu - parameter of the distribution
  ##
  ## Returns the value of density
  
  (1 + 1/nu*(x - mu)^2)^(-(nu+1)/2)
}

predictive.t <- function(xstar, x, ...) {
  ## Estimate the value of predictive distribution
  ##
  ## xstar - vector of x where evaluate distribution
  ## x - data vector
  ## ... - additional arguments to pass to generator of posterior distributions
  ## of parameters of the model
  ##
  ## Returns vector of values of distribution
  
  # Generate sample from the posterior distribution of parameters
  mu <- mcmc.t(x = x, ...)[[1]]
  
  dens <- array(dim = length(xstar))
  
  # Estimate posterior density by Monte Carlo Integration
  for (i in 1:length(xstar)){
    dens[i] <- mean(density.t(xstar[i], mu))
  }
  
  dens
}

out <- predictive.t(seq(-10, 10, 0.5), x = rt(10000, 10, 1))

## 4.4

predictive.sampler <- function(n, x, nburnx = 100, propVx = 4, ...) {
  ## Generate a sample from the posterior predictive distirbution
  ## using MCMC
  ##
  ## n - number of variates to generate
  ## x - vector of datapoints
  ## nburnx - number of values to discard
  ## propVx - variance the proposal generator
  ## ... - additional arguments to pass to generator of posterior distributions
  ## of parameters of the model
  ##
  ## Returns list: sample, acceptance rate

  # Generate posterior distribution of mu
  mu <- mcmc.t(x = x, ...)[[1]]

  # Initialise
  estimates <- array(dim = n)
  iter <- -nburnx
  naccept <- 0
  xstar <- mean(x)

  while (iter < n){
    iter <- iter + 1
    prop <- xstar + rnorm(1, 0, propVx)
    u <- runif(1)

    # Estimate posterior density using Monte Carlo Integration
    propDensity <- mean(density.t(prop, mu))
    xstarDensity <- mean(density.t(xstar, mu))

    alpha <- min(1, exp(log(propDensity) - log(xstarDensity)))

    if (u < alpha){
      naccept <- naccept + 1
      xstar <- prop
    }

    if (iter > 0 ) estimates[iter] <- xstar
  }

  list(estimates, naccept/(nburnx + n))
}

out <- predictive.sampler(10000, x = rt(10000, 10, 1), nburnx = 100, propVx = 4)

predictive.quantile <- function(p, n = 10000, x, ...) {
  ## Estimate confidence interval for the posterior predictive distribution
  ## through sample generation
  ##
  ## p - vector of probabilities
  ## n - number of variates to generate
  ## x - vector of datapoints
  ## nburnx - number of values to discard
  ## ... - additional arguments to pass to generator of posterior distributions
  ## of parameters of the model
  ##
  ## Return the estimated quantiles
  
  estimates <- predictive.sampler(n, x, ...)
  quantile(estimates[[1]], p)
}

confInt <- predictive.quantile(c(0.025,0.975), x = rt(10000, 10, 1))

## ------------------------------------------------

## ;;
## -------------------------------------------
## DRAFT
## -------------------------------------------
## ;;
## foo <- rnorm(100)
##
##
## 



