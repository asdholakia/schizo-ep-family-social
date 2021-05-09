data {
  int<lower=0> N;          // number of respondents
  int<lower=0> K;          // number of dims (analog to PC)
  int<lower=0,upper=1> y[N];             // psychotic episode (binary)
  matrix[N, K] X;          // design matrix
}
parameters {
  real alpha;
  vector[K] beta; 
}
model {
  //priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  
  //likelihood
  y ~ bernoulli_logit(alpha+X*beta);
}
generated quantities {
  vector[N] log_lik;
  vector[N] logit_ep;

  for (n in 1:N){
    log_lik[n] = bernoulli_logit_lpmf(y | alpha+X[n,]*beta);
    logit_ep[n] = bernoulli_logit_rng(alpha+X[n,]*beta);
  }
}