
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;
using namespace arma;


//---------------------------------------------------------------------------------------------------
// a transformed sample implementation taken from Rcpp Gallery:
// https://gallery.rcpp.org/articles/using-the-Rcpp-based-sample-implementation/
// fixed to one draw, sampling without replacement, and changed output type to int
// IMPORTANT: always #include <RcppArmadilloExtensions/sample.h>
//---------------------------------------------------------------------------------------------------
// [[Rcpp::export]]
int csample_num1 (
    Rcpp::NumericVector x,
    Rcpp::NumericVector prob = NumericVector::create()
) {
  bool replace = false;
  NumericVector ret = Rcpp::RcppArmadillo::sample(x, 1, replace, prob);
  int out           = ret(0);
  return out;
} // END csample_num1


// [[Rcpp::export]]
arma::mat count_regime_transitions (
    const arma::mat& xi
) {
  const int M = xi.n_rows;
  const int T = xi.n_cols;
  
  mat count(M, M);
  urowvec s   = index_max( xi, 0 );
  for (int t=1; t<T; t++) {
    count( s(t-1), s(t))++;
  }
  return count;
} // END count_regime_transitions

// [[Rcpp::export]]
arma::rowvec rDirichlet1 (
    const arma::rowvec&   alpha     // Kx1
) {
  const int K   = alpha.size();
  rowvec    draw(K);
  for (int k=0; k<K; k++) {
    draw(k)     = randg(distr_param(alpha(k), 1.0));
  }
  return draw/sum(draw);
} // END rDirichlet1


// [[Rcpp::export]]
arma::mat filtering (
    const arma::cube& Z,                  // NxTxM state-specific standardised residuals
    const arma::mat&  aux_PR_TR,          // MxM
    const arma::vec&  pi_0                // Mx1
) {
  
  // filtered probabilities for a model with MS structural matrix of SVAR-MSS-SV model
  
  const int   T = Z.n_cols;
  const int   N = Z.n_rows;
  const int   M = aux_PR_TR.n_rows;
  
  mat         eta_t(M, T);
  mat         xi_t_t(M, T);
  
  // This loop evaluates mvnormal pdf at Z being multivariate standard normal distribution
  for (int m=0; m<M; m++) {
    rowvec log_d    = -0.5 * sum(square(Z.slice(m)), 0);
    log_d          += -0.5 * N * log(2*M_PI);
    NumericVector   exp_log_d   = wrap(exp(log_d));
    exp_log_d[exp_log_d==0]     = 1e-300;
    eta_t.row(m)    = as<rowvec>(exp_log_d);
  } // END m loop
  
  vec xi_tm1_tm1    = pi_0;
  
  for (int t=0; t<T; t++) {
    vec     num     = eta_t.col(t) % (aux_PR_TR.t() * xi_tm1_tm1);
    double  den     = sum(num);
    xi_t_t.col(t)   = num/den;
    xi_tm1_tm1      = xi_t_t.col(t);
  } // END t loop
  
  return xi_t_t;
} // END filtering


// [[Rcpp::export]]
arma::mat smoothing (
    const arma::mat&  filtered,           // MxT
    const arma::mat&  aux_PR_TR           // MxM
) {
  // NOT the same as for msh (but could be the same if you get rid of arg U in the other one)
  const int   T = filtered.n_cols;
  const int   M = aux_PR_TR.n_rows;
  
  mat   smoothed(M, T);
  smoothed.col(T-1)   = filtered.col(T-1);
  
  for (int t=T-2; t>=0; --t) {
    smoothed.col(t)   = (aux_PR_TR * (smoothed.col(t+1)/(aux_PR_TR.t() * filtered.col(t)) )) % filtered.col(t);
  } // END t loop
  
  return smoothed;
} // smoothing


// [[Rcpp::export]]
List sample_Markov_process (
    const arma::cube&  Z,                  // NxTxM
    arma::mat         aux_xi,             // MxT
    arma::mat&        aux_PR_TR,          // MxM
    arma::vec&        aux_pi_0
) {

  const bool finiteM = true;

  int minimum_regime_occurrences = 10;
  int max_iterations = 50;
  if ( finiteM ) {
    minimum_regime_occurrences = 10;
    max_iterations = 50;
  }

  const int   T   = Z.n_cols;
  const int   M   = aux_PR_TR.n_rows;
  mat aux_xi_tmp = aux_xi;
  mat aux_xi_out = aux_xi;

  const mat   prior_PR_TR = eye(M, M) + 1;

  mat filtered    = filtering(Z, aux_PR_TR, aux_pi_0);
  mat smoothed    = smoothing(filtered, aux_PR_TR);
  mat    aj       = eye(M, M);

  mat xi(M, T);
  int draw        = csample_num1(wrap(seq_len(M)), wrap(smoothed.col(T-1)));
  aux_xi_tmp.col(T-1)     = aj.col(draw-1);

  if ( minimum_regime_occurrences==0 ) {
    for (int t=T-2; t>=0; --t) {
      vec xi_Tmj    = (aux_PR_TR * (aux_xi.col(t+1)/(aux_PR_TR.t() * filtered.col(t)))) % filtered.col(t);
      draw          = csample_num1(wrap(seq_len(M)), wrap(xi_Tmj));
      aux_xi_tmp.col(t)   = aj.col(draw-1);
    }
    aux_xi_out = aux_xi_tmp;
  } else {
    int regime_occurrences  = 1;
    int iterations  = 1;
    while ( (regime_occurrences<minimum_regime_occurrences) & (iterations<max_iterations) ) {
      for (int t=T-2; t>=0; --t) {
        vec xi_Tmj    = (aux_PR_TR * (aux_xi.col(t+1)/(aux_PR_TR.t() * filtered.col(t)))) % filtered.col(t);
        draw          = csample_num1(wrap(seq_len(M)), wrap(xi_Tmj));
        aux_xi_tmp.col(t)   = aj.col(draw-1);
      }
      mat transitions       = count_regime_transitions(aux_xi_tmp);
      regime_occurrences    = min(transitions.diag());
      iterations++;
    } // END while
    if ( iterations<max_iterations ) aux_xi_out = aux_xi_tmp;
  }

  mat transitions       = count_regime_transitions(aux_xi);
  mat posterior_alpha   = transitions + prior_PR_TR;
  
  for (int m=0; m<M; m++) {
    aux_PR_TR.row(m)    = rDirichlet1(posterior_alpha.row(m));
  }
  vec prob_xi1          = aux_PR_TR *aux_xi.col(0);
  prob_xi1             /= sum(prob_xi1);
  int S0_draw           = csample_num1(wrap(seq_len(M)), wrap(prob_xi1));
  rowvec posterior_alpha_0(M, fill::value((1.0)));
  posterior_alpha_0(S0_draw-1)++;
  aux_pi_0              = trans(rDirichlet1(posterior_alpha_0));

  return List::create(
    _["aux_PR_TR"]        = aux_PR_TR,
    _["aux_pi_0"]         = aux_pi_0,
    _["aux_xi"]           = aux_xi_out
  );
} // END sample_Markov_process

/*** R
xi = diag(2)[,sample(1:2,100,replace=TRUE)]
count_regime_transitions(xi)

Z = array(rnorm(3 * 200 * 2), c(2,200,2))
Z[,1:100,1] = Z[,1:100,1] * 2
Z[,101:200,2] = Z[,101:200,2] * 2
xi = rbind(c(rep(1,50), rep(0,150)), c(rep(0,50), rep(1,150)))
P = 0.95 * diag(2) + 0.025
pi = c(0.5, 0.5)

gibbs_draw = sample_Markov_process(Z, xi, P, pi)
plot.ts(t(gibbs_draw$aux_xi))

*/