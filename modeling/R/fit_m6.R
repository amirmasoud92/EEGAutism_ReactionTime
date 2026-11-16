# fit_m6.R -- M6, the ex-Gaussian distributional model of the RT distribution (exploratory; reported as not
# converged), and its odd/even-block split. Same hand-over as the model engine: a JSON spec + the model-input TSV
# (spline bases rebuilt with splines::ns and compared; engine_common.R).
#
#   bf(rt_ms ~ a_wb * grp + <M1 fixed covariates> + (1 + a_wb |c| pid),
#      sigma ~ grp + (1 |c| pid),
#      beta  ~ a_wb * grp + (1 |c| pid)), family = exgaussian()      brms, rstan backend, default links
#
# Priors: normal(396, 166) mu intercept; normal(0, 166) every mu coefficient; exponential(0.006)
# mu random-effect SDs; normal(3.36, 1) sigma intercept; normal(0, 1) sigma coefficients; exponential(1) sigma SDs;
# normal(4.97, 1) beta intercept; normal(0, 1) beta coefficients; exponential(1) beta SDs; lkj(2) correlations.
# Sampler: chains x iter (warmup iter / 2), seed, adapt_delta 0.95, max_treedepth 12, brms default inits.
# Convergence: R-hat < 1.01 and bulk ESS > 400 for every fixed effect (incl. the distributional intercepts and
# coefficients), random-effect SD and correlation; divergences reported. If it fails: ONE refit with refit_iter
# (warmup refit_iter / 2, refit_seed); if that fails too the result is "not converged" (not interpreted).
#
# mode = "full": the model on every row; quantities: b_a_wb (mu, ms per SD_i), b_a_wb:grp, b_beta_a_wb (log scale
#   of the exponential mean), b_beta_a_wb:grp, b_sigma_grp, and cor(child a_wb slope on mu, child beta intercept)
#   = cor_pid__a_wb__beta_Intercept, each with its posterior mean, SD and 95% equal-tailed interval.
# mode = "split": the same model fitted separately on the odd and on the even blocks of each child
#   (block_order, the order of the child's blocks), same priors, sampler and refit rule. Location slope from the odd
#   blocks, tail from the even blocks: per posterior draw s, the correlation across children (in both fits) of the
#   child's mu a_wb slope (odd fit: b_a_wb + b_a_wb:grp grp_i + r_pid[i, a_wb]) and the child's beta intercept (even
#   fit: b_beta_Intercept + b_beta_grp grp_i + r_pid__beta[i, Intercept]); draws paired by index. Reported: its mean
#   and 95% interval, and the odd-fit b_a_wb and the even-fit b_beta_a_wb. No child-level value is written.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_m6.R <spec.json>")
.self <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(dirname(.self), "engine_common.R"))
suppressPackageStartupMessages({ library(brms); library(rstan) })
rstan::rstan_options(auto_write = TRUE)

t_start <- Sys.time()
spec <- read_spec(args[1])
spec$outcome <- "rt_ms"
inp <- load_input(spec)
d <- inp$d
res <- list(schema = "alpha_state.m6_result/1", label = spec$label, mode = spec$mode,
            ns_check = inp$ns_check)

has_grp <- isTRUE(spec$has_grp)
cov <- if (nzchar(spec$covariate_rhs)) paste("+", spec$covariate_rhs) else ""
lead <- if (has_grp) "a_wb * grp" else "a_wb"
fm <- brms::bf(stats::as.formula(sprintf("rt_ms ~ %s %s + (1 + a_wb |c| pid)", lead, cov)),
               stats::as.formula(sprintf("sigma ~ %s + (1 |c| pid)", if (has_grp) "grp" else "1")),
               stats::as.formula(sprintf("beta ~ %s + (1 |c| pid)", lead)))
pri <- c(brms::prior(normal(396, 166), class = Intercept),
         brms::prior(normal(0, 166), class = b),
         brms::prior(exponential(0.006), class = sd),
         brms::prior(normal(3.36, 1), class = Intercept, dpar = sigma),
         brms::prior(exponential(1), class = sd, dpar = sigma),
         brms::prior(normal(4.97, 1), class = Intercept, dpar = beta),
         brms::prior(normal(0, 1), class = b, dpar = beta),
         brms::prior(exponential(1), class = sd, dpar = beta),
         brms::prior(lkj(2), class = cor))
if (has_grp) pri <- c(pri, brms::prior(normal(0, 1), class = b, dpar = sigma))

diag_of <- function(f) {
  sm <- summary(f)
  rh <- c(sm$fixed$Rhat, unlist(lapply(sm$random, function(z) z$Rhat)))
  es <- c(sm$fixed$Bulk_ESS, unlist(lapply(sm$random, function(z) z$Bulk_ESS)))
  np <- brms::nuts_params(f)
  list(max_rhat = max(rh, na.rm = TRUE), min_bulk_ess = min(es, na.rm = TRUE),
       n_divergent = sum(np$Value[np$Parameter == "divergent__"]),
       converged = (max(rh, na.rm = TRUE) < 1.01) && (min(es, na.rm = TRUE) > 400))
}

fit_rule <- function(dd) {
  att <- list()
  # attempt 1 ran past its wall-clock budget: counted as not converged, go straight to the refit
  if (!is.null(spec$skip_first) && nzchar(spec$skip_first)) {
    att[[1]] <- list(iter = spec$iter, warmup = spec$warmup, seed = spec$seed, converged = FALSE,
                     failed = spec$skip_first)
    cap <- .cap(brms::brm(fm, data = dd, family = brms::exgaussian(), prior = pri, chains = as.integer(spec$chains),
                          iter = as.integer(spec$refit_iter), warmup = as.integer(spec$refit_warmup),
                          seed = as.integer(spec$refit_seed), cores = as.integer(spec$chains),
                          control = list(adapt_delta = spec$adapt_delta, max_treedepth = as.integer(spec$max_treedepth)),
                          refresh = 0, silent = 2, backend = "rstan"))
    if (inherits(cap$val, "fit_error")) return(list(error = cap$val$msg))
    f <- cap$val
    dg <- diag_of(f)
    att[[2]] <- c(list(iter = spec$refit_iter, warmup = spec$refit_warmup, seed = spec$refit_seed), dg)
    return(list(fit = f, attempts = att, converged = isTRUE(dg$converged), warnings = unique(c(cap$warn))))
  }
  cap <- .cap(brms::brm(fm, data = dd, family = brms::exgaussian(), prior = pri, chains = as.integer(spec$chains),
                        iter = as.integer(spec$iter), warmup = as.integer(spec$warmup), seed = as.integer(spec$seed),
                        cores = as.integer(spec$chains), control = list(adapt_delta = spec$adapt_delta,
                                                                         max_treedepth = as.integer(spec$max_treedepth)),
                        refresh = 0, silent = 2, backend = "rstan"))
  if (inherits(cap$val, "fit_error")) return(list(error = cap$val$msg))
  f <- cap$val
  dg <- diag_of(f)
  att[[1]] <- c(list(iter = spec$iter, warmup = spec$warmup, seed = spec$seed), dg)
  if (!isTRUE(dg$converged)) {
    cap2 <- .cap(stats::update(f, iter = as.integer(spec$refit_iter), warmup = as.integer(spec$refit_warmup),
                               seed = as.integer(spec$refit_seed), chains = as.integer(spec$chains),
                               cores = as.integer(spec$chains), refresh = 0, silent = 2, recompile = FALSE))
    if (inherits(cap2$val, "fit_error")) {
      att[[2]] <- list(iter = spec$refit_iter, error = cap2$val$msg)
    } else {
      f <- cap2$val
      dg <- diag_of(f)
      att[[2]] <- c(list(iter = spec$refit_iter, warmup = spec$refit_warmup, seed = spec$refit_seed), dg)
    }
  }
  list(fit = f, attempts = att, converged = isTRUE(dg$converged), warnings = unique(c(cap$warn)))
}

qsum <- function(v) list(estimate = mean(v), sd = stats::sd(v), ci95 = as.numeric(stats::quantile(v, c(.025, .975))),
                         p_gt_0 = mean(v > 0))
draws_of <- function(f, var) {
  m <- as.matrix(f)
  if (!var %in% colnames(m)) return(NULL)
  as.numeric(m[, var])
}
quantities <- function(f) {
  out <- list()
  vars <- c(mu_a_wb = "b_a_wb", mu_a_wb_x_grp = "b_a_wb:grp", beta_a_wb = "b_beta_a_wb",
            beta_a_wb_x_grp = "b_beta_a_wb:grp", sigma_grp = "b_sigma_grp", beta_intercept = "b_beta_Intercept",
            cor_mu_a_wb__beta_intercept = "cor_pid__a_wb__beta_Intercept")
  for (k in names(vars)) { v <- draws_of(f, vars[[k]]); if (!is.null(v)) out[[k]] <- c(list(variable = vars[[k]]), qsum(v)) }
  out
}

run_one <- function(dd) {
  r <- fit_rule(dd)
  if (!is.null(r$error)) return(r)
  r$quantities <- quantities(r$fit)
  r$nobs <- nrow(dd); r$n_children <- length(unique(as.character(dd$pid)))
  r
}

child_draws <- function(f, dd, kind) {
  m <- as.matrix(f)
  kids <- levels(factor(as.character(dd$pid)))
  g <- if (has_grp) tapply(dd$grp, as.character(dd$pid), function(x) x[1]) else NULL
  out <- matrix(NA_real_, nrow(m), length(kids), dimnames = list(NULL, kids))
  for (k in kids) {
    gi <- if (has_grp) g[[k]] else 0
    if (kind == "slope") {
      v <- m[, "b_a_wb"] + (if (has_grp) m[, "b_a_wb:grp"] * gi else 0)
      rn <- sprintf("r_pid[%s,a_wb]", k)
    } else {
      v <- m[, "b_beta_Intercept"] + (if (has_grp) m[, "b_beta_grp"] * gi else 0)
      rn <- sprintf("r_pid__beta[%s,Intercept]", k)
    }
    if (rn %in% colnames(m)) out[, k] <- v + m[, rn]
  }
  out
}

if (identical(spec$mode, "split")) {
  odd <- d[d$block_order %% 2 == 1, , drop = FALSE]
  even <- d[d$block_order %% 2 == 0, , drop = FALSE]
  ro <- run_one(odd); re <- run_one(even)
  if (!is.null(ro$error) || !is.null(re$error)) {
    res$error <- paste("split fit error:", ro$error, re$error)
  } else {
    S <- child_draws(ro$fit, odd, "slope"); Tt <- child_draws(re$fit, even, "tail")
    kids <- intersect(colnames(S)[colSums(is.finite(S)) > 0], colnames(Tt)[colSums(is.finite(Tt)) > 0])
    n <- min(nrow(S), nrow(Tt))
    r <- vapply(seq_len(n), function(s) stats::cor(S[s, kids], Tt[s, kids]), 0)
    res$split_correlation <- c(qsum(r), list(n_children = length(kids), n_draws = n,
                                             definition = paste("per draw, the correlation across children of the",
                                                                "odd-block mu a_wb slope and the even-block beta",
                                                                "intercept (draws paired by index)")))
    res$odd <- list(attempts = ro$attempts, converged = ro$converged, quantities = ro$quantities, nobs = ro$nobs)
    res$even <- list(attempts = re$attempts, converged = re$converged, quantities = re$quantities, nobs = re$nobs)
    res$quantities <- list(mu_a_wb_odd = ro$quantities$mu_a_wb, beta_a_wb_even = re$quantities$beta_a_wb)
    res$converged <- isTRUE(ro$converged) && isTRUE(re$converged)
    res$attempts <- list(odd = ro$attempts, even = re$attempts)
    res$nobs <- nrow(d)
  }
} else {
  r1 <- run_one(d)
  if (!is.null(r1$error)) res$error <- r1$error else {
    res$attempts <- r1$attempts; res$converged <- r1$converged; res$quantities <- r1$quantities
    res$nobs <- r1$nobs; res$n_children <- r1$n_children
    ps <- brms::prior_summary(r1$fit)
    res$priors <- lapply(seq_len(nrow(ps)), function(i) as.list(ps[i, c("prior", "class", "coef", "dpar")]))
    res$warnings <- as.list(r1$warnings)
  }
}
res$interpretable <- isTRUE(res$converged)
res$seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
write_result(res, spec$output_json)
if (!is.null(res$error)) quit(status = 2)
