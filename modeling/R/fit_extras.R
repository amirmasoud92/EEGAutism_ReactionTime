# fit_extras.R -- inference checks and sensitivity analyses that need R (modeling/extras_bridge.py). Same hand-over
# as the model engine: a JSON spec + the model-input TSV (spline bases rebuilt with splines::ns and compared,
# engine_common.R::load_input).
#
# Usage: Rscript fit_extras.R <spec.json>      spec$extra selects:
#   A1       nlme::lme(<fixed part>, random = <spec$random, default ~ 1 + a_wb | pid>,
#            correlation = corAR1(form = ~ <t_seq> | pid/run), control = lmeControl(opt = "optim")), REML.
#            Targets: contrasts of the fixed effects, t with the nlme (containment) df of the target's leading
#            coefficient. A1 omits M1's block intercept and child time slopes.
#   A3L      logistic regression of miss on the per-child detrended a_wb (column xr) with child fixed intercepts,
#            the M2 fixed covariates without alpha terms and child-specific linear t_run / tib slopes, children with
#            >= 1 miss; aliased columns dropped by a pivoted QR; CR2 (clubSandwich vcovCR type "CR2",
#            cluster = child) with Satterthwaite df (coef_test). Also the nuisance model without xr and its fitted
#            p0 (for A2-L), written to <output>.p0.tsv.
#   CR2CHECK clubSandwich CR2 of lm(yr ~ 0 + <x columns>) clustered by child (cross-check of the Python closed form).
#   CENS     S-CENS: brms gaussian with lrt | cens(cens) (right-censored at the response-window end), the fixed part
#            and spec$random; chains / iter / seed from the spec; flat priors on b (brms default), brms default
#            student-t priors on sd / Intercept / sigma, LKJ(1) on correlations; convergence R-hat < 1.01 and
#            bulk ESS > 400 (as for M6). Optional survreg cross-check (spec$survreg_check): gaussian AFT on lrt
#            with child fixed intercepts (no random slope).
#   HET      S-HET: glmmTMB gaussian with dispformula = ~ grp and the random string of the spec
#            (group-specific a_wb slopes in place of the pooled slope); Wald z for the targets.
#   E3C      E3c: the final M1 (spec$full_formula = the fixed part + the final random structure of the main
#            fit) is refitted once with lmer (REML,
#            bobyqa; no reduction path) and lme4::bootMer(type = "parametric", use.u = FALSE, nsim = B,
#            seed = spec$seed) simulates B outcomes from it and refits the same structure (lme4 refit, no reduction
#            path). Statistic per fit: beta = the target contrast of the fixed effects (P1 = a_wb), tau2 = the
#            variance of the a_wb random slope (0 if the structure has none), e3c = beta^2 + tau2, and
#            e3c / spec$denominator (the denominator is fixed, computed by the caller). Percentile 95% CIs.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_extras.R <spec.json>")
.self <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(dirname(.self), "engine_common.R"))

t_start <- Sys.time()
spec <- read_spec(args[1])
inp <- load_input(spec)
d <- inp$d
res <- list(schema = "alpha_state.extras_result/1", extra = spec$extra, label = spec$label, model = spec$model,
            ns_check = inp$ns_check)

target_rows <- function(beta, V, targets, df_fun) {
  out <- list()
  for (k in names(targets)) {
    w <- targets[[k]]
    L <- setNames(rep(0, length(beta)), names(beta))
    for (nm in names(w)) {
      if (!nm %in% names(beta)) stop(sprintf("target %s: coefficient %s not in the model", k, nm))
      L[nm] <- as.numeric(w[[nm]])
    }
    est <- sum(L * beta); se <- sqrt(as.numeric(t(L) %*% V %*% L))
    df <- df_fun(names(w)[1])
    out[[k]] <- c(infer_row(est, se, df), list(weights = as.list(L[L != 0])))
  }
  out
}

run_a1 <- function() {
  suppressPackageStartupMessages(library(nlme))
  fx <- stats::as.formula(spec$fixed_formula)
  rnd <- stats::as.formula(if (is.null(spec$random_formula)) "~ 1 + a_wb | pid" else spec$random_formula)
  tseq <- if (is.null(spec$t_seq)) "trial_index_in_run" else spec$t_seq
  grp <- if (is.null(spec$ar_group)) "pid/run" else spec$ar_group
  d$run <- factor(d$run)
  key <- interaction(d$pid, d$run, drop = TRUE)
  if (anyDuplicated(data.frame(key, d[[tseq]]))) stop("t_seq is not unique within pid x run")
  cor <- nlme::corAR1(form = stats::as.formula(sprintf("~ %s | %s", tseq, grp)))
  fit_opt <- function(opt) .cap(nlme::lme(fixed = fx, random = rnd, correlation = cor, data = d, method = "REML",
                                          control = nlme::lmeControl(opt = opt, msMaxIter = 200,
                                                                     returnObject = TRUE)))
  cap <- fit_opt("optim")                       # lmeControl(opt = "optim")
  res$optimizer_used <<- "optim"
  if (inherits(cap$val, "fit_error")) {         # on an optim failure, nlme's default nlminb, logged
    res$optim_error <<- cap$val$msg
    cap <- fit_opt("nlminb")
    res$optimizer_used <<- "nlminb (fallback after optim failed)"
  }
  f <- cap$val
  if (inherits(f, "fit_error")) { res$error <<- f$msg; return(invisible()) }
  tt <- summary(f)$tTable
  beta <- nlme::fixef(f); V <- as.matrix(stats::vcov(f))
  dfs <- setNames(tt[, "DF"], rownames(tt))
  res$targets <<- target_rows(beta, V, spec$targets, function(nm) as.numeric(dfs[[nm]]))
  phi <- as.numeric(coef(f$modelStruct$corStruct, unconstrained = FALSE))
  res$phi <<- phi
  res$sigma <<- f$sigma
  res$loglik <<- as.numeric(stats::logLik(f))
  res$random <<- lapply(seq_len(nrow(nlme::VarCorr(f))), function(i) as.list(nlme::VarCorr(f)[i, ]))
  res$warnings <<- as.list(cap$warn)
  res$nobs <<- nrow(d)
  res$df_method <<- "nlme containment df of the target's leading coefficient"
  res$note <<- "A1 omits the block intercept and the child time slopes of the main model"
}

drop_aliased <- function(X) {
  q <- qr(X)
  keep <- sort(q$pivot[seq_len(q$rank)])
  list(X = X[, keep, drop = FALSE], dropped = colnames(X)[setdiff(seq_len(ncol(X)), keep)])
}

run_a3l <- function() {
  suppressPackageStartupMessages(library(clubSandwich))
  keep_kids <- names(which(tapply(d$miss, d$pid, sum) > 0))
  dd <- d[d$pid %in% keep_kids, , drop = FALSE]
  dd$fc <- factor(as.character(dd$pid))
  nuis <- unlist(spec$nuisance_terms)
  rhs0 <- paste(c("0 + fc", "fc:t_run", "fc:tib", nuis), collapse = " + ")
  X0 <- stats::model.matrix(stats::as.formula(paste("~", rhs0)), dd)
  a0 <- drop_aliased(X0)
  X1 <- cbind(xr = dd$xr, a0$X)
  a1 <- drop_aliased(X1)
  if (!"xr" %in% colnames(a1$X)) stop("xr is aliased with the nuisance design")
  y <- dd$miss
  g0 <- .cap(stats::glm(y ~ 0 + a0$X, family = stats::binomial()))
  g1 <- .cap(stats::glm(y ~ 0 + a1$X, family = stats::binomial()))
  if (inherits(g1$val, "fit_error")) { res$error <<- g1$val$msg; return(invisible()) }
  t0 <- Sys.time()
  V <- clubSandwich::vcovCR(g1$val, cluster = dd$pid, type = "CR2")
  ct <- clubSandwich::coef_test(g1$val, vcov = V, test = "Satterthwaite", coefs = 1)
  b <- ct$beta[1]; se <- ct$SE[1]; dfv <- ct$df_Satt[1]
  res$targets <<- list(S1 = c(infer_row(b, se, dfv), list(df_method = "clubSandwich CR2 Satterthwaite")))
  res$cr2_seconds <<- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res$n_children_kept <<- length(keep_kids)
  res$n_children_dropped_no_miss <<- length(unique(d$pid)) - length(keep_kids)
  res$n_rows <<- nrow(dd)
  res$aliased_dropped <<- as.list(unique(c(a0$dropped, a1$dropped)))
  res$glm_converged <<- isTRUE(g1$val$converged)
  res$warnings <<- as.list(unique(c(g0$warn, g1$warn)))
  if (!inherits(g0$val, "fit_error")) {
    p0 <- rep(NA_real_, nrow(d))
    p0[d$pid %in% keep_kids] <- stats::fitted(g0$val)
    data.table::fwrite(data.frame(p0 = p0), paste0(spec$output_json, ".p0.tsv"), sep = "\t", na = "NA")
    res$p0_file <<- paste0(spec$output_json, ".p0.tsv")
  }
}

run_cr2check <- function() {
  suppressPackageStartupMessages(library(clubSandwich))
  xs <- unlist(spec$x_columns)
  fm <- stats::as.formula(paste("yr ~ 0 +", paste(xs, collapse = " + ")))
  f <- stats::lm(fm, data = d)
  V <- clubSandwich::vcovCR(f, cluster = d$pid, type = "CR2")
  w <- unlist(spec$contrast_weights)
  if (is.null(w)) w <- c(1, rep(0, length(xs) - 1))
  L <- matrix(w, nrow = 1)
  res$estimate <<- as.numeric(L %*% stats::coef(f))
  res$se <<- sqrt(as.numeric(L %*% as.matrix(V) %*% t(L)))
  res$coef <<- as.list(stats::coef(f))
}

run_cens <- function() {
  suppressPackageStartupMessages({ library(brms); library(rstan) })
  rstan::rstan_options(auto_write = TRUE)
  fs <- paste(spec$fixed_formula_rhs, collapse = "")
  fm <- brms::bf(stats::as.formula(sprintf("lrt | cens(cens) ~ %s + %s", fs, spec$random_string)))
  chains <- if (is.null(spec$chains)) 4L else as.integer(spec$chains)
  iter <- if (is.null(spec$iter)) 4000L else as.integer(spec$iter)              # 4 x 4,000, warmup 2,000
  warmup <- if (is.null(spec$warmup)) iter %/% 2L else as.integer(spec$warmup)
  cap <- .cap(brms::brm(fm, data = d, family = stats::gaussian(), chains = chains, iter = iter, warmup = warmup,
                        cores = min(chains, if (is.null(spec$cores)) chains else as.integer(spec$cores)),
                        seed = as.integer(spec$seed), refresh = 0, silent = 2, backend = "rstan"))
  f <- cap$val
  if (inherits(f, "fit_error")) { res$error <<- f$msg; return(invisible()) }
  dr <- as.matrix(f, variable = paste0("b_", names(brms::fixef(f)[, 1])))
  colnames(dr) <- sub("^b_", "", colnames(dr))
  targets <- list()
  for (k in names(spec$targets)) {
    w <- spec$targets[[k]]
    v <- rep(0, nrow(dr))
    for (nm in names(w)) v <- v + as.numeric(w[[nm]]) * dr[, nm]
    targets[[k]] <- list(estimate = mean(v), median = stats::median(v), se = stats::sd(v), df = NULL,
                         ci95 = as.numeric(stats::quantile(v, c(0.025, 0.975))),
                         ci90 = as.numeric(stats::quantile(v, c(0.05, 0.95))),
                         p_two_sided_posterior = 2 * min(mean(v > 0), mean(v < 0)),
                         df_method = "posterior (brms); CrI = equal-tailed quantiles")
  }
  sm <- summary(f)
  rh <- c(sm$fixed$Rhat, unlist(lapply(sm$random, function(z) z$Rhat)), sm$spec_pars$Rhat)
  es <- c(sm$fixed$Bulk_ESS, unlist(lapply(sm$random, function(z) z$Bulk_ESS)), sm$spec_pars$Bulk_ESS)
  np <- brms::nuts_params(f)
  res$targets <<- targets
  res$max_rhat <<- max(rh, na.rm = TRUE)
  res$min_bulk_ess <<- min(es, na.rm = TRUE)
  res$n_divergent <<- sum(np$Value[np$Parameter == "divergent__"])
  res$converged <<- (res$max_rhat < 1.01) && (res$min_bulk_ess > 400)
  res$n_censored <<- sum(d$cens == 1)
  res$nobs <<- nrow(d)
  res$chains <<- chains; res$iter <<- iter; res$warmup <<- warmup; res$seed <<- spec$seed
  res$priors <<- lapply(seq_len(nrow(brms::prior_summary(f))), function(i) as.list(brms::prior_summary(f)[i, c("prior", "class", "coef", "group")]))
  res$warnings <<- as.list(unique(cap$warn))
  if (isTRUE(spec$survreg_check)) {
    suppressPackageStartupMessages(library(survival))
    d$fc <- factor(d$pid)
    sf <- stats::as.formula(sprintf("survival::Surv(lrt, 1 - cens) ~ 0 + fc + %s", fs))
    s <- .cap(survival::survreg(sf, data = d, dist = "gaussian"))
    if (!inherits(s$val, "fit_error")) {
      cf <- stats::coef(s$val); Vs <- stats::vcov(s$val)
      res$survreg <<- lapply(names(spec$targets), function(k) {
        w <- unlist(spec$targets[[k]]); nm <- names(w)
        list(target = k, estimate = sum(w * cf[nm]),
             se = sqrt(as.numeric(t(w) %*% as.matrix(Vs[nm, nm, drop = FALSE]) %*% w)),
             note = "gaussian AFT on lrt, child fixed intercepts, no random slope (cross-check)")
      })
    } else res$survreg <<- list(error = s$val$msg)
  }
}

run_het <- function() {
  suppressPackageStartupMessages(library(glmmTMB))
  fm <- stats::as.formula(paste(spec$fixed_formula, "+", spec$random_string))
  cap <- .cap(glmmTMB::glmmTMB(fm, dispformula = ~ grp, data = d, family = stats::gaussian()))
  f <- cap$val
  if (inherits(f, "fit_error")) { res$error <<- f$msg; return(invisible()) }
  beta <- glmmTMB::fixef(f)$cond; V <- as.matrix(stats::vcov(f)$cond)
  res$targets <<- target_rows(beta, V, spec$targets, function(nm) Inf)
  res$dispersion <<- as.list(glmmTMB::fixef(f)$disp)
  res$random_blocks <<- varcorr_blocks(lapply(glmmTMB::VarCorr(f)$cond, function(S) {
    S <- as.matrix(S); attr(S, "stddev") <- NULL; attr(S, "correlation") <- NULL; S }))
  res$pdHess <<- isTRUE(f$sdr$pdHess)
  res$optimizer_code <<- f$fit$convergence
  res$loglik <<- as.numeric(stats::logLik(f))
  res$warnings <<- as.list(cap$warn)
}

run_e3c <- function() {
  suppressPackageStartupMessages(library(lme4))
  fm <- stats::as.formula(spec$full_formula)
  cap <- .cap(lme4::lmer(fm, data = d, REML = TRUE, control = lme4::lmerControl(optimizer = "bobyqa")))
  f <- cap$val
  if (inherits(f, "fit_error")) { res$error <<- f$msg; return(invisible()) }
  w <- unlist(spec$target_weights)
  slope <- if (is.null(spec$slope_term)) "a_wb" else spec$slope_term
  denom <- as.numeric(spec$denominator)
  stat <- function(m) {
    b <- lme4::fixef(m)
    if (!all(names(w) %in% names(b))) stop("target coefficient not in the model")
    est <- sum(w * b[names(w)])
    vc <- as.data.frame(lme4::VarCorr(m))
    t2 <- vc$vcov[vc$grp != "Residual" & vc$var1 == slope & is.na(vc$var2)]
    t2 <- if (length(t2)) t2[1] else 0
    c(beta = est, tau2 = t2, e3c = est^2 + t2, fraction = (est^2 + t2) / denom)
  }
  pt <- stat(f)
  B <- as.integer(spec$B)
  ncpus <- if (is.null(spec$ncpus)) 1L else as.integer(spec$ncpus)
  t0 <- Sys.time()
  bm <- .cap(lme4::bootMer(f, stat, nsim = B, seed = as.integer(spec$seed), type = "parametric", use.u = FALSE,
                           parallel = if (ncpus > 1) "snow" else "no", ncpus = ncpus))
  if (inherits(bm$val, "fit_error")) { res$error <<- bm$val$msg; return(invisible()) }
  tt <- bm$val$t
  ok <- stats::complete.cases(tt)
  q <- function(v) as.numeric(stats::quantile(v, c(0.025, 0.975), names = FALSE))
  res$point <<- as.list(pt)
  res$ci95_percentile <<- list(beta = q(tt[ok, "beta"]), tau2 = q(tt[ok, "tau2"]), e3c = q(tt[ok, "e3c"]),
                               fraction = q(tt[ok, "fraction"]))
  res$bootstrap_sd <<- as.list(apply(tt[ok, , drop = FALSE], 2, stats::sd))
  res$B <<- B
  res$B_ok <<- sum(ok)
  res$seed <<- spec$seed
  res$denominator <<- denom
  res$boot_seconds <<- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res$singular_refit <<- lme4::isSingular(f)
  res$loglik_refit <<- as.numeric(stats::logLik(f))
  res$full_formula <<- spec$full_formula
  res$warnings <<- as.list(unique(c(cap$warn, bm$warn)))
  dr <- lapply(seq_len(ncol(tt)), function(j) as.numeric(tt[, j]))
  names(dr) <- colnames(tt)
  res$draws <<- dr
  res$note <<- paste("parametric bootstrap (lme4::bootMer, use.u = FALSE) from the refitted final M1;",
                     "refits keep the same random structure (no simplification path); denominator fixed")
}

switch(spec$extra, A1 = run_a1(), A3L = run_a3l(), CR2CHECK = run_cr2check(), CENS = run_cens(), HET = run_het(),
       E3C = run_e3c(), stop("unknown extra ", spec$extra))
res$seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
write_result(res, spec$output_json)
if (!is.null(res$error)) quit(status = 2)
