# fit_glmm.R -- binomial mixed models of misses: AVSRT miss model M2.
#
# Usage: Rscript fit_glmm.R <spec.json>
#
# glmmTMB, binomial logit, Laplace ML, default nlminb; Wald z SEs from the joint Hessian (sdreport). Simplification
# path (engine_common.R) with the glmmTMB non-convergence rule: refit with optim BFGS and with nlminb started from the
# glmer estimates; consensus as for allFit (>= 2 fits, fixed effects within 1% of SE, log-likelihoods within 0.01).
# The LRT of step S4 is ML. Second implementation: glmer (bobyqa, Laplace, derivative-based Hessian; never nAGQ = 0
# or calc.derivs = FALSE) on the final structure; tolerance fixed effects within 5% of SE and log-likelihood within
# 0.05.
#
# Definitions for glmmTMB: non-convergence = optimiser code != 0, or a non-positive-definite Hessian
# that is not explained by a singular fit; singular = a random-effect covariance block whose Cholesky factor has a
# diagonal element < 1e-4 (as lme4::isSingular on the theta scale of a binomial GLMM) or is not positive definite;
# block intercept at the boundary = SD < 1e-4 (the logit scale has no residual SD).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_glmm.R <spec.json>")
.self <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(dirname(.self), "engine_common.R"))
suppressPackageStartupMessages({ library(glmmTMB); library(lme4) })

t_start <- Sys.time()
spec <- read_spec(args[1])
inp <- load_input(spec)
d <- inp$d
eng <- spec$engine

tmb_vc <- function(f) {
  vc <- glmmTMB::VarCorr(f)$cond
  lapply(vc, function(S) { S <- as.matrix(S); attr(S, "stddev") <- NULL; attr(S, "correlation") <- NULL; S })
}

tmb_singular <- function(f) {
  for (S in tmb_vc(f)) {
    ch <- tryCatch(chol(S), error = function(e) NULL)
    if (is.null(ch) || any(!is.finite(diag(ch))) || min(diag(ch)) < eng$singular_tol) return(TRUE)
  }
  FALSE
}

tmb_converged <- function(f, sing) {
  code <- f$fit$convergence
  pd <- isTRUE(f$sdr$pdHess)
  !(is.null(code) || code != 0 || (!pd && !sing))
}

tmb_ll <- function(f) as.numeric(stats::logLik(f))
tmb_fe <- function(f) glmmTMB::fixef(f)$cond
tmb_se <- function(f) sqrt(diag(as.matrix(stats::vcov(f)$cond)))

# glmmTMB theta (per term: log SDs, then correlation parameters) from a glmer fit; correlations started at 0
theta_from_glmer <- function(f_tmb, g) {
  vt <- tmb_vc(f_tmb)
  vg <- lme4::VarCorr(g)
  key <- function(grp, terms) paste(sub("\\.[0-9]+$", "", grp), paste(sort(terms), collapse = ","), sep = "|")
  gl <- list()
  for (nm in names(vg)) { S <- as.matrix(vg[[nm]]); gl[[key(nm, colnames(S))]] <- sqrt(diag(S)) }
  th <- numeric(0)
  for (nm in names(vt)) {
    S <- vt[[nm]]; k <- ncol(S)
    sdg <- gl[[key(nm, colnames(S))]]
    sdv <- if (is.null(sdg)) sqrt(diag(S)) else sdg[colnames(S)]
    sdv[!is.finite(sdv) | sdv <= 0] <- 1e-3
    th <- c(th, log(sdv), if (k > 1) rep(0, k * (k - 1) / 2) else numeric(0))
  }
  th
}

n_fits <- 0L
glmm_fitter <- function(st) {
  fm <- stats::as.formula(paste(spec$fixed_formula, "+", re_string(st)))
  n_fits <<- n_fits + 1L
  ctl <- glmmTMB::glmmTMBControl()
  cap <- .cap(eval(bquote(glmmTMB::glmmTMB(.(fm), family = binomial, data = d, control = ctl))))
  f <- cap$val
  if (inherits(f, "fit_error")) return(list(fit = NULL, conv = FALSE, sing = NA, allfit = FALSE, err = TRUE,
                                            info = f$msg, warnings = cap$warn))
  sing <- tmb_singular(f)
  conv <- tmb_converged(f, sing)
  used <- FALSE; af_info <- NULL
  if (!conv) {
    used <- TRUE
    fits <- list(nlminb = f)
    f2 <- .cap(eval(bquote(glmmTMB::glmmTMB(.(fm), family = binomial, data = d,
                                            control = glmmTMB::glmmTMBControl(optimizer = stats::optim,
                                                                              optArgs = list(method = "BFGS"))))))$val
    if (!inherits(f2, "fit_error")) fits$optim_BFGS <- f2
    g <- .cap(eval(bquote(lme4::glmer(.(fm), family = binomial, data = d,
                                      control = lme4::glmerControl(optimizer = "bobyqa")))))$val
    if (!inherits(g, "fit_error")) {
      stv <- list(beta = unname(lme4::fixef(g)), theta = theta_from_glmer(f, g))
      f3 <- .cap(eval(bquote(glmmTMB::glmmTMB(.(fm), family = binomial, data = d, start = .(stv)))))$val
      if (!inherits(f3, "fit_error")) fits$nlminb_from_glmer <- f3
    }
    ok <- Filter(function(z) isTRUE(z$fit$convergence == 0) && all(is.finite(tmb_fe(z))) &&
                   is.finite(tmb_ll(z)), fits)
    af_info <- list(n_returned = length(ok), optimizers = as.list(names(ok)))
    if (length(ok) >= 2) {
      fe <- t(vapply(ok, tmb_fe, tmb_fe(f)))
      se <- tmb_se(f); if (any(!is.finite(se))) se <- tmb_se(ok[[which.max(vapply(ok, tmb_ll, 0))]])
      ll <- vapply(ok, tmb_ll, 0)
      agree <- all(is.finite(se)) && isTRUE(consensus_ok(fe, se, ll, eng$consensus_fe_rel_se,
                                                         eng$consensus_loglik))
      af_info$consensus <- agree
      af_info$loglik_range <- diff(range(ll))
      best <- ok[[which.max(ll)]]
      if (agree) { conv <- TRUE; if (!isTRUE(f$sdr$pdHess)) f <- best } else { f <- best; af_info$kept <- names(ok)[which.max(ll)] }
      sing <- tmb_singular(f)
    } else af_info$consensus <- FALSE
  }
  list(fit = f, conv = conv, sing = sing, allfit = used, err = FALSE, info = "", warnings = cap$warn,
       messages = cap$msg, allfit_info = af_info)
}

block_sd_zero_glmm <- function(f) {
  if (is.null(f)) return(FALSE)
  vc <- tmb_vc(f)
  b <- vc[["pid_run_block"]]
  !is.null(b) && sqrt(b[1, 1]) < eng$block_boundary_rel
}

lrt_p_glmm <- function(red, cur) {
  a <- .cap(stats::anova(red, cur))$val                    # ML LRT
  if (inherits(a, "fit_error")) NA_real_ else a[["Pr(>Chisq)"]][2]
}

start <- state_from_spec(spec$random$start)
if (isTRUE(spec$reduction_path)) {
  rp <- reduce_path(start, unlist(spec$random$droppable), glmm_fitter, block_sd_zero_glmm, lrt_p_glmm, eng$lrt_drop_p)
} else {
  st <- if (is.null(spec$fixed_state)) start else state_from_spec(spec$fixed_state)
  rp <- fit_fixed_state(st, glmm_fitter)
}
res <- list(schema = ENGINE_SCHEMA, label = spec$label, model = spec$model, task = spec$task, cohort = spec$cohort,
            family = "glmm", outcome_kind = spec$outcome_kind, ns_check = inp$ns_check,
            reduction_log = rp$log, flag = rp$flag, final_state = state_to_list(rp$state), n_fits = n_fits)
if (rp$flag == "error" || is.null(rp$cur$fit)) {
  res$error <- rp$cur$info
  write_result(res, spec$output_json)
  quit(status = 2)
}
f <- rp$cur$fit
cf <- summary(f)$coefficients$cond
cn <- rownames(cf)
fixed <- lapply(cn, function(k) c(list(term = k), infer_row(cf[k, "Estimate"], cf[k, "Std. Error"], Inf)))
L <- contrast_matrix(spec$targets, cn)
V <- as.matrix(stats::vcov(f)$cond)
beta <- tmb_fe(f)
targets <- list()
for (k in rownames(L)) {
  est <- sum(L[k, ] * beta); se <- sqrt(as.numeric(t(L[k, ]) %*% V %*% L[k, ]))
  targets[[k]] <- c(infer_row(est, se, Inf), list(df_method = "Wald z", weights = as.list(L[k, L[k, ] != 0])))
}
vcl <- tmb_vc(f)
res$fixed <- fixed
res$coef_names <- as.list(cn)
res$targets <- targets
res$random_blocks <- varcorr_blocks(vcl)
res$loglik <- tmb_ll(f)
res$loglik_df <- attr(stats::logLik(f), "df")
res$reml <- FALSE
res$nobs <- stats::nobs(f)
res$singular <- tmb_singular(f)
res$a_wb_slope_sd <- {
  s <- NA_real_
  for (S in vcl) if ("a_wb" %in% colnames(S)) s <- sqrt(S["a_wb", "a_wb"])
  s
}
res$convergence <- list(converged = rp$cur$conv, allfit_used = rp$cur$allfit, allfit = rp$cur$allfit_info,
                        optimizer_code = f$fit$convergence, optimizer_message = f$fit$message,
                        pdHess = isTRUE(f$sdr$pdHess), warnings = as.list(rp$cur$warnings),
                        messages = as.list(rp$cur$messages))
# second implementation: glmer bobyqa on the same final structure
if (isTRUE(spec$second_impl)) {
  t0 <- Sys.time()
  fm <- stats::as.formula(paste(spec$fixed_formula, "+", re_string(rp$state)))
  gc <- .cap(eval(bquote(lme4::glmer(.(fm), family = binomial, data = d,
                                     control = lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))))))
  if (inherits(gc$val, "fit_error")) res$second_impl <- list(error = gc$val$msg) else {
    g <- gc$val
    bg <- lme4::fixef(g); seg <- sqrt(diag(as.matrix(stats::vcov(g))))
    se_t <- sqrt(diag(V))
    common <- intersect(names(beta), names(bg))
    rel <- abs(beta[common] - bg[common]) / se_t[common]
    lld <- as.numeric(stats::logLik(g)) - tmb_ll(f)
    res$second_impl <- list(
      engine = "glmer bobyqa Laplace", seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      max_fe_diff_rel_se = max(rel), loglik_glmer = as.numeric(stats::logLik(g)), loglik_diff = lld,
      se_ratio_glmer_over_tmb = as.list(seg[common] / se_t[common]),
      fe_diff_rel_se = as.list(rel), singular = lme4::isSingular(g),
      warnings = as.list(gc$warn),
      tolerance = list(fe_rel_se = eng$glmm_second_impl_fe_rel_se, loglik = eng$glmm_second_impl_loglik),
      pass = (max(rel) <= eng$glmm_second_impl_fe_rel_se) && (abs(lld) <= eng$glmm_second_impl_loglik),
      targets = lapply(setNames(rownames(L), rownames(L)), function(k) {
        Lk <- L[k, names(bg)]; est <- sum(Lk * bg)
        se <- sqrt(as.numeric(t(Lk) %*% as.matrix(stats::vcov(g)) %*% Lk)); infer_row(est, se, Inf) }))
  }
}
res$fixed_formula <- spec$fixed_formula
res$full_formula <- paste(spec$fixed_formula, "+", re_string(rp$state))
res$seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
write_result(res, spec$output_json)
