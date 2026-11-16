# fit_lmm.R -- linear mixed models of log RT: M1, M1-H2, M1-hist, M3, M4, M5h (and their group forms).
#
# Usage: Rscript fit_lmm.R <spec.json>
#
# lme4::lmer, REML, lmerControl(optimizer = "bobyqa"); the simplification path (engine_common.R) with allFit over
# five optimisers on non-convergence (consensus: >= 2 fits, every fixed effect within 1% of its SE, log-likelihoods
# within 0.01); the final-fit optimum check; the accepted fit is converted to lmerTest (Satterthwaite t) and every
# target coefficient / contrast is reported with its Satterthwaite df, 95% and 90% CIs, and the CR2 cluster-robust SE
# with Satterthwaite df (clubSandwich vcovCR type "CR2" on the lmer fit, cluster = child).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_lmm.R <spec.json>")
.self <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(dirname(.self), "engine_common.R"))
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(clubSandwich) })

t_start <- Sys.time()
spec <- read_spec(args[1])
inp <- load_input(spec)
d <- inp$d
eng <- spec$engine
METH <- do.call(rbind, lapply(eng$lmm$allfit, function(o) c(optimizer = o[[1]], method = o[[2]])))

# The control is embedded in the fitted call as an unevaluated lmerControl() call: allFit() rebuilds the control
# from getCall(fit)$control and fails ("object of type 'symbol' is not subsettable") when the call holds a symbol.
# The optimiser is bobyqa with the lme4 defaults.
ctrl_call <- function() {
  quote(lme4::lmerControl(optimizer = "bobyqa"))
}

nonconv_lmm <- function(f, warn) {
  msgs <- c(warn, f@optinfo$conv$lme4$messages)
  any(grepl("converge|Hessian|eigenvalue", msgs, ignore.case = TRUE) & !grepl("singular", msgs, ignore.case = TRUE))
}

n_fits <- 0L
lmm_fitter <- function(st) {
  fm <- stats::as.formula(paste(spec$fixed_formula, "+", re_string(st)))
  ctl <- ctrl_call()
  n_fits <<- n_fits + 1L
  cap <- .cap(eval(bquote(lme4::lmer(.(fm), data = d, REML = TRUE, control = .(ctl)))))
  f <- cap$val
  if (inherits(f, "fit_error")) return(list(fit = NULL, conv = FALSE, sing = NA, allfit = FALSE, err = TRUE,
                                            info = f$msg, warnings = cap$warn))
  nc <- nonconv_lmm(f, cap$warn)
  used <- FALSE; af_info <- NULL
  if (nc) {
    used <- TRUE
    af <- .cap(lme4::allFit(f, meth.tab = METH, data = d, verbose = FALSE))$val
    ok <- if (inherits(af, "fit_error")) list() else
      Filter(function(z) inherits(z, "merMod") && is.finite(as.numeric(stats::logLik(z))), unclass(af))
    af_info <- list(n_returned = length(ok), optimizers = as.list(names(ok)))
    if (length(ok) >= 2) {
      fe <- t(vapply(ok, lme4::fixef, lme4::fixef(f)))
      se <- sqrt(diag(as.matrix(stats::vcov(f))))
      ll <- vapply(ok, function(z) as.numeric(stats::logLik(z)), 0)
      agree <- isTRUE(consensus_ok(fe, se, ll, eng$consensus_fe_rel_se, eng$consensus_loglik))
      af_info$max_fe_range_rel_se <- max(apply(fe, 2, function(v) diff(range(v))) / se)
      af_info$loglik_range <- diff(range(ll))
      af_info$consensus <- agree
      if (agree) nc <- FALSE else { f <- ok[[which.max(ll)]]; af_info$kept <- names(ok)[which.max(ll)] }
    } else af_info$consensus <- FALSE
  }
  list(fit = f, conv = !nc, sing = lme4::isSingular(f, tol = eng$singular_tol), allfit = used, err = FALSE,
       info = "", warnings = cap$warn, messages = cap$msg, allfit_info = af_info)
}

# Final-fit optimum check: allFit on the selected fit; a refit that reports convergence (no lme4 convergence message
# other than a singular-fit message, no optimiser warning matching the non-convergence rule, and optimiser code 0)
# with finite fixed effects and a log-likelihood more than `tol` above the selected fit replaces it (the highest such
# refit; engine_common.R optimum_check_select). Applies to every LMM, path or fixed structure.
optimum_check_lmm <- function(f) {
  on <- eng$lmm$optimum_check
  if (!is.null(on) && !isTRUE(on)) return(list(fit = f, info = list(ran = FALSE)))
  tol <- if (is.null(eng$lmm$optimum_check_tol)) 0.01 else as.numeric(eng$lmm$optimum_check_tol)
  ll0 <- as.numeric(stats::logLik(f))
  info <- list(ran = TRUE, tol = tol, selected_optimizer = f@optinfo$optimizer,
               selected_loglik = ll0, refits = list(), switched = FALSE)
  t0 <- Sys.time()
  cap <- .cap(lme4::allFit(f, meth.tab = METH, data = d, verbose = FALSE))
  af <- cap$val
  if (inherits(af, "fit_error")) {
    info$error <- af$msg
    return(list(fit = f, info = info))
  }
  fits <- list(); refits <- list()
  for (nm in names(af)) {
    z <- af[[nm]]
    if (!inherits(z, "merMod")) {
      info$refits[[nm]] <- list(name = nm, ok = FALSE, error = if (inherits(z, "error")) conditionMessage(z) else "no fit")
      next
    }
    ll <- as.numeric(stats::logLik(z))
    code <- z@optinfo$conv$opt
    conv <- !nonconv_lmm(z, as.character(unlist(z@optinfo$warnings))) &&
      (is.null(code) || (is.numeric(code) && all(code == 0)))
    fe_ok <- all(is.finite(lme4::fixef(z)))
    r <- list(name = nm, ok = TRUE, loglik = ll, gain = ll - ll0, converged = conv, fixef_finite = fe_ok)
    info$refits[[nm]] <- r; refits[[length(refits) + 1]] <- r; fits[[nm]] <- z
  }
  sel <- optimum_check_select(ll0, refits, tol)
  info$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  info$n_refits <- length(refits)
  if (isTRUE(sel$switch)) {
    info$switched <- TRUE; info$kept <- sel$kept; info$loglik_gain <- sel$gain
    return(list(fit = fits[[sel$kept]], info = info))
  }
  list(fit = f, info = info)
}

block_sd_zero_lmm <- function(f) {
  if (is.null(f)) return(FALSE)
  vc <- as.data.frame(lme4::VarCorr(f))
  sb <- vc$sdcor[vc$grp == "pid_run_block" & vc$var1 == "(Intercept)"]
  length(sb) == 1 && sb < eng$block_boundary_rel * stats::sigma(f)
}

lrt_p_lmm <- function(red, cur) {
  a <- .cap(stats::anova(red, cur, refit = FALSE))$val       # REML LRT (step S4)
  if (inherits(a, "fit_error")) NA_real_ else a[["Pr(>Chisq)"]][2]
}

start <- state_from_spec(spec$random$start)
if (isTRUE(spec$reduction_path)) {
  rp <- reduce_path(start, unlist(spec$random$droppable), lmm_fitter, block_sd_zero_lmm, lrt_p_lmm, eng$lrt_drop_p)
} else {
  st <- if (is.null(spec$fixed_state)) start else state_from_spec(spec$fixed_state)
  rp <- fit_fixed_state(st, lmm_fitter)
}
res <- list(schema = ENGINE_SCHEMA, label = spec$label, model = spec$model, task = spec$task, cohort = spec$cohort,
            family = "lmm", outcome_kind = spec$outcome_kind, ns_check = inp$ns_check,
            reduction_log = rp$log, flag = rp$flag, final_state = state_to_list(rp$state), n_fits = n_fits)
if (rp$flag == "error" || is.null(rp$cur$fit)) {
  res$error <- rp$cur$info
  write_result(res, spec$output_json)
  quit(status = 2)
}
f0 <- rp$cur$fit
oc <- optimum_check_lmm(f0)
f0 <- oc$fit
res$optimum_check <- oc$info
ft <- .cap(lmerTest::as_lmerModLmerTest(f0))
f <- if (inherits(ft$val, "fit_error")) f0 else ft$val
sm <- summary(f, ddf = "Satterthwaite")$coefficients
cn <- rownames(sm)
fixed <- lapply(cn, function(k) c(list(term = k), infer_row(sm[k, "Estimate"], sm[k, "Std. Error"],
                                                           if ("df" %in% colnames(sm)) sm[k, "df"] else Inf)))
L <- contrast_matrix(spec$targets, cn)
targets <- list()
for (k in rownames(L)) {
  ct <- .cap(lmerTest::contest1D(f, L[k, ], ddf = "Satterthwaite"))$val
  if (inherits(ct, "fit_error")) {
    est <- sum(L[k, ] * lme4::fixef(f)); se <- sqrt(as.numeric(t(L[k, ]) %*% as.matrix(stats::vcov(f)) %*% L[k, ]))
    targets[[k]] <- c(infer_row(est, se, Inf), list(df_method = "z (contest1D failed)"))
  } else targets[[k]] <- c(infer_row(ct$Estimate, ct$`Std. Error`, ct$df), list(df_method = "Satterthwaite"))
  targets[[k]]$weights <- as.list(L[k, L[k, ] != 0])
}
cr2 <- NULL
if (isTRUE(spec$cr2)) {
  t0 <- Sys.time()
  V <- .cap(clubSandwich::vcovCR(f, cluster = d[[spec$cluster]], type = "CR2"))
  if (inherits(V$val, "fit_error")) cr2 <- list(error = V$val$msg) else {
    lc <- .cap(clubSandwich::linear_contrast(f, vcov = V$val, contrasts = L, test = "Satterthwaite", level = 0.95))
    cr2 <- list(seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), targets = list())
    if (inherits(lc$val, "fit_error")) cr2$error <- lc$val$msg else {
      lcv <- as.data.frame(lc$val)
      for (i in seq_len(nrow(L))) {
        k <- rownames(L)[i]
        cr2$targets[[k]] <- c(infer_row(lcv$Est[i], lcv$SE[i], lcv$df[i]), list(df_method = "clubSandwich Satterthwaite"))
      }
    }
    if (isTRUE(spec$export_cr2_vcov)) cr2$vcov <- lapply(seq_len(nrow(V$val)), function(r) as.numeric(V$val[r, ]))
  }
}
vc <- lme4::VarCorr(f)
vca <- as.data.frame(vc)
re_tab <- lapply(seq_len(nrow(vca)), function(i) list(grp = vca$grp[i], var1 = vca$var1[i], var2 = vca$var2[i],
                                                     vcov = vca$vcov[i], sdcor = vca$sdcor[i]))
a_wb_sd <- vca$sdcor[vca$grp != "Residual" & vca$var1 %in% c("a_wb") & is.na(vca$var2)]
res$fixed <- fixed
res$coef_names <- as.list(cn)
res$targets <- targets
res$cr2 <- cr2
res$random <- re_tab
res$random_blocks <- varcorr_blocks(unclass(vc)[names(vc)])
res$sigma <- stats::sigma(f)
res$loglik <- as.numeric(stats::logLik(f))
res$loglik_df <- attr(stats::logLik(f), "df")
res$reml <- lme4::isREML(f)
res$nobs <- stats::nobs(f)
res$ngroups <- as.list(lme4::ngrps(f))
res$singular <- lme4::isSingular(f, tol = eng$singular_tol)
res$a_wb_slope_sd <- if (length(a_wb_sd)) a_wb_sd[1] else NA_real_
res$a_wb_slope_variance_zero <- length(a_wb_sd) > 0 && a_wb_sd[1] < eng$singular_tol * stats::sigma(f)
res$theta <- as.numeric(lme4::getME(f, "theta"))
res$convergence <- list(converged = rp$cur$conv, allfit_used = rp$cur$allfit, allfit = rp$cur$allfit_info,
                        optimizer = f@optinfo$optimizer, lme4_messages = as.list(f@optinfo$conv$lme4$messages),
                        warnings = as.list(rp$cur$warnings), messages = as.list(rp$cur$messages),
                        lmertest_conversion = if (inherits(ft$val, "fit_error")) ft$val$msg else "ok")
if (isTRUE(spec$export_fixef_vcov)) res$vcov <- lapply(seq_len(length(cn)), function(r) as.numeric(as.matrix(stats::vcov(f))[r, ]))
res$fixed_formula <- spec$fixed_formula
res$full_formula <- paste(spec$fixed_formula, "+", re_string(rp$state))
res$seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
write_result(res, spec$output_json)
