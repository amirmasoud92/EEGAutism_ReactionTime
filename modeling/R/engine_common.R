# engine_common.R -- shared code of the R model engine (modeling/r_bridge.py).
#
# Reads a JSON model spec (modeling.model_specs.build_spec) and its model-input TSV
# (modeling.model_data.write_model_input), rebuilds every spline basis with splines::ns(x, knots, Boundary.knots) from
# the knot table (the R basis is the one used in the fit) and compares it with the Python basis in the TSV
# (max |difference| <= ns_check_tol, else the fit is refused), and provides the random-structure string, warning
# capture and the random-effects simplification path applied on non-convergence (reduce_path()).
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(splines)
})

ENGINE_SCHEMA <- "alpha_state.model_result/1"

read_spec <- function(path) jsonlite::fromJSON(path, simplifyVector = FALSE)

load_input <- function(spec) {
  d <- data.table::fread(spec$input_tsv, sep = "\t", na.strings = "NA", data.table = FALSE, showProgress = FALSE)
  for (f in unlist(spec$factors)) if (f %in% names(d)) d[[f]] <- factor(as.character(d[[f]]))
  ns_check <- list()
  for (s in spec$splines) {
    x <- d[[s$variable]]
    B <- splines::ns(x, knots = unlist(s$interior_knots), Boundary.knots = unlist(s$boundary_knots))
    cols <- unlist(s$columns)
    P <- as.matrix(d[, cols, drop = FALSE])
    dif <- max(abs(unclass(B) - P))
    ns_check[[sprintf("ns(%s, %d)", s$variable, as.integer(s$df))]] <- dif
    if (!is.finite(dif) || dif > spec$ns_check_tol)
      stop(sprintf("ns(%s): Python and R spline bases differ (max |diff| %g)", s$variable, dif))
    for (j in seq_along(cols)) d[[cols[j]]] <- as.numeric(B[, j])      # the R basis is used in the fit
  }
  used <- unique(c(spec$outcome, unlist(lapply(spec$fixed_terms, function(t) strsplit(t, ":", fixed = TRUE)[[1]]))))
  bad <- used[!used %in% names(d)]
  if (length(bad)) stop("model-input TSV lacks columns: ", paste(bad, collapse = ", "))
  na <- vapply(used, function(v) sum(!is.finite(as.numeric(d[[v]]))), 0)
  if (any(na > 0)) stop("non-finite values in model columns: ", paste(names(na)[na > 0], collapse = ", "))
  list(d = d, ns_check = ns_check)
}

# ------------------------------------------------------------------ random structure (= Python random_string)
re_string <- function(st) {
  ct <- unlist(st$corr_terms)
  has_int <- "1" %in% ct
  slopes <- setdiff(ct, "1")
  g <- if (is.null(st$group)) "pid" else st$group
  s <- character(0)
  if (isTRUE(st$corr)) {
    if (length(ct)) s <- c(s, sprintf("(%s | %s)", paste(c(if (has_int) "1" else "0", slopes), collapse = " + "), g))
  } else {
    if (has_int) s <- c(s, sprintf("(1 | %s)", g))
    s <- c(s, sprintf("(0 + %s | %s)", slopes, g))
  }
  s <- c(s, sprintf("(0 + %s | %s)", unlist(st$uncorrelated), g))
  if (isTRUE(st$block)) s <- c(s, sprintf("(1 | %s)", if (is.null(st$block_factor)) "pid_run_block" else st$block_factor))
  paste(s, collapse = " + ")
}

state_from_spec <- function(x) list(corr = isTRUE(x$corr), corr_terms = unlist(x$corr_terms),
                                    uncorrelated = unlist(x$uncorrelated), block = isTRUE(x$block),
                                    group = if (is.null(x$group)) "pid" else x$group,
                                    block_factor = if (is.null(x$block_factor)) "pid_run_block" else x$block_factor)

state_to_list <- function(st) list(corr = st$corr, corr_terms = as.list(st$corr_terms),
                                   uncorrelated = as.list(st$uncorrelated), block = st$block, group = st$group,
                                   block_factor = st$block_factor, re = re_string(st))

# ------------------------------------------------------------------ warnings / errors
.cap <- function(expr) {
  w <- character(0); m <- character(0)
  val <- withCallingHandlers(
    tryCatch(expr, error = function(e) structure(list(msg = conditionMessage(e)), class = "fit_error")),
    warning = function(wn) { w <<- c(w, conditionMessage(wn)); invokeRestart("muffleWarning") },
    message = function(mm) { m <<- c(m, conditionMessage(mm)); invokeRestart("muffleMessage") })
  list(val = val, warn = w, msg = m)
}

consensus_ok <- function(fe, se, ll, rel_se, tol_ll) {
  # fe: n_fits x p matrix; se: p vector (SE of the reference fit); ll: n_fits
  all(apply(fe, 2, function(v) diff(range(v))) <= rel_se * se) && diff(range(ll)) <= tol_ll
}

# ------------------------------------------------------------------ simplification path
# S1 fit the starting structure (allFit consensus on non-convergence, inside fitter); S3 on non-convergence drop the
# correlations of the child block; S4 on non-convergence or a singular fit drop the droppable covariate slopes in
# their fixed order, each only if the LRT p >= p_drop; S5 drop the block intercept if its SD is at the boundary.
# fitter(st) -> list(fit, conv, sing, allfit, err, info); block_sd_zero(fit) -> logical; lrt_p(reduced, current) -> p
reduce_path <- function(start, droppable, fitter, block_sd_zero, lrt_p, p_drop = 0.2) {
  st <- start
  log <- list()
  add <- function(step, info, st_) log[[length(log) + 1]] <<- list(step = step, info = info, re = re_string(st_))
  cur <- fitter(st)
  add("S1_fit", sprintf("conv=%s sing=%s allfit=%s", cur$conv, cur$sing, cur$allfit), st)
  if (isTRUE(cur$err)) {
    add("S6_terminal", paste("error:", cur$info), st)
    return(list(cur = cur, state = st, log = log, flag = "error"))
  }
  n_corr <- length(unlist(st$corr_terms))
  if (!cur$conv && isTRUE(st$corr) && n_corr >= 2) {        # S3: only non-convergence removes correlations
    st2 <- st; st2$corr <- FALSE
    red <- fitter(st2)
    if (!isTRUE(red$err)) { st <- st2; cur <- red }
    add("S3_uncorrelated", sprintf("conv=%s sing=%s allfit=%s err=%s", red$conv, red$sing, red$allfit,
                                   isTRUE(red$err)), st)
  }
  if (!cur$conv || isTRUE(cur$sing)) {                      # S4: droppable covariate slopes, fixed order, once each
    for (s in droppable) {
      if (!s %in% st$corr_terms) next
      st2 <- st; st2$corr_terms <- setdiff(st$corr_terms, s)
      red <- fitter(st2)
      p <- if (!isTRUE(red$err) && isTRUE(red$conv)) lrt_p(red$fit, cur$fit) else NA_real_
      if (is.finite(p) && p >= p_drop) {
        st <- st2; cur <- red
        add(paste0("S4_drop_", s), sprintf("p=%.4f", p), st)
      } else add(paste0("S4_keep_", s), sprintf("p=%s conv_reduced=%s", format(signif(p, 4)), red$conv), st)
    }
  }
  if (isTRUE(st$block) && isTRUE(block_sd_zero(cur$fit))) {   # S5: block intercept at the boundary
    st2 <- st; st2$block <- FALSE
    red <- fitter(st2)
    if (!isTRUE(red$err)) { st <- st2; cur <- red; add("S5_drop_block", "block-intercept SD at the boundary", st) }
    else add("S5_keep_block", paste("refit error:", red$info), st)
  }
  flag <- if (cur$conv) (if (isTRUE(cur$sing)) "singular_accepted" else "regular") else "nonconverged"
  add("S6_terminal", flag, st)
  list(cur = cur, state = st, log = log, flag = flag)
}

# ------------------------------------------------------------------ final-fit optimum check
# After the path has selected the final model (or after the fixed-structure fit), the selected fit is refitted with
# allFit over the five optimisers. If any refit that reports convergence attains a log-likelihood higher than
# the selected fit's by more than `tol` (0.01) and has finite fixed effects, the refit with the highest
# log-likelihood replaces the selected fit (ties: the first in the allFit order). Deterministic (it compares
# likelihoods of the same model on the same data). `refits`: list of list(name, loglik, converged,
# fixef_finite). Returns list(switch = TRUE / FALSE, kept = name or NA, gain = loglik gain or 0).
optimum_check_select <- function(ll0, refits, tol = 0.01) {
  best <- NA_character_; best_ll <- ll0 + tol
  for (r in refits) {
    ll <- as.numeric(r$loglik)
    if (isTRUE(r$converged) && isTRUE(r$fixef_finite) && length(ll) == 1 && is.finite(ll) && ll > best_ll) {
      best <- r$name; best_ll <- ll
    }
  }
  list(switch = !is.na(best), kept = best, gain = if (is.na(best)) 0 else best_ll - ll0)
}

fit_fixed_state <- function(st, fitter) {
  cur <- fitter(st)
  log <- list(list(step = "fixed_structure", info = sprintf("conv=%s sing=%s allfit=%s", cur$conv, cur$sing,
                                                             cur$allfit), re = re_string(st)))
  flag <- if (isTRUE(cur$err)) "error" else if (cur$conv) (if (isTRUE(cur$sing)) "singular_accepted" else "regular") else "nonconverged"
  list(cur = cur, state = st, log = log, flag = flag)
}

# ------------------------------------------------------------------ inference helpers
contrast_matrix <- function(targets, coef_names) {
  L <- matrix(0, nrow = length(targets), ncol = length(coef_names), dimnames = list(names(targets), coef_names))
  for (k in names(targets)) {
    w <- targets[[k]]
    for (nm in names(w)) {
      if (!nm %in% coef_names) stop(sprintf("target %s: coefficient %s not in the model", k, nm))
      L[k, nm] <- as.numeric(w[[nm]])
    }
  }
  L
}

ci_t <- function(est, se, df, level) {
  q <- if (is.finite(df)) stats::qt(1 - (1 - level) / 2, df) else stats::qnorm(1 - (1 - level) / 2)
  c(est - q * se, est + q * se)
}

p_t <- function(est, se, df) {
  z <- est / se
  if (is.finite(df)) 2 * stats::pt(-abs(z), df) else 2 * stats::pnorm(-abs(z))
}

infer_row <- function(est, se, df) {
  list(estimate = est, se = se, df = df, stat = est / se, p = p_t(est, se, df),
       ci95 = ci_t(est, se, df, 0.95), ci90 = ci_t(est, se, df, 0.90))
}

write_result <- function(res, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  tmp <- paste0(path, ".tmp")
  jsonlite::write_json(res, tmp, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
  file.rename(tmp, path)
}

varcorr_blocks <- function(vc_list) {
  # vc_list: named list of covariance matrices (lme4 VarCorr or glmmTMB VarCorr()$cond)
  out <- list()
  nms <- names(vc_list)
  for (i in seq_along(vc_list)) {
    S <- as.matrix(vc_list[[i]])
    out[[length(out) + 1]] <- list(group = sub("\\.[0-9]+$", "", nms[i]), name = nms[i], terms = as.list(colnames(S)),
                                   cov = lapply(seq_len(nrow(S)), function(r) as.numeric(S[r, ])),
                                   sd = as.numeric(sqrt(diag(S))))
  }
  out
}
