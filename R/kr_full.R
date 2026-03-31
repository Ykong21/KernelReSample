#' KernelReSample: shared-weight bootstrap KRR for a full test set
#'
#' Fits the full-test-set simultaneous variant of the KernelReSample workflow.
#' The procedure first clusters the training covariates to obtain centers,
#' computes center labels by cluster means of the responses, tunes the
#' regularization parameter (and optionally the center bandwidth scale) by BIC
#' on the center-level problem, constructs a single shared sampling probability
#' vector for the whole test set, and then performs bootstrap KRR fitting on
#' resampled subdata to predict all test points jointly.
#'
#' Compared with the pointwise implementation in \code{kr_boot_plus}, this
#' function uses one shared probability vector for the entire test set rather
#' than building test-point-specific probabilities. This substantially reduces
#' runtime when the test set is moderately large, at the cost of replacing the
#' pointwise weighting rule by a simultaneous weighting rule.
#'
#' @param N Integer. Number of training points (must equal \code{nrow(x)}).
#' @param n Integer. Bootstrap subdata size drawn at each repetition.
#' @param B Integer. Number of bootstrap resamples.
#' @param x Matrix. Training covariates of size \code{N x p}.
#' @param y Numeric vector of length \code{N}. Training responses.
#' @param x0s Matrix. Test covariates of size \code{m x p}.
#' @param y0s Numeric vector of length \code{m}. True responses at \code{x0s},
#'   used for evaluation.
#' @param lambda0 Numeric vector. Candidate regularization parameters for BIC
#'   tuning on the center-level problem.
#'
#' @param kernel Character. Kernel family: \code{"gaussian"} (default),
#'   \code{"laplace"}, or \code{"matern"}.
#' @param matern_nu Numeric. Mat\'ern smoothness parameter, one of
#'   \code{0.5}, \code{1.5}, or \code{2.5}. Used only when
#'   \code{kernel = "matern"}.
#'
#' @param clustering Character. Clustering backend used to construct centers:
#'   \code{"kmeans"} (default), \code{"kmeans_pp"}, \code{"kmeans_rp"}, or
#'   \code{"kmedoids"}.
#' @param n_centers Integer. Number of centers/clusters used in the weighting
#'   stage. Default is \code{200}.
#' @param kmeans_maxiter Integer. Maximum iterations for k-means backends.
#' @param rp_dim Integer. Random projection dimension for
#'   \code{clustering = "kmeans_rp"}.
#'
#' @param bw_centers_method Character. Bandwidth rule for the center-level
#'   weighting stage: \code{"median"} (default) or \code{"detcov"}.
#' @param bw_scope Character. Scope used when computing the center bandwidth:
#'   \code{"centers"} (default) uses the centers themselves, whereas
#'   \code{"train_sample"} uses a subsample of the training covariates.
#' @param bw_train_sample Integer. Subsample size used when
#'   \code{bw_scope = "train_sample"}.
#' @param tune_bw Logical. If \code{TRUE}, allow bandwidth scaling over
#'   \code{bw_scales} during center-stage tuning.
#' @param bw_scales Numeric vector. Multiplicative scales applied to the base
#'   center bandwidth when \code{tune_bw = TRUE}.
#' @param bw_subset_method Character. Bandwidth rule recomputed on each
#'   bootstrap subset: \code{"median"} (default) or \code{"detcov"}.
#' @param joint_tune Logical. If \code{TRUE} and \code{tune_bw = TRUE}, jointly
#'   tune the bandwidth scale and regularization parameter by BIC. Otherwise,
#'   only \code{lambda0} is tuned while the center bandwidth is fixed.
#'
#' @param full_weight_mode Character. Rule used to aggregate the full test set
#'   into a single shared center-weight vector:
#'   \itemize{
#'     \item \code{"transformed"}: uses the transformed center features
#'       \eqn{A_W^{-1} K_{C,\mathrm{test}}} and sets center weights by row norms.
#'     \item \code{"direct"}: uses direct norms of \eqn{K_{\mathrm{test},C}}.
#'   }
#' @param verbose Logical. Print progress messages.
#'
#' @return A list containing
#' \itemize{
#'   \item \code{y0s}: true test responses.
#'   \item \code{est}: \code{m x 1} matrix of averaged bootstrap predictions.
#'   \item \code{MSE}: pointwise squared prediction errors.
#'   \item \code{Ave_MSE}: average test-set MSE.
#'   \item \code{fx0s_boot}: \code{B x m} matrix of bootstrap predictions.
#'   \item \code{finalprobs}: shared sampling probability vector over the
#'     training sample.
#'   \item \code{omega_centers}: normalized center weights used to induce
#'     \code{finalprobs}.
#'   \item tuning and metadata fields such as \code{lambda_opt},
#'     \code{bw_centers}, \code{kernel}, \code{clustering},
#'     \code{full_weight_mode}, and \code{timing}.
#' }
#'
#' @examples
#' set.seed(1)
#' N <- 2000
#' p <- 5
#' n <- 300
#' B <- 3
#' x <- matrix(rnorm(N * p), N, p)
#' f_true <- function(z) sin(2 * z[, 1]) + 0.5 * z[, 2]^2
#' y <- f_true(x) + rnorm(N, sd = 0.5)
#' x0s <- matrix(rnorm(50 * p), 50, p)
#' y0s <- f_true(x0s)
#' lambda0 <- 10^seq(-2, 2, length.out = 8)
#'
#' out <- kr_full(
#'   N = N, n = n, B = B,
#'   x = x, y = y,
#'   x0s = x0s, y0s = y0s,
#'   lambda0 = lambda0,
#'   kernel = "gaussian",
#'   clustering = "kmeans",
#'   n_centers = 50,
#'   bw_centers_method = "median",
#'   bw_subset_method = "median",
#'   full_weight_mode = "transformed",
#'   verbose = FALSE
#' )
#' out$Ave_MSE
#' head(out$finalprobs)
#'
#' @export
kr_full <- function(
    N, n, B, x, y, x0s, y0s, lambda0,
    kernel = c("gaussian", "laplace", "matern"),
    matern_nu = c(0.5, 1.5, 2.5),
    clustering = c("kmeans", "kmeans_pp", "kmeans_rp", "kmedoids"),
    n_centers = 200,
    kmeans_maxiter = 50,
    rp_dim = 24,
    bw_centers_method = c("median", "detcov"),
    bw_scope = c("centers", "train_sample"),
    bw_train_sample = 2000,
    tune_bw = FALSE,
    bw_scales = c(0.25, 0.5, 1, 2),
    bw_subset_method = c("median", "detcov"),
    joint_tune = FALSE,
    full_weight_mode = c("transformed", "direct"),
    verbose = FALSE
) {
  # ---- coerce / match args ----
  x <- as.matrix(x)
  x0s <- as.matrix(x0s)
  y <- as.numeric(y)
  y0s <- as.numeric(y0s)

  kernel <- match.arg(kernel)
  bw_centers_method <- match.arg(bw_centers_method)
  bw_scope <- match.arg(bw_scope)
  bw_subset_method <- match.arg(bw_subset_method)
  clustering <- match.arg(clustering)
  full_weight_mode <- match.arg(full_weight_mode)

  if (kernel == "matern") {
    matern_nu <- match.arg(as.character(matern_nu), c("0.5", "1.5", "2.5"))
    matern_nu <- as.numeric(matern_nu)
  } else {
    matern_nu <- NULL
  }

  # ---- checks ----
  if (nrow(x) != N) stop("N must equal nrow(x).")
  if (length(y) != N) stop("y must have length N.")
  if (n <= 0 || n > N) stop("n must be in {1, ..., N}.")
  if (B <= 0) stop("B must be positive.")
  if (ncol(x0s) != ncol(x)) stop("x0s must have same number of columns as x.")
  if (length(y0s) != nrow(x0s)) stop("y0s must have length nrow(x0s).")
  if (!is.numeric(lambda0) || length(lambda0) < 1) {
    stop("lambda0 must be a numeric vector.")
  }
  if (!is.numeric(n_centers) || length(n_centers) != 1 || n_centers < 2) {
    stop("n_centers must be an integer >= 2.")
  }
  n_centers <- min(as.integer(n_centers), N)

  # ---- timing ----
  t_all <- proc.time()
  timing <- list(
    cluster = NA_real_,
    ystar = NA_real_,
    tune = NA_real_,
    weights = NA_real_,
    est = NA_real_,
    total = NA_real_
  )

  # ---------------------------
  # Utilities
  # ---------------------------
  sqdist <- function(A, B) {
    A <- as.matrix(A)
    B <- as.matrix(B)
    AA <- rowSums(A^2)
    BB <- rowSums(B^2)
    D2 <- outer(AA, BB, "+") - 2 * (A %*% t(B))
    D2[D2 < 0] <- 0
    D2
  }

  median_dist <- function(Z, max_m = 2000L) {
    Z <- as.matrix(Z)
    nZ <- nrow(Z)
    if (nZ <= 1) return(1)
    take <- min(nZ, max_m)
    id <- if (nZ > take) sample.int(nZ, take) else seq_len(nZ)
    Zs <- Z[id, , drop = FALSE]
    D2 <- sqdist(Zs, Zs)
    D <- sqrt(D2[upper.tri(D2)])
    md <- median(D[D > 0], na.rm = TRUE)
    if (!is.finite(md) || md <= 0) md <- 1
    md
  }

  detcov_safe <- function(Z) {
    Z <- as.matrix(Z)
    if (nrow(Z) <= 1 || ncol(Z) == 0) return(1)
    S <- tryCatch(stats::cov(Z), error = function(e) NULL)
    if (is.null(S)) return(1)
    if (length(S) == 1L) return(max(as.numeric(S), 1e-12))
    ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
    ev <- pmax(ev, 1e-12)
    prod(ev)
  }

  bw_from_points <- function(Z, method, kernel, max_m = 2000L) {
    Z <- as.matrix(Z)
    if (method == "detcov") {
      d <- detcov_safe(Z)
      if (kernel == "gaussian") {
        return(max(d, 1e-12))
      } else {
        p <- ncol(Z)
        return(max(d^(1 / (2 * p)), 1e-12))
      }
    } else {
      md <- median_dist(Z, max_m = max_m)
      if (kernel == "gaussian") return(max(md^2, 1e-12))
      return(max(md, 1e-12))
    }
  }

  kernel_mat <- function(A, B, bw) {
    D2 <- sqdist(A, B)
    if (kernel == "gaussian") {
      return(exp(-D2 / max(bw, 1e-12)))
    }
    D <- sqrt(D2)
    ell <- max(bw, 1e-12)
    if (kernel == "laplace") {
      return(exp(-D / ell))
    }
    if (matern_nu == 0.5) {
      return(exp(-D / ell))
    } else if (matern_nu == 1.5) {
      s <- sqrt(3) * D / ell
      return((1 + s) * exp(-s))
    } else {
      s <- sqrt(5) * D / ell
      return((1 + s + (s^2) / 3) * exp(-s))
    }
  }

  chol_solve_spd <- function(A, b) {
    R <- chol(A)
    backsolve(R, forwardsolve(t(R), b))
  }

  kmeanspp_seed <- function(X, k) {
    X <- as.matrix(X)
    N0 <- nrow(X)
    centers <- matrix(0, k, ncol(X))
    centers[1, ] <- X[sample.int(N0, 1), ]
    d2 <- rowSums((X - matrix(centers[1, ], N0, ncol(X), byrow = TRUE))^2)
    for (j in 2:k) {
      s <- sum(d2)
      if (!is.finite(s) || s <= 0) {
        idx <- sample.int(N0, 1)
      } else {
        probs <- d2 / s
        idx <- sample.int(N0, 1, prob = probs)
      }
      centers[j, ] <- X[idx, ]
      d2_new <- rowSums((X - matrix(centers[j, ], N0, ncol(X), byrow = TRUE))^2)
      d2 <- pmin(d2, d2_new)
    }
    centers
  }

  do_clustering <- function(x, k) {
    x <- as.matrix(x)
    N <- nrow(x)

    if (clustering == "kmeans") {
      fit <- stats::kmeans(x, centers = k, iter.max = kmeans_maxiter)
      return(list(idx = fit$cluster, centers = fit$centers))
    }

    if (clustering == "kmeans_pp") {
      initC <- kmeanspp_seed(x, k)
      fit <- stats::kmeans(x, centers = initC, iter.max = kmeans_maxiter)
      return(list(idx = fit$cluster, centers = fit$centers))
    }

    if (clustering == "kmeans_rp") {
      p <- ncol(x)
      d <- min(rp_dim, p)
      R <- matrix(stats::rnorm(p * d), p, d) / sqrt(d)
      xr <- x %*% R
      fit <- stats::kmeans(xr, centers = k, iter.max = kmeans_maxiter)
      idx <- fit$cluster
      centers <- matrix(0, k, p)
      for (j in 1:k) {
        sel <- which(idx == j)
        if (length(sel) == 0) {
          centers[j, ] <- x[sample.int(N, 1), ]
        } else {
          centers[j, ] <- colMeans(x[sel, , drop = FALSE])
        }
      }
      return(list(idx = idx, centers = centers))
    }

    if (!requireNamespace("cluster", quietly = TRUE)) {
      stop("Package 'cluster' is required for clustering = 'kmedoids'.")
    }
    pam_fit <- cluster::pam(x, k = k)
    idx <- pam_fit$clustering
    centers <- x[pam_fit$id.med, , drop = FALSE]
    list(idx = idx, centers = centers)
  }

  tune_lambda_bw_centers <- function(centers, ystar, base_bw, lambda_grid, scales, joint) {
    best <- list(score = Inf, lam = lambda_grid[1], bw = base_bw)

    for (s in scales) {
      bw_try <- base_bw * s
      KC <- kernel_mat(centers, centers, bw_try)
      KC <- (KC + t(KC)) / 2 + 1e-10 * diag(nrow(KC))

      eg <- eigen(KC, symmetric = TRUE)
      evals <- pmax(eg$values, 0)
      Q <- eg$vectors
      Vy <- drop(t(Q) %*% ystar)

      for (lam in lambda_grid) {
        frac <- evals / (evals + lam)
        fc <- Q %*% (frac * Vy)
        resid <- ystar - drop(fc)
        res2 <- max(sum(resid^2), .Machine$double.eps)
        df <- sum(frac)
        bic <- nrow(centers) * log(res2) + log(nrow(centers)) * df

        if (bic < best$score) {
          best <- list(score = bic, lam = lam, bw = bw_try)
        }
      }

      if (!joint) break
    }
    best
  }

  # =========================
  # 1) clustering
  # =========================
  if (verbose) message("[kr_full] clustering...")
  t0 <- proc.time()
  cl <- do_clustering(x, n_centers)
  timing$cluster <- (proc.time() - t0)[3]
  idx <- as.integer(cl$idx)
  centers <- as.matrix(cl$centers)

  # =========================
  # 2) ystar per center
  # =========================
  if (verbose) message("[kr_full] center labels (cluster means)...")
  t0 <- proc.time()

  counts <- tabulate(idx, nbins = n_centers)
  sumy <- tapply(y, idx, sum)
  sumy <- as.numeric(sumy)
  sumy[is.na(sumy)] <- 0
  cnt <- counts
  cnt[cnt == 0] <- 1
  ystar <- sumy / cnt

  timing$ystar <- (proc.time() - t0)[3]

  # =========================
  # 3) choose centers bandwidth + lambda
  # =========================
  if (verbose) message("[kr_full] tuning lambda (and optionally bw_scale) on centers...")
  t0 <- proc.time()

  base_bw <- if (bw_scope == "centers") {
    bw_from_points(centers, bw_centers_method, kernel)
  } else {
    S <- min(bw_train_sample, nrow(x))
    bw_from_points(x[sample.int(nrow(x), S), , drop = FALSE], bw_centers_method, kernel)
  }

  scales <- if (isTRUE(tune_bw)) bw_scales else 1
  tuned <- tune_lambda_bw_centers(
    centers = centers,
    ystar = ystar,
    base_bw = base_bw,
    lambda_grid = lambda0,
    scales = scales,
    joint = isTRUE(joint_tune) && isTRUE(tune_bw)
  )

  lambda_opt <- tuned$lam
  bw_centers <- tuned$bw
  timing$tune <- (proc.time() - t0)[3]

  # =========================
  # 4) ONE shared probability vector for whole test set
  # =========================
  if (verbose) message("[kr_full] computing shared sampling probabilities...")
  t0 <- proc.time()

  KC <- kernel_mat(centers, centers, bw_centers)
  KC <- (KC + t(KC)) / 2 + 1e-10 * diag(n_centers)

  tau <- 1 / lambda_opt
  A_W <- diag(n_centers) + tau * KC

  Kxc <- kernel_mat(x0s, centers, bw_centers)

  if (full_weight_mode == "transformed") {
    Kx0s <- chol_solve_spd(A_W, t(Kxc))
    omega_centers <- sqrt(rowSums(Kx0s^2))
  } else {
    omega_centers <- sqrt(colSums(Kxc^2))
  }

  omega_centers[!is.finite(omega_centers) | omega_centers < 0] <- 0
  s <- sum(omega_centers)
  if (!is.finite(s) || s <= 0) {
    omega_centers <- rep(1 / n_centers, n_centers)
  } else {
    omega_centers <- omega_centers / s
  }

  finalweights <- (n_centers / N) * omega_centers
  finalprobs <- finalweights[idx]
  finalprobs[!is.finite(finalprobs) | finalprobs < 0] <- 0
  s <- sum(finalprobs)
  if (!is.finite(s) || s <= 0) {
    finalprobs <- rep(1 / N, N)
  } else {
    finalprobs <- finalprobs / s
  }

  timing$weights <- (proc.time() - t0)[3]

  # =========================
  # 5) bootstrap estimation: fit once per bootstrap, predict all x0s
  # =========================
  if (verbose) message("[kr_full] bootstrap KRR...")
  t0 <- proc.time()

  m_test <- nrow(x0s)
  fx0s <- matrix(0, B, m_test)

  for (b in seq_len(B)) {
    XIND <- sample.int(N, n, replace = TRUE, prob = finalprobs)
    xsub <- x[XIND, , drop = FALSE]
    ysub <- y[XIND]

    bw_sub <- bw_from_points(xsub, bw_subset_method, kernel)

    Kss <- kernel_mat(xsub, xsub, bw_sub)
    Kss <- (Kss + t(Kss)) / 2 + 1e-10 * diag(n)

    A <- diag(n) + tau * Kss
    rhs <- tau * ysub
    a <- chol_solve_spd(A, rhs)

    Ksub_test <- kernel_mat(xsub, x0s, bw_sub)
    fx0s[b, ] <- drop(t(Ksub_test) %*% a)
  }

  est <- matrix(colMeans(fx0s), m_test, 1)
  mse <- (y0s - est[, 1])^2
  Ave_MSE <- mean(mse)

  timing$est <- (proc.time() - t0)[3]
  timing$total <- (proc.time() - t_all)[3]

  list(
    y0s = y0s,
    est = est,
    MSE = mse,
    Ave_MSE = Ave_MSE,
    fx0s_boot = fx0s,
    finalprobs = finalprobs,
    omega_centers = omega_centers,
    lambda_opt = lambda_opt,
    bw_centers = bw_centers,
    bw_centers_method = bw_centers_method,
    bw_subset_method = bw_subset_method,
    clustering = clustering,
    kernel = kernel,
    matern_nu = if (kernel == "matern") matern_nu else NULL,
    full_weight_mode = full_weight_mode,
    timing = timing
  )
}
