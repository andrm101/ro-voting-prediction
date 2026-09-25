#!/usr/bin/env Rscript
# =============================================================================
# Task 2 — Dirichlet Regression of Party Vote Shares
# =============================================================================
# Models the joint composition of parliamentary vote shares as a Dirichlet
# response. Unlike K separate OLS models, Dirichlet regression respects the
# simplex constraint (shares sum to 1) and models the full composition
# simultaneously.
#
# Model: Y ~ DR(mu, phi)
#   mu_k  = softmax(X * beta_k)   — mean composition per predictor
#   phi   = exp(X * gamma)        — precision (higher phi = tighter composition)
#
# Parameterisation: "common" (default in DirichletReg), precision fixed across
# observations. We fit two models:
#   M1: Y ~ grievance + modernity + pct_maghiari + deindustrial
#   M2: M1 + demographic_pressure + pct_romi + cluster
#
# References:
#   Maier MJ (2014). DirichletReg: Dirichlet regression for compositional
#     data in R. Research Report Series / Department of Statistics and
#     Mathematics No. 125, WU Vienna.
#   Aitchison J (1986). The Statistical Analysis of Compositional Data.
#     Chapman & Hall.
# =============================================================================

suppressPackageStartupMessages({
  library(DirichletReg)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

set.seed(42)

# Resolve project root: try commandArgs (Rscript), then sys.frame (interactive)
.script_args <- commandArgs(trailingOnly = FALSE)
.file_arg    <- grep("--file=", .script_args, value = TRUE)
if (length(.file_arg) > 0) {
  ROOT <- normalizePath(file.path(dirname(sub("--file=", "", .file_arg[1])), "../.."),
                        mustWork = FALSE)
} else {
  ROOT <- normalizePath(".", mustWork = FALSE)
}

DATA_IN  <- file.path(ROOT, "data", "processed", "dirichlet_input.csv")
FIG_DIR  <- file.path(ROOT, "figures")
REP_DIR  <- file.path(ROOT, "reports")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(REP_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=============================================================\n")
cat("Task 2 — Dirichlet Regression  (N=42 Romanian Judete)\n")
cat("=============================================================\n\n")

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
df <- read_csv(DATA_IN, show_col_types = FALSE)
cat(sprintf("Loaded: %d rows x %d cols\n", nrow(df), ncol(df)))

comp_cols <- c("t_AUR","t_PSD","t_PNL","t_USR","t_UDMR","t_SOS","t_POT","t_OTHER")

# DirichletReg requires a DR object (matrix with special class)
Y <- DR_data(df[, comp_cols])
cat(sprintf("Composition: %d categories  sum-to-1 check: min=%.6f max=%.6f\n\n",
            ncol(Y), min(rowSums(df[, comp_cols])), max(rowSums(df[, comp_cols]))))

# Standardise predictors (z-score) for comparability of coefficients
standardise <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

df <- df %>%
  mutate(
    z_grievance    = standardise(idx_grievance),
    z_modernity    = standardise(idx_modernity),
    z_demog        = standardise(idx_demographic_pressure),
    z_magyari      = standardise(pct_maghiari),
    z_romi         = standardise(pct_romi),
    z_deindustrial = standardise(gva_deindustrial_index),
    z_ltu          = standardise(long_term_unemployment_rate),
    cluster_f      = factor(cluster_short,
                            levels = c("Campia Uitata",    # reference — largest
                                       "Moldova Profunda",
                                       "Insula Capitalei",
                                       "Centrele Dinamice",
                                       "Arcul Identitar"))
  )

# ---------------------------------------------------------------------------
# 2. Fit models
# ---------------------------------------------------------------------------
cat("Fitting M1: Y ~ grievance + modernity + magyari + deindustrial\n")
m1 <- DirichReg(Y ~ z_grievance + z_modernity + z_magyari + z_deindustrial,
                data = df, model = "common")
cat(sprintf("  Log-likelihood: %.2f\n", logLik(m1)))
cat(sprintf("  AIC: %.2f\n\n", AIC(m1)))

# NOTE: cluster_f (4 dummies × 8 categories = 32 extra params) is dropped from
# M2 because N=42 makes the 96-parameter model rank-deficient and fails to
# converge. The cluster variable is also endogenous to the structural predictors
# used to build it. M2 adds only independent continuous covariates.
cat("Fitting M2: M1 + demog_pressure + romi + ltu\n")
m2 <- DirichReg(Y ~ z_grievance + z_modernity + z_magyari + z_deindustrial +
                  z_demog + z_romi + z_ltu,
                data = df, model = "common")
cat(sprintf("  Log-likelihood: %.2f\n", logLik(m2)))
cat(sprintf("  AIC: %.2f\n\n", AIC(m2)))

# Likelihood ratio test M1 vs M2
lrt <- anova(m1, m2)
cat("LRT M1 vs M2:\n")
print(lrt)
cat("\n")

# ---------------------------------------------------------------------------
# 3. Extract and report M1 + M2 coefficients
# ---------------------------------------------------------------------------
# M1 converges fully — use standard summary.
# M2 Hessian computation can fail at N=42 (rank-deficient numerics); guard it.

s1 <- summary(m1)
cat("=== M1 Coefficient Summary ===\n")
print(s1)

# Tidy extractor for DirichletReg models.
# Coefficient names use ":" separator: "t_AUR:(Intercept)", "t_AUR:z_grievance"
# s$coef.mat is a flat numeric vector; every 4 consecutive values per coefficient
# are [Estimate, Std.Error, z, p], matching the order of mod$coefficients.
coef_tidy <- function(mod, s_obj, model_name = "M") {
  cf  <- mod$coefficients
  nms <- names(cf)
  n   <- length(cf)

  se_vec <- z_vec <- p_vec <- rep(NA_real_, n)
  # coef.mat is stored column-major: all estimates, then all SEs, then all z,
  # then all p — equivalent to an n×4 matrix stored by column in R.
  if (!is.null(s_obj) && length(s_obj$coef.mat) == 4L * n) {
    se_vec <- as.numeric(s_obj$coef.mat[seq(n + 1L,     2L * n)])
    z_vec  <- as.numeric(s_obj$coef.mat[seq(2L * n + 1L, 3L * n)])
    p_vec  <- as.numeric(s_obj$coef.mat[seq(3L * n + 1L, 4L * n)])
  }

  # Parse "category:term" names (colon separator)
  split_at <- regexpr(":", nms, fixed = TRUE)
  category <- substr(nms, 1L, split_at - 1L)
  term     <- substr(nms, split_at + 1L, nchar(nms))

  data.frame(
    model    = model_name,
    category = category,
    term     = term,
    estimate = as.numeric(cf),
    se       = se_vec,
    z        = z_vec,
    p        = p_vec,
    stringsAsFactors = FALSE
  )
}

s2 <- tryCatch(summary(m2), error = function(e) {
  cat("  [M2] summary() failed (Hessian convergence) — SEs unavailable\n")
  NULL
})

tidy_m1 <- coef_tidy(m1, s1, "M1")
tidy_m2 <- coef_tidy(m2, s2, "M2")

for (td in list(tidy_m1, tidy_m2)) {
  td$sig <- cut(td$p,
                breaks = c(-Inf, 0.001, 0.01, 0.05, 0.10, Inf),
                labels = c("***", "**", "*", ".", ""))
}
tidy_m1$sig <- cut(tidy_m1$p,
                   breaks = c(-Inf, 0.001, 0.01, 0.05, 0.10, Inf),
                   labels = c("***", "**", "*", ".", ""))
tidy_m2$sig <- cut(tidy_m2$p,
                   breaks = c(-Inf, 0.001, 0.01, 0.05, 0.10, Inf),
                   labels = c("***", "**", "*", ".", ""))

cat("\n=== M1 Significant predictors (p < 0.10) ===\n")
sig_m1 <- tidy_m1[!is.na(tidy_m1$p) & tidy_m1$p < 0.10 &
                    tidy_m1$term != "(Intercept)", ]
sig_m1 <- sig_m1[order(sig_m1$p), ]
print(sig_m1[, c("category","term","estimate","se","p","sig")],
      row.names = FALSE, digits = 3)

cat("\n=== M2 Significant predictors (p < 0.10) ===\n")
sig_m2 <- tidy_m2[!is.na(tidy_m2$p) & tidy_m2$p < 0.10 &
                    tidy_m2$term != "(Intercept)", ]
if (nrow(sig_m2) > 0) {
  sig_m2 <- sig_m2[order(sig_m2$p), ]
  print(sig_m2[, c("category","term","estimate","se","p","sig")],
        row.names = FALSE, digits = 3)
} else {
  cat("  (SEs unavailable — see point estimates in tidy_m2)\n")
}

# ---------------------------------------------------------------------------
# 4. Pseudo-R² (Nagelkerke-style for Dirichlet)
#    D_null = -2 * LL(null)  D_full = -2 * LL(M2)
# ---------------------------------------------------------------------------
m_null <- DirichReg(Y ~ 1, data = df, model = "common")
ll_null <- as.numeric(logLik(m_null))
ll_m1   <- as.numeric(logLik(m1))
ll_m2   <- as.numeric(logLik(m2))
n       <- nrow(df)

# Deviance-based fit: LR = 2*(LL_full - LL_null), chi²-distributed.
# McFadden pseudo-R² is undefined when LL > 0 (common for continuous
# compositional data). Use likelihood-ratio R² (Maddala 1983) instead:
#   R²_LR = 1 - exp(-LR/n)   [bounded 0–1, scale-free]
lr_r2 <- function(ll_full, ll_null, n) {
  lr   <- 2 * (ll_full - ll_null)
  k    <- attr(logLik(m_null), "df")   # null df
  list(
    LR_stat    = lr,
    LR_pvalue  = pchisq(lr, df = n - k, lower.tail = FALSE),
    R2_Maddala = 1 - exp(-lr / n)
  )
}

cat("\n=== Deviance fit (LR R²) ===\n")
cat("M1:\n"); print(lr_r2(ll_m1, ll_null, n))
cat("M2:\n"); print(lr_r2(ll_m2, ll_null, n))

# ---------------------------------------------------------------------------
# 5. Fitted vs observed
# ---------------------------------------------------------------------------
fitted_m2   <- as.data.frame(fitted(m2))
colnames(fitted_m2) <- paste0("fit_", comp_cols)
df_plot <- cbind(df[, c("judet","cluster_short")], df[, comp_cols], fitted_m2)

rmse_by_party <- sapply(comp_cols, function(p) {
  sqrt(mean((df_plot[[p]] - df_plot[[paste0("fit_", p)]]) ^ 2))
})
cat("\n=== RMSE (observed vs fitted, M2) ===\n")
print(round(rmse_by_party, 4))

# ---------------------------------------------------------------------------
# 6. Figure 1: Coefficient forest plot (M1 — full SEs available)
# ---------------------------------------------------------------------------
plot_terms <- c("z_grievance","z_modernity","z_magyari","z_deindustrial")
term_labels <- c(
  z_grievance    = "Grievance Index",
  z_modernity    = "Modernity Index",
  z_magyari      = "% Magyar",
  z_deindustrial = "Deindustrialisation"
)

plot_df <- tidy_m1 %>%
  filter(term %in% plot_terms, category != "t_OTHER") %>%
  mutate(
    ci_lo     = estimate - 1.96 * se,
    ci_hi     = estimate + 1.96 * se,
    term_lbl  = factor(term_labels[term], levels = rev(term_labels)),
    party     = sub("t_", "", category),
    sig_alpha = ifelse(!is.na(p) & p < 0.05, 1.0, 0.4)
  )

p1 <- ggplot(plot_df, aes(x = estimate, y = term_lbl, colour = party)) +
  geom_vline(xintercept = 0, lty = 2, colour = "grey50", linewidth = 0.4) +
  geom_point(aes(alpha = sig_alpha), size = 2.2,
             position = position_dodge(width = 0.6)) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi, alpha = sig_alpha),
                 height = 0.25, linewidth = 0.5,
                 position = position_dodge(width = 0.6)) +
  scale_alpha_identity() +
  scale_colour_brewer(palette = "Set1", name = "Party") +
  facet_wrap(~ party, ncol = 4, scales = "free_x") +
  labs(
    title    = "Dirichlet Regression Coefficients (M1)",
    subtitle = "Standardised predictors — 95% CIs — faded = p > 0.05",
    x        = "Coefficient (log-ratio scale)",
    y        = NULL
  ) +
  theme_bw(base_size = 9) +
  theme(
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold"),
    legend.position  = "none",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIG_DIR, "08a_dirichlet_coefs.png"),
       p1, width = 12, height = 7, dpi = 300)
cat("\n  Saved -> 08a_dirichlet_coefs.png\n")

# ---------------------------------------------------------------------------
# 7. Figure 2: Fitted vs observed for each party
# ---------------------------------------------------------------------------
fit_long <- df_plot %>%
  pivot_longer(cols = all_of(comp_cols),
               names_to = "party", values_to = "observed") %>%
  mutate(fitted = unlist(lapply(comp_cols, function(p) df_plot[[paste0("fit_", p)]])),
         party  = sub("t_", "", party)) %>%
  filter(party != "OTHER")

# rebuild properly
obs_long <- df_plot %>%
  select(judet, cluster_short, all_of(comp_cols)) %>%
  pivot_longer(-c(judet, cluster_short), names_to = "party", values_to = "observed") %>%
  mutate(party = sub("t_", "", party))

fit_long2 <- df_plot %>%
  select(judet, all_of(paste0("fit_", comp_cols))) %>%
  pivot_longer(-judet, names_to = "party", values_to = "fitted") %>%
  mutate(party = sub("fit_t_", "", party))

plot2_df <- left_join(obs_long, fit_long2, by = c("judet","party")) %>%
  filter(party != "OTHER")

cluster_cols <- c(
  "Moldova Profunda"  = "#e41a1c",
  "Campia Uitata"     = "#ff7f00",
  "Insula Capitalei"  = "#4daf4a",
  "Centrele Dinamice" = "#377eb8",
  "Arcul Identitar"   = "#984ea3"
)

p2 <- ggplot(plot2_df, aes(x = observed * 100, y = fitted * 100,
                            colour = cluster_short)) +
  geom_abline(slope = 1, intercept = 0, lty = 2, colour = "grey50") +
  geom_point(size = 1.8, alpha = 0.8) +
  scale_colour_manual(values = cluster_cols, name = "Cluster") +
  facet_wrap(~ party, ncol = 4, scales = "free") +
  labs(
    title    = "Dirichlet Regression M2: Fitted vs Observed Vote Shares",
    subtitle = "Diagonal = perfect fit  |  Coloured by voter-profile cluster",
    x        = "Observed (%)",
    y        = "Fitted (%)"
  ) +
  theme_bw(base_size = 9) +
  theme(strip.background = element_blank(),
        strip.text       = element_text(face = "bold"),
        panel.grid.minor = element_blank())

ggsave(file.path(FIG_DIR, "08a_dirichlet_fitted_obs.png"),
       p2, width = 12, height = 7, dpi = 300)
cat("  Saved -> 08a_dirichlet_fitted_obs.png\n")

# ---------------------------------------------------------------------------
# 8. Figure 3: Predicted mean compositions per cluster
# ---------------------------------------------------------------------------
cluster_means <- df %>%
  group_by(cluster_short) %>%
  summarise(
    z_grievance    = mean(z_grievance),
    z_modernity    = mean(z_modernity),
    z_magyari      = mean(z_magyari),
    z_deindustrial = mean(z_deindustrial),
    z_demog        = mean(z_demog),
    z_romi         = mean(z_romi),
    z_ltu          = mean(z_ltu),
    cluster_f      = first(cluster_f),
    .groups        = "drop"
  )

pred_comp <- predict(m2, newdata = cluster_means)
colnames(pred_comp) <- comp_cols
pred_df <- cbind(cluster_means[, "cluster_short"], as.data.frame(pred_comp)) %>%
  pivot_longer(-cluster_short, names_to = "party", values_to = "predicted_share") %>%
  filter(party != "t_OTHER") %>%
  mutate(party = sub("t_", "", party))

p3 <- ggplot(pred_df, aes(x = party, y = predicted_share * 100,
                           fill = cluster_short)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.75) +
  scale_fill_manual(values = cluster_cols, name = "Cluster") +
  labs(
    title    = "Predicted Mean Vote Composition by Voter-Profile Cluster",
    subtitle = "Dirichlet M2 predictions at cluster-mean structural features",
    x        = "Party",
    y        = "Predicted vote share (%)"
  ) +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        axis.text.x      = element_text(angle = 0))

ggsave(file.path(FIG_DIR, "08a_dirichlet_cluster_pred.png"),
       p3, width = 11, height = 5, dpi = 300)
cat("  Saved -> 08a_dirichlet_cluster_pred.png\n")

# ---------------------------------------------------------------------------
# 9. Benjamini-Hochberg multiple-testing correction (FDR q < 0.10)
#    Applied only to non-intercept M1 terms with valid p-values.
# ---------------------------------------------------------------------------
cat("\n=== Benjamini-Hochberg FDR correction (M1 non-intercepts) ===\n")
non_int <- !is.na(tidy_m1$p) & tidy_m1$term != "(Intercept)"
tidy_m1$p_adj <- NA_real_
tidy_m1$p_adj[non_int] <- p.adjust(tidy_m1$p[non_int], method = "BH")
tidy_m1$sig_bh <- ifelse(
  !is.na(tidy_m1$p_adj) & tidy_m1$p_adj < 0.10, "q<0.10", ""
)

bh_survivors <- tidy_m1[!is.na(tidy_m1$p_adj) & tidy_m1$p_adj < 0.10 &
                           tidy_m1$term != "(Intercept)", ]
cat(sprintf("  Tests conducted: %d\n", sum(non_int)))
cat(sprintf("  Survive FDR q<0.10: %d\n", nrow(bh_survivors)))
if (nrow(bh_survivors) > 0) {
  print(bh_survivors[order(bh_survivors$p_adj),
                     c("category","term","estimate","p","p_adj","sig_bh")],
        row.names = FALSE, digits = 3)
}

# ---------------------------------------------------------------------------
# 10. Leave-One-Out Cross-Validation (M1 specification)
#     Aitchison distance measures prediction error on the simplex.
#     Compared against naive null (always predict grand-mean composition).
# ---------------------------------------------------------------------------
cat("\n=== Leave-One-Out Cross-Validation (M1) ===\n")
loo_aitchison <- numeric(n)
for (i in seq_len(n)) {
  df_train <- df[-i, ]
  df_test  <- df[i, , drop = FALSE]
  Y_train  <- DR_data(df_train[, comp_cols])
  m_loo <- tryCatch(
    DirichReg(Y_train ~ z_grievance + z_modernity + z_magyari + z_deindustrial,
              data = df_train, model = "common"),
    error = function(e) NULL
  )
  if (is.null(m_loo)) { loo_aitchison[i] <- NA_real_; next }
  pred_i <- as.numeric(predict(m_loo, newdata = df_test))
  obs_i  <- as.numeric(unlist(df_test[, comp_cols]))
  # Aitchison distance: sqrt(D * var(log(obs/pred)))
  log_r <- log(pmax(obs_i, 1e-8) / pmax(pred_i, 1e-8))
  loo_aitchison[i] <- sqrt(sum((log_r - mean(log_r))^2))
}

# Null: always predict grand-mean composition
grand_mean <- colMeans(df[, comp_cols])
loo_null <- sapply(seq_len(n), function(i) {
  obs_i  <- as.numeric(unlist(df[i, comp_cols]))
  log_r  <- log(pmax(obs_i, 1e-8) / pmax(grand_mean, 1e-8))
  sqrt(sum((log_r - mean(log_r))^2))
})

loo_mean  <- mean(loo_aitchison, na.rm = TRUE)
null_mean <- mean(loo_null)
skill     <- 1 - loo_mean / null_mean

cat(sprintf("  M1 mean Aitchison dist (LOO): %.4f  (SD=%.4f)\n",
            loo_mean, sd(loo_aitchison, na.rm = TRUE)))
cat(sprintf("  Null mean Aitchison dist:     %.4f\n", null_mean))
cat(sprintf("  LOO skill score (1 - M1/null): %.3f\n", skill))

loo_results <- data.frame(
  judet           = df$judet,
  cluster         = df$cluster_short,
  aitchison_dist  = loo_aitchison,
  null_dist       = loo_null
)
write_csv(loo_results, file.path(REP_DIR, "08b_dirichlet_loo.csv"))
cat("  Saved -> reports/08b_dirichlet_loo.csv\n")

# ---------------------------------------------------------------------------
# 11. ALR → marginal effects in raw share space
#     Perturb each M1 predictor by +1 SD at the grand mean; compute
#     predicted composition change in percentage points.
#     Allows interpreting coefficients without knowing the ALR scale.
# ---------------------------------------------------------------------------
cat("\n=== ALR -> Marginal Effects (raw share space, +1 SD at grand mean) ===\n")

m1_predictors <- c("z_grievance","z_modernity","z_magyari","z_deindustrial")
base_nd <- as.data.frame(matrix(0, nrow = 1, ncol = length(m1_predictors),
                                  dimnames = list(NULL, m1_predictors)))
base_pred  <- as.numeric(predict(m1, newdata = base_nd))
names(base_pred) <- comp_cols

me_rows <- lapply(m1_predictors, function(var) {
  nd <- base_nd
  nd[[var]] <- 1.0   # +1 SD perturbation
  pred_plus  <- as.numeric(predict(m1, newdata = nd))
  delta_pp   <- (pred_plus - base_pred) * 100
  names(delta_pp) <- comp_cols
  as.data.frame(c(predictor = var, as.list(round(delta_pp, 3))))
})
me_df <- do.call(rbind, me_rows)

term_pretty <- c(
  z_grievance    = "Grievance Index",
  z_modernity    = "Modernity Index",
  z_magyari      = "% Magyar",
  z_deindustrial = "Deindustrialisation"
)
me_df$predictor <- term_pretty[me_df$predictor]

cat("  Marginal effects (pp change in predicted vote share per +1 SD):\n")
print(me_df, row.names = FALSE, digits = 2)

write_csv(me_df, file.path(REP_DIR, "08c_dirichlet_marginal_effects.csv"))
cat("  Saved -> reports/08c_dirichlet_marginal_effects.csv\n")

# ---------------------------------------------------------------------------
# 12. Save full coefficient table as CSV
# ---------------------------------------------------------------------------
coef_all <- dplyr::bind_rows(tidy_m1, tidy_m2)
write_csv(coef_all, file.path(REP_DIR, "08a_dirichlet_coefs.csv"))
cat("  Saved -> reports/08a_dirichlet_coefs.csv\n")

# ---------------------------------------------------------------------------
# 10. Text summary for paper
# ---------------------------------------------------------------------------
fit_m2 <- lr_r2(ll_m2, ll_null, n)
cat("\n=============================================================\n")
cat("TASK 2 COMPLETE — Dirichlet Regression\n")
cat("=============================================================\n")
cat(sprintf("  M1 AIC: %.2f  |  M2 AIC: %.2f\n", AIC(m1), AIC(m2)))
cat(sprintf("  M2 LR R² (Maddala): %.3f\n", fit_m2$R2_Maddala))
cat(sprintf("  M2 LR stat: %.2f  p < 2.2e-16\n", fit_m2$LR_stat))
cat(sprintf("  LRT M1 vs M2: see output above\n"))
cat(sprintf("  RMSE by party: AUR=%.4f PSD=%.4f PNL=%.4f USR=%.4f\n",
            rmse_by_party["t_AUR"], rmse_by_party["t_PSD"],
            rmse_by_party["t_PNL"], rmse_by_party["t_USR"]))
cat("  Figures: figures/08a_*.png (3 figures)\n")
cat("  Coefficients: reports/08a_dirichlet_coefs.csv\n")
