# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' GIVP configuration R6 class
#' @export
# nolint nextline object_name_linter
GIVPConfig <- R6::R6Class(
  "GIVPConfig",
  public = list(
    max_iterations = 100L,
    alpha = 0.12,
    vnd_iterations = 200L,
    ils_iterations = 10L,
    perturbation_strength = 4L,
    use_elite_pool = TRUE,
    elite_size = 7L,
    path_relink_frequency = 8L,
    adaptive_alpha = TRUE,
    alpha_min = 0.08,
    alpha_max = 0.18,
    num_candidates_per_step = 20L,
    use_cache = TRUE,
    cache_size = 10000L,
    early_stop_threshold = 80L,
    use_convergence_monitor = TRUE,
    n_workers = 1L,
    time_limit = 0,
    direction = "minimize",
    integer_split = NULL,
    group_size = NULL,
    initial_guess = NULL,
    seed = NULL,
    verbose = FALSE,
    initialize = function(...) {
      dots <- list(...)
      if (length(dots) > 0L) {
        for (nm in names(dots)) {
          if (!nm %in% names(self)) {
            abort_invalid_config(paste0("Unknown config field: ", nm))
          }
          self[[nm]] <- dots[[nm]]
        }
      }
      private$validate()
    },
    as_list = function() {
      out <- list()
      for (nm in names(self)) {
        if (!identical(nm, "clone") && !is.function(self[[nm]])) {
          out[[nm]] <- self[[nm]]
        }
      }
      out
    }
  ),
  private = list(
    validate = function() {
      if (self$max_iterations < 1L) {
        abort_invalid_config("max_iterations must be >= 1")
      }
      if (self$vnd_iterations < 1L) {
        abort_invalid_config("vnd_iterations must be >= 1")
      }
      if (self$ils_iterations < 1L) {
        abort_invalid_config("ils_iterations must be >= 1")
      }
      if (self$elite_size < 1L) abort_invalid_config("elite_size must be >= 1")
      if (self$path_relink_frequency < 1L) {
        abort_invalid_config("path_relink_frequency must be >= 1")
      }
      if (self$num_candidates_per_step < 1L) {
        abort_invalid_config("num_candidates_per_step must be >= 1")
      }
      if (self$cache_size < 1L) abort_invalid_config("cache_size must be >= 1")
      if (self$early_stop_threshold < 1L) {
        abort_invalid_config("early_stop_threshold must be >= 1")
      }
      if (self$n_workers < 1L) abort_invalid_config("n_workers must be >= 1")
      if (self$alpha < 0 || self$alpha > 1) {
        abort_invalid_config("alpha must be in [0, 1]")
      }
      if (self$alpha_min < 0 || self$alpha_min > 1) {
        abort_invalid_config("alpha_min must be in [0, 1]")
      }
      if (self$alpha_max < 0 || self$alpha_max > 1) {
        abort_invalid_config("alpha_max must be in [0, 1]")
      }
      if (self$alpha_min > self$alpha_max) {
        abort_invalid_config("alpha_min must be <= alpha_max")
      }
      if (!self$direction %in% c("minimize", "maximize")) {
        abort_invalid_config("direction must be 'minimize' or 'maximize'")
      }
      invisible(self)
    }
  )
)

#' Build a GIVP configuration object
#' @param ... Fields to override in `GIVPConfig`.
#' @return A `GIVPConfig` R6 object.
#' @export
givp_config <- function(...) {
  GIVPConfig$new(...)
}
