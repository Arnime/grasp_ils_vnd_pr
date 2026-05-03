# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Build convergence monitor
#' @keywords internal
make_convergence_monitor <- function() {
    structure(
        list(no_improve = 0L, last_best = NULL),
        class = "givp_convergence_monitor"
    )
}

#' Update convergence monitor
#' @keywords internal
convergence_update <- function(monitor, best_value, direction) {
    if (is.null(monitor$last_best)) {
        monitor$last_best <- best_value
        monitor$no_improve <- 0L
        return(monitor)
    }

    improved <- is_improvement( # nolint: object_usage_linter
        best_value,
        monitor$last_best,
        direction
    )
    if (improved) {
        monitor$no_improve <- 0L
        monitor$last_best <- best_value
    } else {
        monitor$no_improve <- monitor$no_improve + 1L
    }
    monitor
}
