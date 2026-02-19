library(tidyverse)
library(fixest)
library(emmeans)
library(ggpubr)
library(ggsci)


################################################################################
# DATA LOADING
################################################################################

read_combined_file <- function(filename) {
  d <- read.csv(filename) %>%
    select(
      model,
      image_class,
      base1,
      base2,
      choice,
      task.name,
      strategy = evaluate.strategy_name
    ) %>%
    clean_choice_data(recode_model = TRUE, include_strategy = TRUE) %>%
    select(model, image_class, pair_id, chosen_base, base1_id, base1_type, base2_id, base2_type, task.name, strategy) %>%
    reshape_to_long()

  list(
    hotels = d %>% subset(task.name == "hotels"),
    houses = d %>% subset(task.name == "houses"),
    people = d %>% subset(task.name == "people"),
    products = d %>% subset(task.name == "products")
  )
}


################################################################################
# ANALYSIS-SPECIFIC FUNCTIONS
################################################################################

plot_emm_faceted <- function(
  emm_tbl,
  contrasts_tbl = NULL,
  group_vars = NULL,
  facet_formula = NULL,
  title = NULL
) {
  p <- ggplot(emm_tbl, aes(x = type, y = emmean)) +
    geom_pointrange(
      aes(ymin = lower.CL, ymax = upper.CL),
      size = 0.6,
      fatten = 2
    ) +
    geom_line(aes(group = 1)) +
    xlab("Type") +
    ylab("P(Choice)") +
    scale_y_continuous(
      limits = c(0, NA),
      breaks = seq(0.25, 1, by = 0.25),
      expand = expansion(mult = c(0, 0.05)),
      labels = scales::percent_format(accuracy = 1)
    ) +
    theme_custom() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  if (!is.null(contrasts_tbl)) {
    contrast_df <- build_contrast_df(emm_tbl, contrasts_tbl, group_vars)
    p <- p + add_significance_brackets(p, contrast_df)
  }

  if (!is.null(facet_formula)) {
    p <- p + facet_grid(facet_formula)
  }

  if (!is.null(title)) {
    p <- p + ggtitle(title)
  }

  p
}

compute_emmeans_full <- function(
  model_fit,
  d,
  has_image_class
) {
  list(
    by_model_image = if (has_image_class) {
      emmeans(model_fit, ~ type | model | image_class | strategy, data = d)
    } else {
      NULL
    },
    by_model = emmeans(model_fit, ~ type | model | strategy, data = d),
    by_image = if (has_image_class) {
      emmeans(model_fit, ~ type | image_class | strategy, data = d)
    } else {
      NULL
    },
    overall = emmeans(model_fit, ~ type | strategy, data = d)
  )
}

analyze_task <- function(
  name,
  d,
  has_image_class,
  has_strategy = TRUE
) {
  message("Analyzing: ", name)

  d_prepped <- prep_for_analysis(d, has_image_class, has_model = TRUE)

  model_fit <- fit_choice_model(
    d_prepped,
    has_image_class,
    has_model = TRUE,
    has_strategy = has_strategy,
    response = "chosen",
    vcov = ~ pair_id
  )
  emmeans_list <- compute_emmeans_full(model_fit, d_prepped, has_image_class)

  emmeans_tbls <- list(
    by_model_image = if (has_image_class) as_emm_tbl(emmeans_list$by_model_image) else NULL,
    by_model = as_emm_tbl(emmeans_list$by_model),
    by_image = if (has_image_class) as_emm_tbl(emmeans_list$by_image) else NULL,
    overall = as_emm_tbl(emmeans_list$overall)
  )

  contrasts_tbls <- list(
    by_model_image = if (has_image_class) compute_contrasts(emmeans_list$by_model_image) else NULL,
    by_model = compute_contrasts(emmeans_list$by_model),
    by_image = if (has_image_class) compute_contrasts(emmeans_list$by_image) else NULL,
    overall = compute_contrasts(emmeans_list$overall)
  )

  plots <- list(
    by_model_strategy = plot_emm_faceted(
      emmeans_tbls$by_model,
      contrasts_tbls$by_model,
      c("model", "strategy"),
      strategy ~ model,
      paste(name %>% str_to_title(), "- by Model x Strategy")
    ),
    by_strategy = plot_emm_faceted(
      emmeans_tbls$overall,
      contrasts_tbls$overall,
      "strategy",
      . ~ strategy,
      paste(name %>% str_to_title(), "- by Strategy")
    )
  )

  if (has_image_class) {
    plots$by_image_strategy <- plot_emm_faceted(
      emmeans_tbls$by_image,
      contrasts_tbls$by_image,
      c("image_class", "strategy"),
      strategy ~ image_class,
      paste(name %>% str_to_title(), "- by Image Class x Strategy")
    )
  }

  list(
    data = d_prepped,
    model = model_fit,
    emmeans = emmeans_tbls,
    contrasts = contrasts_tbls,
    plots = plots
  )
}


################################################################################
# CONSTANTS
################################################################################

MODEL_MAPPING <- list(
  "gemini-2.5-flash" = "Gemini 2.5 Flash",
  "gemini-2.5-pro" = "Gemini 2.5 Pro",
  "gemini-3-flash-preview" = "Gemini 3 Flash",
  "gemini-3-pro-preview" = "Gemini 3 Pro",
  "gpt-4o-2024-08-06" = "GPT-4o",
  "gpt-4.1-nano" = "GPT-4.1 Nano",
  "gpt-5-mini-2025-08-07" = "GPT-5 Mini",
  "gpt-5-2025-08-07" = "GPT-5",
  "gpt-5.2-2025-12-11" = "GPT-5.2",
  "claude-haiku-4-5-20251001" = "Claude Haiku 4.5",
  "claude-sonnet-4-5-20250929" = "Claude Sonnet 4.5",
  "qwen.qwen3-vl-235b-a22b" = "Qwen-VL 235B",
  "us.meta.llama4-maverick-17b-instruct-v1:0" = "Llama 4 Maverick"
)

STRATEGY_MAPPING <- list(
  "competition" = "CVPO",
  "textgrad" = "VTG",
  "feedback_descent" = "VFD",
  "feedback-descent" = "VFD"
)

STRATEGY_LEVELS = c("VTG", "VFD", "CVPO")

TYPE_LEVELS <- c("original", "zero-shot", "final", "distillation")
TYPE_LABELS <- c("Original", "Zero-shot", "Final", "Distilled")


################################################################################
# DATA PROCESSING
################################################################################

parse_base_components <- function(base_col) {
  parsed <- str_match(base_col, "^(.*)_(original|zero-shot|final|distillation)$")
  list(
    id = parsed[, 2],
    type = parsed[, 3]
  )
}

clean_choice_data <- function(
  d,
  recode_model = TRUE,
  include_strategy = FALSE
) {
  d %>%
    mutate(
      choice_clean = ifelse(choice == "inconsistent", NA_character_, choice),
      chosen_base = case_when(
        choice_clean == base1 ~ "base1",
        choice_clean == base2 ~ "base2",
        TRUE ~ NA_character_
      ),
      base1_id = parse_base_components(base1)$id,
      base1_type = parse_base_components(base1)$type,
      base2_id = parse_base_components(base2)$id,
      base2_type = parse_base_components(base2)$type,
      pair_id = row_number(),
      image_class = image_class %>% str_to_title()
    ) %>%
    {
      if (recode_model) mutate(., model = recode(model, !!!MODEL_MAPPING)) else .
    } %>%
    {
      if (include_strategy) mutate(., strategy = recode(strategy, !!!STRATEGY_MAPPING)) else .
    }
}

reshape_to_long <- function(d) {
  d %>%
    pivot_longer(
      cols = c(base1_id, base1_type, base2_id, base2_type),
      names_to = c("base", ".value"),
      names_pattern = "base(1|2)_(id|type)"
    ) %>%
    rename(item_id = id) %>%
    mutate(
      chosen = case_when(
        is.na(chosen_base) ~ NA_integer_,
        base == "1" & chosen_base == "base1" ~ 1L,
        base == "2" & chosen_base == "base2" ~ 1L,
        TRUE ~ 0L
      )
    )
}

prep_for_analysis <- function(
  d,
  has_image_class = TRUE,
  has_model = TRUE
) {
  d %>%
    filter(!is.na(chosen)) %>%
    mutate(type = factor(type, levels = TYPE_LEVELS, labels = TYPE_LABELS)) %>%
    {
      if (has_model) mutate(., model = factor(model)) else .
    } %>%
    {
      if (has_image_class) {
        mutate(., image_class = factor(image_class)) %>% filter(!is.na(image_class))
      } else {
        .
      }
    }
}


################################################################################
# MODELING
################################################################################

build_model_formula <- function(
  has_image_class = TRUE,
  has_model = TRUE,
  has_strategy = FALSE,
  response = "chosen"
) {
  terms <- c()
  if (has_strategy) terms <- c(terms, "strategy")
  terms <- c(terms, "type")
  if (has_model) terms <- c(terms, "model")
  if (has_image_class) terms <- c(terms, "image_class")

  formula_str <- sprintf("%s ~ %s", response, paste(terms, collapse = " * "))
  as.formula(formula_str)
}

fit_choice_model <- function(
  d,
  has_image_class = TRUE,
  has_model = TRUE,
  has_strategy = FALSE,
  response = "chosen",
  vcov = ~ item_id
) {
  model_formula <- build_model_formula(has_image_class, has_model, has_strategy, response)
  feols(model_formula, vcov = vcov, data = d)
}


################################################################################
# EMMEANS & CONTRASTS
################################################################################

as_emm_tbl <- function(emm) {
  as_tibble(emm) %>%
    mutate(type = factor(type, levels = TYPE_LABELS))
}

compute_contrasts <- function(emm, adjust = "BH") {
  contrast(emm, method = "revpairwise", adjust = adjust) %>% as_tibble()
}

build_delta_table <- function(
  emm_tbl,
  contrast_tbl,
  group_vars,
  grouping_var,
  caption,
  label,
  column_vars = NULL,
  column_order = NULL,
  names_sep = "__",
  row_vars = NULL,
  grouping_levels = NULL
) {
  if (is.null(row_vars)) row_vars <- group_vars
  contrast_levels <- if (!is.null(grouping_levels)) {
    grouping_levels
  } else {
    levels(factor(emm_tbl[[grouping_var]]))
  }

  if (!is.null(column_vars)) {
    column_vars <- column_vars[column_vars %in% names(emm_tbl)]
  }
  if (!is.null(column_order)) {
    column_order <- column_order[column_order %in% names(emm_tbl)]
  }

  column_count <- if (!is.null(column_order)) length(column_order) else length(contrast_levels)

  contrast_tbl <- contrast_tbl %>%
    mutate(contrast = str_replace_all(contrast, "[()]", "")) %>%
    separate(contrast, into = c("group2", "group1"), sep = " - ") %>%
    mutate(
      group1 = factor(group1, levels = contrast_levels),
      group2 = factor(group2, levels = contrast_levels),
      label = case_when(
        p.value < 0.0001 ~ "****",
        p.value < 0.001 ~ "***",
        p.value < 0.01 ~ "**",
        p.value < 0.05 ~ "*",
        TRUE ~ ""
      )
    )

  best_by_group <- emm_tbl %>%
    group_by(across(all_of(group_vars))) %>%
    slice_max(emmean, n = 1, with_ties = FALSE) %>%
    select(all_of(group_vars), best_level = all_of(grouping_var), best_emmean = emmean)

  contrast_lookup <- contrast_tbl %>%
    select(all_of(group_vars), group1, group2, estimate, label)

  pivot_vars <- if (is.null(column_vars)) grouping_var else column_vars

  emm_tbl %>%
    left_join(best_by_group, by = group_vars) %>%
    mutate(is_best = .data[[grouping_var]] == best_level) %>%
    left_join(
      contrast_lookup %>%
        transmute(
          across(all_of(group_vars)),
          !!grouping_var := group2,
          best_level = group1,
          delta = estimate,
          label
        ),
      by = c(group_vars, grouping_var, "best_level")
    ) %>%
    left_join(
      contrast_lookup %>%
        transmute(
          across(all_of(group_vars)),
          !!grouping_var := group1,
          best_level = group2,
          delta = -estimate,
          label
        ),
      by = c(group_vars, grouping_var, "best_level"),
      suffix = c("", ".rev")
    ) %>%
    mutate(
      delta = coalesce(delta, delta.rev, ifelse(is_best, 0, emmean - best_emmean)),
      label = coalesce(label, label.rev, ""),
      main_str = ifelse(is_best, sprintf("\\textbf{%.3f}", emmean), sprintf("%.3f", emmean)),
      delta_str = ifelse(
        label != "",
        sprintf("%.3f^{%s}", delta, label),
        sprintf("%.3f", delta)
      ),
      cell_str = ifelse(
        is_best,
        main_str,
        sprintf("%s {\\scriptsize ($\\Delta$=$%s$)}", main_str, delta_str)
      ),
      cell_str = kableExtra::cell_spec(
        cell_str,
        format = "latex",
        escape = FALSE,
        background = ifelse(is_best, "#c4dd88", "#ffffff")
      )
    ) %>%
    select(all_of(row_vars), all_of(pivot_vars), cell_str) %>%
    pivot_wider(
      names_from = all_of(pivot_vars),
      values_from = cell_str,
      names_sep = names_sep
    ) %>%
    {
      if (!is.null(column_order)) {
        select(., all_of(row_vars), all_of(column_order))
      } else {
        .
      }
    } %>%
    knitr::kable(
      "latex",
      caption = caption,
      align = paste0(
        paste(rep("l", length(row_vars)), collapse = ""),
        paste(rep("l", column_count), collapse = "")
      ),
      booktabs = TRUE,
      label = label,
      table.envir = "table*",
      linesep = "",
      escape = FALSE
    ) %>%
    kableExtra::kable_styling(full_width = FALSE) %>%
    kableExtra::column_spec(1, bold = TRUE)
}

build_contrast_df <- function(
  emm_tbl,
  contrasts_tbl,
  group_vars = character(0),
  grouping_var = "type",
  p_adjust = "BH",
  dodge = 0.03,
  dodge_var = NULL,
  dodge_width = 0.6,
  dodge_n = NULL
) {
  # Helper: stable levels for grouping_var (x-axis categories)
  contrast_levels <- if (is.factor(emm_tbl[[grouping_var]])) {
    levels(emm_tbl[[grouping_var]])
  } else {
    unique(emm_tbl[[grouping_var]])
  }

  # p-adjust
  contrasts_tbl$p.value <- p.adjust(contrasts_tbl$p.value, method = p_adjust)

  # Parse contrast labels -> group1/group2, compute integer x positions
  out <- contrasts_tbl %>%
    dplyr::mutate(contrast = stringr::str_replace_all(.data$contrast, "[()]", "")) %>%
    tidyr::separate(.data$contrast, into = c("group2", "group1"), sep = " - ", remove = FALSE) %>%
    dplyr::mutate(
      group1 = factor(.data$group1, levels = contrast_levels),
      group2 = factor(.data$group2, levels = contrast_levels),
      x1 = as.integer(.data$group1),
      x2 = as.integer(.data$group2),
      label = dplyr::case_when(
        .data$p.value < 0.0001 ~ "****",
        .data$p.value < 0.001 ~ "***",
        .data$p.value < 0.01 ~ "**",
        .data$p.value < 0.05 ~ "*",
        TRUE ~ ""
      )
    )

  # Determine max upper.CL per facet/group (same as your original intent)
  max_tbl <- if (length(group_vars) > 0) {
    emm_tbl %>%
      dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
      dplyr::summarize(max_upper = max(.data$upper.CL, na.rm = TRUE), .groups = "drop")
  } else {
    tibble::tibble(max_upper = max(emm_tbl$upper.CL, na.rm = TRUE))
  }

  # y positions (stack within each group_vars bucket)
  out <- if (length(group_vars) > 0) {
    out %>%
      dplyr::left_join(max_tbl, by = group_vars) %>%
      dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
      dplyr::mutate(y = .data$max_upper + 0.03 * dplyr::row_number()) %>%
      dplyr::ungroup()
  } else {
    out %>%
      dplyr::mutate(y = max_tbl$max_upper + 0.03 * dplyr::row_number())
  }

  # Automatic x dodging (optional)
  if (!is.null(dodge_var)) {
    if (!(dodge_var %in% names(out))) {
      stop("build_contrast_df: dodge_var '", dodge_var, "' not found in contrasts_tbl after joins/parsing.")
    }

    dv <- out[[dodge_var]]

    # Determine levels and n used for dodging
    dv_levels <- if (is.factor(dv)) levels(dv) else unique(dv)
    n <- if (!is.null(dodge_n)) dodge_n else length(dv_levels)
    if (n <= 0) stop("build_contrast_df: dodge_n must be > 0.")

    # Centered offsets, matching ggplot's position_dodge behavior conceptually
    offsets <- (seq_len(n) - (n + 1) / 2) * (dodge_width / n)

    out <- out %>%
      dplyr::mutate(
        .dv_idx = match(.data[[dodge_var]], dv_levels),
        .dv_off = offsets[.data$.dv_idx],
        xmin = .data$x1 + .data$.dv_off,
        xmax = .data$x2 + .data$.dv_off
      ) %>%
      dplyr::select(-.data$.dv_idx, -.data$.dv_off)
  }

  out
}


################################################################################
# PLOTTING
################################################################################

theme_custom <- function() {
  theme_minimal() +
    theme(
      axis.line.y.left = element_line(color = "black"),
      axis.line.x.bottom = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.title = element_text(face = "bold"),
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold")
    )
}

add_significance_brackets <- function(
  p,
  contrasts_tbl,
  scale.factor = 3
) {
  mu <- mean(contrasts_tbl$y)
  sigma <- sd(contrasts_tbl$y)
  stat_pvalue_manual(
    contrasts_tbl %>%
      filter(label != "") %>%
      mutate(
        y = mu + ((scale.factor * ((y - mu) / sigma)) * sigma)
      ),
    label = "label",
    xmin = "x1",
    xmax = "x2",
    y.position = "y",
    bracket.nudge.y = 0.2,
    tip.length = 0.01,
    size = 4
  )
}


################################################################################
# OUTPUT HELPERS
################################################################################

save_plot <- function(
  plot,
  filename,
  width = 8,
  height = 6,
  device = "pdf",
  dir = "plots"
) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  filepath <- file.path(dir, filename)
  ggsave(
    filepath,
    plot = plot,
    width = width,
    height = height,
    device = device
  )
  message("Saved plot to: ", filepath)
  invisible(plot)
}

save_table <- function(
  kable_obj,
  filename,
  dir = "tables"
) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  filepath <- file.path(dir, filename)
  cat(kable_obj, file = filepath)
  message("Saved table to: ", filepath)
  invisible(kable_obj)
}
