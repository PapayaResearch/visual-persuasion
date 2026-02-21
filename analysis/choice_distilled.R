source("utils.R")


################################################################################
# MAIN EXECUTION
################################################################################

d <- read_combined_file("data/distillation_results.csv")

analyze_task.simplified <- function(
  name,
  d,
  has_image_class
) {
  message("Analyzing: ", name)

  d_prepped <- prep_for_analysis(d, has_image_class, has_model = TRUE) %>%
    droplevels()

  model_fit <- feols(
    chosen ~ type * model,
    vcov = ~ pair_id,
    data = d_prepped
  )
  emm <- emmeans(model_fit, ~ type, data = d_prepped)

  emm.by_model <- emmeans(model_fit, ~ type | model, data = d_prepped)

  list(
    data = d_prepped,
    model = model_fit,
    emmeans = emm %>% as.data.frame(),
    emmeans_by_model = emm.by_model %>% as.data.frame(),
    contrasts = compute_contrasts(emm),
    contrasts_by_model = compute_contrasts(emm.by_model)
  )
}

results_hotels <- analyze_task.simplified("hotels", d$hotels %>% droplevels(), has_image_class = TRUE)
results_houses <- analyze_task.simplified("houses", d$houses %>% droplevels(), has_image_class = FALSE)
results_people <- analyze_task.simplified("people", d$people %>% droplevels(), has_image_class = FALSE)
results_products <- analyze_task.simplified("products", d$products %>% droplevels(), has_image_class = TRUE)

################################################################################
# COMBINED ACROSS TASKS
################################################################################

emmeans_combined <- rbind(
  results_hotels$emmeans %>% mutate(task = "Hotels"),
  results_houses$emmeans %>% mutate(task = "Houses"),
  results_people$emmeans %>% mutate(task = "People"),
  results_products$emmeans %>% mutate(task = "Products")
)

contrasts_combined <- rbind(
  results_hotels$contrasts %>% mutate(task = "Hotels"),
  results_houses$contrasts %>% mutate(task = "Houses"),
  results_people$contrasts %>% mutate(task = "People"),
  results_products$contrasts %>% mutate(task = "Products")
)

plot_combined.solutions <- emmeans_combined %>%
  ggplot(aes(x = type, y = emmean)) +
    geom_col(fill = "steelblue", alpha = 0.8, width = 0.8) +
    geom_errorbar(
      aes(ymin = lower.CL, ymax = upper.CL),
      width = 0.2,
      color = "black"
    ) +
    stat_pvalue_manual(
      build_contrast_df(
        emmeans_combined,
        contrasts_combined,
        group_vars = "task",
        grouping_var = "type"
      ) %>%
        filter(label != ""),
      label = "label",
      x1 = "group1",
      x2 = "group2",
      y.position = "y",
      tip.length = 0.02,
      step.increase = 0.06,
      step.group.by = "task",
      size = 4,
      label.size = 4
    ) +
    xlab("") +
    ylab("P(Choice)") +
    scale_y_continuous(
      limits = c(0, NA),
      breaks = seq(0.25, 1, by = 0.25),
      expand = expansion(mult = c(0, 0.1)),
      labels = scales::percent_format(accuracy = 1)
    ) +
    facet_wrap(~ task) +
    theme_custom() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "bottom"
    )

save_plot(plot_combined.solutions, "combined_by_task_strategy-distillation.pdf", width = 4, height = 4)
