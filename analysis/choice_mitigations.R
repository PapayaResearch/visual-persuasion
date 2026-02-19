source("utils.R")


################################################################################
# MAIN EXECUTION
################################################################################

d <- read.csv("data/mitigation_results.csv") %>%
  mutate(
    strategy = "competition",
    k = evaluate.iterations %>% as.character() %>%
      recode("0" = "k=0", "1" = "k=1", "3" = "k=3"),
  ) %>%
  select(-evaluate.iterations) %>%
  clean_choice_data(recode_model = TRUE, include_strategy = TRUE) %>%
  reshape_to_long() %>%
  group_by(task.name, pair_id) %>%
  mutate(is_err = any(is.na(chosen))) %>%
  ungroup() %>%
  mutate(
    chosen = ifelse(is_err, 0, chosen)
  ) %>%
  bind_rows(
    # Create the 3rd "inconsistent" row for EVERY pair
    distinct(., pair_id, task.name, model, k, is_err) %>%
      mutate(
        type = "inconsistent",
        chosen = ifelse(is_err, 1, 0) # 1 if error occurred, 0 if a valid choice happened
      )
  ) %>%
  select(-is_err)

analyze_task.simplified <- function(
  name,
  d
) {
  message("Analyzing: ", name)

  d_prepped <- d %>%
    mutate(
      type = factor(type, levels = c("original", "final", "inconsistent"), labels = c("Original", "Final", "Inconsistent")),
      model = factor(model)
    )

  model_fit <- feols(
    chosen ~ type * model * k,
    vcov = ~ pair_id,
    data = d_prepped
  )
  emm <- emmeans(model_fit, ~ type | k, data = d_prepped)

  emm.by_model <- emmeans(model_fit, ~ type | model | k, data = d_prepped)

  list(
    data = d_prepped,
    model = model_fit,
    emmeans = emm %>% as.data.frame(),
    emmeans_by_model = emm.by_model %>% as.data.frame(),
    contrasts = compute_contrasts(emm),
    contrasts_by_model = compute_contrasts(emm.by_model)
  )
}

d.hotels <- d %>% filter(task.name == "hotels")
d.houses <- d %>% filter(task.name == "houses")
d.people <- d %>% filter(task.name == "people")
d.products <- d %>% filter(task.name == "products")

results_hotels <- analyze_task.simplified("hotels", d.hotels)
results_houses <- analyze_task.simplified("houses", d.houses)
results_people <- analyze_task.simplified("people", d.people)
results_products <- analyze_task.simplified("products", d.products)


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
  ggplot(aes(x = type, y = emmean, fill = k)) +
    geom_col(
      position = position_dodge(width = 0.8),
      alpha = 0.5,
      width = 0.8,
      color = NA
    ) +
    geom_errorbar(
      aes(ymin = lower.CL, ymax = upper.CL),
      position = position_dodge(width = 0.8),
      width = 0.2
    ) +
    xlab("Type") +
    ylab("P(Choice)") +
    scale_y_continuous(
      limits = c(0, NA),
      breaks = seq(0.25, 1, by = 0.25),
      expand = expansion(mult = c(0, 0.1)),
      labels = scales::percent_format(accuracy = 1)
    ) +
    scale_fill_manual(
      name = "Num. Passes",
      values = c("k=0" = "black", "k=1" = "#26B4F9", "k=3" = "#0072B2"),
      labels = c("k=0" = "No Mitigation", "k=1" = "1 Pass", "k=3" = "3 Passes")
    ) +
    facet_wrap(~ task) +
    theme_custom() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "bottom"
    )

save_plot(plot_combined.solutions, "combined_by_task_strategy-solutions.pdf", width = 4.8, height = 4)

