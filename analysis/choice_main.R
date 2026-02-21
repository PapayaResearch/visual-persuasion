source("utils.R")


################################################################################
# MAIN EXECUTION
################################################################################

d <- read_combined_file("data/combined_results.csv")

results_hotels <- analyze_task("hotels", d$hotels, has_image_class = TRUE)
results_houses <- analyze_task("houses", d$houses, has_image_class = FALSE)
results_people <- analyze_task("people", d$people, has_image_class = FALSE)
results_products <- analyze_task("products", d$products, has_image_class = TRUE)


################################################################################
# SAVE INDIVIDUAL TASK PLOTS
################################################################################

save_plot(results_hotels$plots$by_model_strategy, "hotels_by_model_strategy.pdf", width = 12, height = 6)
save_plot(results_hotels$plots$by_strategy, "hotels_by_strategy.pdf", width = 9, height = 4)
save_plot(results_hotels$plots$by_image_strategy, "hotels_by_image_strategy.pdf", width = 6, height = 5.4)

save_plot(results_houses$plots$by_model_strategy, "houses_by_model_strategy.pdf", width = 12, height = 6)
save_plot(results_houses$plots$by_strategy, "houses_by_strategy.pdf", width = 9, height = 4)

save_plot(results_people$plots$by_model_strategy, "people_by_model_strategy.pdf", width = 12, height = 6)
save_plot(results_people$plots$by_strategy, "people_by_strategy.pdf", width = 9, height = 4)

save_plot(results_products$plots$by_model_strategy, "products_by_model_strategy.pdf", width = 12, height = 6)
save_plot(results_products$plots$by_strategy, "products_by_strategy.pdf", width = 9, height = 4)
save_plot(results_products$plots$by_image_strategy, "products_by_image_strategy.pdf", width = 18, height = 6)


################################################################################
# COMBINED ACROSS TASKS
################################################################################

emmeans_combined <- rbind(
  results_hotels$emmeans$overall %>% mutate(task = "Hotels"),
  results_houses$emmeans$overall %>% mutate(task = "Houses"),
  results_people$emmeans$overall %>% mutate(task = "People"),
  results_products$emmeans$overall %>% mutate(task = "Products")
)

contrasts_combined <- rbind(
  results_hotels$contrasts$overall %>% mutate(task = "Hotels"),
  results_houses$contrasts$overall %>% mutate(task = "Houses"),
  results_people$contrasts$overall %>% mutate(task = "People"),
  results_products$contrasts$overall %>% mutate(task = "Products")
)

plot_combined <- plot_emm_faceted(
  emmeans_combined,
  contrasts_combined,
  c("task", "strategy"),
  strategy ~ task,
  ""
)
 save_plot(plot_combined, "combined_by_task_strategy.pdf", width = 4, height = 4)


emmeans_bymodel <- rbind(
  results_hotels$emmeans$by_model %>% mutate(task = "Hotels"),
  results_houses$emmeans$by_model %>% mutate(task = "Houses"),
  results_people$emmeans$by_model %>% mutate(task = "People"),
  results_products$emmeans$by_model %>% mutate(task = "Products")
)

################################################################################
# SUMMARY TABLES
################################################################################

d_all <- bind_rows(d$hotels, d$houses, d$people, d$products)

d_all_prepped <- prep_for_analysis(d_all, has_image_class = FALSE, has_model = TRUE) %>%
  mutate(
    task.name = factor(task.name),
    strategy = factor(strategy, levels = STRATEGY_LEVELS)
  )

model_all <- feols(
  chosen ~ type * model * strategy * task.name,
  vcov = ~ pair_id,
  data = d_all_prepped
)

emm_all <- emmeans(model_all, ~ type | model | strategy) %>% as_emm_tbl()

contrast_all <- emmeans(model_all, ~ type | model | strategy) %>%
  compute_contrasts(adjust = "none") %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

table_choice_by_model <- build_delta_table(
  emm_tbl = emm_all,
  contrast_tbl = contrast_all,
  group_vars = c("model", "strategy"),
  grouping_var = "type",
  caption = "Model-wise choice probabilities by strategy and type, pooled across tasks. Main value is the estimated marginal mean probability; parentheses show $\\Delta$ vs. the \\colorbox[HTML]{c4dd88}{best type} within each strategy for that model.",
  label = "choice_combined_by_model",
  column_vars = c("type"),
  row_vars = c("model", "strategy"),
  grouping_levels = TYPE_LABELS
)

save_table(table_choice_by_model, "choice_combined_by_model.tex")

table_final_by_task <- emmeans_bymodel %>%
  subset(type == "Final") %>%
  select(model, task, strategy, emmean) %>%
  mutate(emmean = round(emmean * 100, 1)) %>%
  group_by(model, task) %>%
  mutate(emmean = ifelse(emmean == max(emmean), paste0(emmean, "*"), emmean)) %>%
  pivot_wider(names_from = c(task, strategy), values_from = emmean) %>%
  knitr::kable(
    "latex",
    digits = 1,
    caption = "P(Choice) for Final Type by Model and Task",
    align = "lcccccccccccc",
    booktabs = TRUE
  ) %>%
  kableExtra::kable_styling(full_width = FALSE) %>%
  kableExtra::column_spec(1, bold = TRUE) %>%
  kableExtra::add_header_above(c(" " = 1, "Hotels" = 3, "Houses" = 3, "People" = 3, "Products" = 3))

save_table(table_final_by_task, "final_choice_by_model_task.tex")
