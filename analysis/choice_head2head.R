source("utils.R")


################################################################################
# DATA LOADING & PREPARATION
################################################################################

d <- read.csv("data/head2head_results.csv") %>%
  subset(choice != "inconsistent") %>%
  mutate(
    model = recode(model, !!!MODEL_MAPPING),
    strategy1 = recode(strategy1, !!!STRATEGY_MAPPING),
    strategy2 = recode(strategy2, !!!STRATEGY_MAPPING),
    chosen = ifelse(choice == "first", strategy1, strategy2)
  )


d_choice <- d %>%
  mutate(pair_id = row_number()) %>%
  pivot_longer(
    cols = c(strategy1, strategy2),
    names_to = "slot",
    values_to = "strategy"
  ) %>%
  mutate(
    chosen_flag = as.integer(strategy == chosen),
    strategy = factor(strategy, levels = STRATEGY_LEVELS)
  )


################################################################################
# MODELING
################################################################################

model_choice <- feols(
  chosen_flag ~ strategy * model | task,
  vcov = ~ pair_id,
  data = d_choice
)

model_choice.by_task <- feols(
  chosen_flag ~ strategy * model * task,
  vcov = ~ pair_id,
  data = d_choice
)


################################################################################
# EMMEANS & CONTRASTS
################################################################################

emm_overall <- emmeans(
  model_choice,
  ~ strategy
) %>% as.data.frame()

contrast_overall <- emmeans(model_choice, ~ strategy) %>%
  contrast("pairwise", adjust = "BH")

emm_bymodel <- emmeans(model_choice, ~ strategy | model) %>% as.data.frame()

contrast_bymodel <- emmeans(model_choice, ~ strategy | model) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

model_order <- emm_bymodel %>%
  filter(strategy == "CVPO") %>%
  arrange(desc(emmean)) %>%
  pull(model)

emm_bymodel <- emm_bymodel %>%
  mutate(model = factor(model, levels = model_order))

contrast_bymodel <- contrast_bymodel %>%
  mutate(model = factor(model, levels = model_order))

emm_bytask <- emmeans(
  model_choice.by_task,
  ~ strategy | task
) %>%
  as.data.frame()

################################################################################
# PLOTS
################################################################################

plot_overall <- emm_overall %>%
  ggplot(aes(x = reorder(strategy, emmean), y = emmean)) +
  geom_col(fill = "steelblue", width = 0.8, alpha = 0.8) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.1,
    color = "black"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0, 0.05)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  stat_pvalue_manual(
    build_contrast_df(
      emm_overall,
      contrast_overall %>% as.data.frame(),
      group_vars = NULL,
      grouping_var = "strategy"
    ) %>%
      filter(label != ""),
    label = "label",
    x1 = "group1",
    x2 = "group2",
    y.position = "y",
    tip.length = 0.02,
    step.increase = 0.06,
    size = 4,
    label.size = 4
  ) +
  xlab("Strategy") +
  ylab("P(Choice)") +
  theme_custom() +
  theme(legend.position = "none")

save_plot(plot_overall, "head2head_overall.pdf", width = 4, height = 2.4)

plot_bymodel <- emm_bymodel %>%
  mutate(strategy = factor(strategy, levels = STRATEGY_LEVELS)) %>%
  ggplot(aes(x = reorder(model, ifelse(strategy == "CVPO", emmean, 0)), y = emmean, fill = strategy)) +
  geom_col(position = position_dodge(width = 0.9), width = 0.8) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.2,
    position = position_dodge(width = 0.9),
    color = "black"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_fill_cosmic() +
  xlab("") +
  ylab("P(Choice)") +
  guides(fill = guide_legend(title = "Strategy")) +
  theme_custom() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

save_plot(plot_bymodel, "head2head_by_model.pdf", width = 8, height = 4)

plot_bytask <- emm_bytask %>%
  mutate(strategy = factor(strategy, levels = STRATEGY_LEVELS)) %>%
  ggplot(aes(x = strategy, y = emmean, fill = strategy)) +
  geom_col(position = position_dodge(width = 0.9), width = 0.8) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.2,
    position = position_dodge(width = 0.9),
    color = "black"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_fill_cosmic() +
  xlab("Strategy") +
  ylab("P(Choice)") +
  guides(fill = guide_legend(title = "Strategy")) +
  facet_wrap(~ str_to_title(task), ncol = 2) +
  theme_custom() +
  theme(legend.position = "none")

save_plot(plot_bytask, "head2head_by_task.pdf", width = 4, height = 4)

################################################################################
# TABLES
################################################################################

table_contrasts <- build_delta_table(
  emm_tbl = emm_bymodel %>% arrange(model) %>% rename(VLM = model, Strategy = strategy),
  contrast_tbl = contrast_bymodel %>% arrange(model) %>% rename(VLM = model),
  group_vars = "VLM",
  grouping_var = "Strategy",
  caption = "Model-wise choice probabilities by strategy (main value) with contrasts vs. the \\colorbox[HTML]{c4dd88}{row-best strategy} in parentheses. Asterisks indicate Benjamini-Hochberg adjusted significance ($^{****}=p<.0001$, $^{***}=p<.001$, $^{**}=p<.01$, $^{*}=p<.05$).",
  label = "choice_head2head_contrasts_bymodel",
  column_order = STRATEGY_LEVELS,
  grouping_levels = STRATEGY_LEVELS
) %>%
  kableExtra::add_header_above(
    c(" " = 1, "Strategy (P(Choice) with $\\\\Delta$ vs. best)" = 3),
    escape = FALSE
  )

save_table(table_contrasts, "head2head_contrasts_by_model.tex")
