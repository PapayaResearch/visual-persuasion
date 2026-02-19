source("utils.R")


################################################################################
# DATA LOADING & PREPARATION (CHOICE ANALYSIS)
################################################################################

d <- read.csv("data/combined_results-human.csv")
d <- d %>%
  mutate(
    chosen = ifelse(choice == 1, image1_status, image2_status) %>%
      factor(levels = TYPE_LEVELS, labels = TYPE_LABELS),
    strategy = recode(strategy, !!!STRATEGY_MAPPING) %>%
      factor(levels = STRATEGY_LEVELS)
  )


################################################################################
# MAIN EXECUTION (CHOICE ANALYSIS)
################################################################################

d_choice <- d %>%
  mutate(pair_id = row_number()) %>%
  pivot_longer(
    cols = c(image1_status, image2_status),
    names_to = "slot",
    values_to = "status"
  ) %>%
  mutate(
    status = factor(status, levels = TYPE_LEVELS, labels = TYPE_LABELS),
    chosen_flag = as.integer(status == chosen)
  ) %>% droplevels()

model_choice <- feols(
  chosen_flag ~ status * strategy * task,
  vcov = ~ participant_id + pair_id,
  data = d_choice
)

reg_choice <- etable(
  model_choice,
  tex = FALSE
) %>% kableExtra::kbl(
  format = "latex",
  booktabs = TRUE,
  align = "l"
)

save_table(reg_choice, "human_choice_regression.tex")

emm <- emmeans(
  model_choice,
  ~ status | strategy
) %>% as.data.frame()

contrast_overall <- emmeans(
  model_choice,
  ~ status | strategy
) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_choice_overall <- emm %>%
  ggplot(aes(x = status, y = emmean)) +
  geom_pointrange(
    aes(ymin = lower.CL, ymax = upper.CL),
    position = position_dodge(width = 0.4),
    size = 0.4
  ) +
  geom_line(
    aes(group = strategy),
    position = position_dodge(width = 0.4)
  ) +
  stat_pvalue_manual(
    build_contrast_df(emm, contrast_overall, group_vars = "strategy", grouping_var = "status") %>%
      filter(label != ""),
    label = "label",
    x1 = "group1",
    x2 = "group2",
    y.position = "y"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  xlab("Status") +
  ylab("P(Choice)") +
  facet_wrap(~ strategy) +
  guides(color = guide_legend(title = "Strategy")) +
  theme_custom()

save_plot(plot_choice_overall, "human_choice_by_strategy.pdf", width = 8, height = 4)

emm_by_task <- emmeans(
  model_choice,
  ~ status | strategy | task
) %>%
  as.data.frame()

contrast_by_task <- emmeans(
  model_choice,
  ~ status | strategy | task
) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_choice_by_task <- emm_by_task %>%
  ggplot(aes(x = status, y = emmean, color = strategy, group = strategy)) +
  geom_pointrange(
    aes(ymin = lower.CL, ymax = upper.CL),
    position = position_dodge(width = 0.4),
    size = 0.4
  ) +
  geom_line(position = position_dodge(width = 0.4)) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  xlab("Status") +
  ylab("P(Choice)") +
  facet_wrap(~ str_to_title(task)) +
  guides(color = guide_legend(title = "Strategy")) +
  theme_custom()

save_plot(plot_choice_by_task, "human_choice_by_strategy_task.pdf", width = 8, height = 4)

table_choice_overall <- build_delta_table(
  emm_tbl = emm %>% rename(Strategy = strategy),
  contrast_tbl = contrast_overall %>% rename(Strategy = strategy),
  group_vars = "Strategy",
  grouping_var = "status",
  caption = "Human choice probabilities by strategy and status. Main value is the estimated marginal mean probability; parentheses show $\\Delta$ vs. the \\colorbox[HTML]{c4dd88}{best status} within each strategy. Asterisks indicate Benjamini-Hochberg adjusted significance ($^{****}=p<.0001$, $^{***}=p<.001$, $^{**}=p<.01$, $^{*}=p<.05$).",
  label = "human_choice_by_strategy_status",
  column_order = TYPE_LABELS,
  grouping_levels = TYPE_LABELS
) %>%
  kableExtra::add_header_above(
    c(" " = 1, "Status (P(Choice) with $\\Delta$ vs. best)" = 3),
    escape = FALSE
  )

save_table(table_choice_overall, "human_choice_emm_by_strategy.tex")

table_choice_by_task <- build_delta_table(
  emm_tbl = emm_by_task %>% mutate(task = str_to_title(task)),
  contrast_tbl = contrast_by_task %>% mutate(task = str_to_title(task)),
  group_vars = c("task", "strategy"),
  grouping_var = "status",
  caption = "Human choice probabilities by task, strategy, and status. Main value is the estimated marginal mean probability; parentheses show $\\Delta$ vs. the \\colorbox[HTML]{c4dd88}{best status} within each task-strategy. Asterisks indicate Benjamini-Hochberg adjusted significance ($^{****}=p<.0001$, $^{***}=p<.001$, $^{**}=p<.01$, $^{*}=p<.05$).",
  label = "human_choice_by_task_strategy_status",
  column_order = TYPE_LABELS,
  grouping_levels = TYPE_LABELS
)

save_table(table_choice_by_task, "human_choice_emm_by_strategy_task.tex")


################################################################################
# DATA LOADING & PREPARATION (HEAD-TO-HEAD ANALYSIS)
################################################################################

d.h2h <- read.csv("data/head2head_results-human.csv")
d.h2h <- d.h2h %>%
  mutate(
    image1_strategy = recode(image1_strategy, !!!STRATEGY_MAPPING),
    image2_strategy = recode(image2_strategy, !!!STRATEGY_MAPPING),
    chosen = ifelse(choice == 1, image1_strategy, image2_strategy) %>%
      factor(levels = STRATEGY_LEVELS)
  )

d.h2h_choice <- d.h2h %>%
  mutate(pair_id = row_number()) %>%
  pivot_longer(
    cols = c(image1_strategy, image2_strategy),
    names_to = "slot",
    values_to = "strategy_raw"
  ) %>%
  mutate(
    strategy = recode(strategy_raw, !!!STRATEGY_MAPPING) %>%
      factor(levels = STRATEGY_LEVELS),
    chosen_flag = as.integer(strategy == chosen)
  )


################################################################################
# MAIN EXECUTION (HEAD-TO-HEAD ANALYSIS)
################################################################################

model_h2h_choice <- feols(
  chosen_flag ~ strategy * task,
  vcov = ~ participant_id + pair_id,
  data = d.h2h_choice
)

reg_h2h <- etable(
  model_h2h_choice,
  tex = FALSE
) %>% kableExtra::kbl(
  format = "latex",
  booktabs = TRUE,
  align = "l"
)
save_table(reg_h2h, "human_head2head_regression.tex")

emm_h2h <- model_h2h_choice %>%
  emmeans(~ strategy) %>%
  as.data.frame()

contrast_h2h_overall <- model_h2h_choice %>%
  emmeans(~ strategy) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_h2h_overall <- emm_h2h %>%
  ggplot(aes(x = strategy, y = emmean)) +
  geom_col(fill = "steelblue", width = 0.8, alpha = 0.8) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.1,
    color = "black"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  xlab("Strategy") +
  ylab("P(Choice)") +
  theme_custom()

save_plot(plot_h2h_overall, "human_head2head_overall.pdf", width = 4, height = 3)

emm_h2h.by_task <- model_h2h_choice %>%
  emmeans(~ strategy | task) %>%
  as.data.frame()

contrast_h2h_by_task <- model_h2h_choice %>%
  emmeans(~ strategy | task) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_h2h_by_task <- emm_h2h.by_task %>%
  ggplot(aes(x = str_to_title(task), y = emmean, fill = strategy)) +
  geom_col(position = position_dodge(width = 0.9), width = 0.8) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.2,
    position = position_dodge(width = 0.9),
    color = "black"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_fill_cosmic() +
  xlab("Strategy") +
  ylab("P(Choice)") +
  guides(fill = guide_legend(title = "Strategy")) +
  theme_custom()

save_plot(plot_h2h_by_task, "human_head2head_by_task.pdf", width = 6, height = 4)

table_h2h_overall <- build_delta_table(
  emm_tbl = emm_h2h %>% mutate(group = "Overall") %>% rename(Group = group),
  contrast_tbl = contrast_h2h_overall %>% mutate(group = "Overall") %>% rename(Group = group),
  group_vars = "Group",
  grouping_var = "strategy",
  caption = "Human head-to-head choice probabilities by strategy. Main value is the estimated marginal mean probability; parentheses show $\\Delta$ vs. the \\colorbox[HTML]{c4dd88}{best strategy}.",
  label = "human_head2head_overall",
  column_order = STRATEGY_LEVELS,
  grouping_levels = STRATEGY_LEVELS
) %>%
  kableExtra::add_header_above(
    c(" " = 1, "Strategy (P(Choice) with $\\\\Delta$ vs. best)" = 3),
    escape = FALSE
  )

save_table(table_h2h_overall, "human_head2head_emm_overall.tex")

table_h2h_by_task <- build_delta_table(
  emm_tbl = emm_h2h.by_task %>% mutate(task = str_to_title(task)) %>% rename(Task = task),
  contrast_tbl = contrast_h2h_by_task %>% mutate(task = str_to_title(task)) %>% rename(Task = task),
  group_vars = "Task",
  grouping_var = "strategy",
  caption = "Human head-to-head choice probabilities by task and strategy. Main value is the estimated marginal mean probability; parentheses show $\\Delta$ vs. the \\colorbox[HTML]{c4dd88}{best strategy} within each task.",
  label = "human_head2head_by_task",
  column_order = STRATEGY_LEVELS,
  grouping_levels = STRATEGY_LEVELS
) %>%
  kableExtra::add_header_above(
    c(" " = 1, "Strategy (P(Choice) with $\\\\Delta$ vs. best)" = 3),
    escape = FALSE
  )

save_table(table_h2h_by_task, "human_head2head_emm_by_task.tex")


################################################################################
# DATA LOADING & PREPARATION (MITIGATION ANALYSIS)
################################################################################

d.mitigation <- read.csv("data/mitigation_results-human.csv") %>%
  rename(
    chosen = chosenImage,
    participant_id = prolificPid
  ) %>%
  mutate(
    chosen = ifelse(chosen == image1_id, image1_status, image2_status) %>%
      factor(levels = c("original", "final"), labels = c("Original", "Final")),
    k = 3
  ) %>%
  select(
    participant_id,
    task,
    image1_status,
    image2_status,
    chosen,
    k
  ) %>%
  rbind(
    d %>%
      mutate(k = 0) %>%
      select(participant_id, task, image1_status, image2_status, chosen, k) %>%
      filter((image1_status != "zero-shot") & (image2_status != "zero-shot")) %>%
      droplevels()
  ) %>%
  mutate(k = factor(k, levels = c(0, 3), labels = c("k=0", "k=3")))

d.mitigation_choice <- d.mitigation %>%
  mutate(pair_id = row_number()) %>%
  pivot_longer(
    cols = c(image1_status, image2_status),
    names_to = "slot",
    values_to = "status"
  ) %>%
  mutate(
    status = factor(status, levels = c("original", "final"), labels = c("Original", "Final")),
    chosen_flag = as.integer(status == chosen)
  )


################################################################################
# MAIN EXECUTION (MITIGATION ANALYSIS)
################################################################################

model_choice_mitigation <- feols(
  chosen_flag ~ status * task * k,
  vcov = ~ participant_id + pair_id,
  data = d.mitigation_choice
)

reg_choice_mitigation <- etable(
  model_choice_mitigation,
  tex = FALSE
) %>% kableExtra::kbl(
  format = "latex",
  booktabs = TRUE,
  align = "l"
)
save_table(reg_choice_mitigation, "human_choice_mitigation_regression.tex")

emm_choice_mitigation_overall <- model_choice_mitigation %>%
  emmeans(~ status | k) %>%
  as.data.frame()

contrast_choice_mitigation_overall <- model_choice_mitigation %>%
  emmeans(~ status | k) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_choice_mitigation_overall <- emm_choice_mitigation_overall %>%
  ggplot(aes(x = status, y = emmean, color = k, group = k)) +
  geom_col(
    aes(fill = k),
    position = position_dodge(width = 0.6),
    alpha = 0.5,
    width = 0.5,
    color = NA
  ) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL, group = k),
    position = position_dodge(width = 0.6),
    width = 0.1,
    color = "black"
  ) +
  stat_pvalue_manual(
    build_contrast_df(emm_choice_mitigation_overall, contrast_choice_mitigation_overall, group_vars = NULL, grouping_var = "status", dodge_var = "k", dodge_width = 0.6) %>%
      filter(label != ""),
    label = "label",
    x1 = "group1",
    x2 = "group2",
    color = "k",
    y.position = "y",
    step.increase = 0.1,
    show.legend = FALSE
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_color_manual(
    name = "Num. Passes",
    values = c("k=0" = "black", "k=1" = "#26B4F9", "k=3" = "#0072B2"),
    labels = c("k=0" = "No Mitigation", "k=1" = "1 Pass", "k=3" = "3 Passes")
  ) +
  scale_fill_manual(
    name = "Num. Passes",
    values = c("k=0" = "black", "k=1" = "#26B4F9", "k=3" = "#0072B2"),
    labels = c("k=0" = "No Mitigation", "k=1" = "1 Pass", "k=3" = "3 Passes")
  ) +
  xlab("Status") +
  ylab("P(Choice)") +
  guides(fill = guide_legend(title = "Num. Passes")) +
  theme_custom() +
  theme(legend.position = "top")

save_plot(plot_choice_mitigation_overall, "human_choice_mitigation_overall.pdf", width = 4, height = 2.4)

emm_choice_mitigation_task <- model_choice_mitigation %>%
  emmeans(~ status | k | task) %>%
  as.data.frame()

contrast_choice_mitigation_task <- model_choice_mitigation %>%
  emmeans(~ status | k | task) %>%
  contrast("pairwise", adjust = "none") %>%
  as.data.frame() %>%
  mutate(p.value = p.adjust(p.value, method = "BH"))

plot_choice_mitigation_task <- emm_choice_mitigation_task %>%
  ggplot(aes(x = status, y = emmean, fill = k)) +
  geom_col(
    position = position_dodge(width = 0.6),
    alpha = 0.5,
    width = 0.5,
    color = NA
  ) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    position = position_dodge(width = 0.6),
    width = 0.1,
    color = "black"
  ) +
  stat_pvalue_manual(
    build_contrast_df(emm_choice_mitigation_task, contrast_choice_mitigation_task, group_vars = "task", grouping_var = "status", dodge_var = "k", dodge_width = 0.6) %>%
      filter(label != ""),
    label = "label",
    x1 = "group1",
    x2 = "group2",
    color = "k",
    y.position = "y",
    step.increase = 0.1,
    step.group.by = "task",
    show.legend = FALSE
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0.1)),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_color_manual(
    name = "Num. Passes",
    values = c("k=0" = "black", "k=1" = "#26B4F9", "k=3" = "#0072B2"),
    labels = c("k=0" = "No Mitigation", "k=1" = "1 Pass", "k=3" = "3 Passes")
  ) +
  scale_fill_manual(
    name = "Num. Passes",
    values = c("k=0" = "black", "k=1" = "#26B4F9", "k=3" = "#0072B2"),
    labels = c("k=0" = "No Mitigation", "k=1" = "1 Pass", "k=3" = "3 Passes")
  ) +
  xlab("Status") +
  ylab("P(Choice)") +
  guides(color = guide_legend(title = "Task")) +
  facet_grid(~ str_to_title(task)) +
  theme_custom()

save_plot(plot_choice_mitigation_task, "human_choice_mitigation_by_task.pdf", width = 8, height = 2.4)
