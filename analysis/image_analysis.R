source("utils.R")
library(showtext)
library(cowplot)
library(patchwork)


################################################################################
# DATA LOADING AND PREPARATION
################################################################################

d.embedding <- read.csv("similarity_analysis/similarity_analysis_out/embedding_similarity_long.csv")
d.lpips <- read.csv("similarity_analysis/similarity_analysis_out/lpips_distance_long.csv")
d.ssim <- read.csv("similarity_analysis/similarity_analysis_out/ssim_similarity_long.csv")
d.matted_embedding <- read.csv("similarity_analysis/similarity_analysis_out/matted_embedding_similarity_long.csv")
d.matted_ssim <- read.csv("similarity_analysis/similarity_analysis_out/matted_ssim_similarity_long.csv")

d.embedding <- d.embedding %>% mutate(metric = "CLIP ↑")
d.lpips <- d.lpips %>% mutate(metric = "LPIPS ↓")
d.ssim <- d.ssim %>% mutate(metric = "SSIM ↑")
d.matted_embedding <- d.matted_embedding %>% mutate(metric = "CLIP (No-BG) ↑")
d.matted_ssim <- d.matted_ssim %>% mutate(metric = "SSIM (No-BG) ↑")

d.all <- bind_rows(d.embedding, d.lpips, d.ssim, d.matted_embedding, d.matted_ssim) %>%
  mutate(
    comparison = case_when(
      comparison == "between_pairs" ~ "Between Pair",
      (comparison == "from_original" & (is_final == "True")) ~ "Mitigated (Original) vs. Original",
      (comparison == "from_original" & (is_final == "False")) ~ "Mitigated (Optimized) vs. Original",
    ),
    comparison = factor(comparison, levels = c("Between Pair", "Mitigated (Original) vs. Original", "Mitigated (Optimized) vs. Original")),
    score = case_when(
      metric == "LPIPS ↓" ~ distance,
      TRUE ~ similarity
    ),
    metric = factor(metric, levels = c("CLIP ↑", "SSIM ↑", "LPIPS ↓", "CLIP (No-BG) ↑", "SSIM (No-BG) ↑"))
  )


################################################################################
# MAIN EXECUTION
################################################################################

plot.ratios <- d.all %>%
  group_by(metric, comparison, debiased_iter) %>%
  summarize(
    mean_score = mean(score),
    sd_score = sd(score),
    se_score = sd(score) / sqrt(n()),
    lci_score  = mean_score - qt(0.975, df = n() - 1) * se_score,
    uci_score = mean_score + qt(0.975, df = n() - 1) * se_score
  ) %>%
  ggplot(aes(x = debiased_iter, y = mean_score, color = comparison)) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = lci_score, ymax = uci_score), width = 0.2, position = position_dodge(width = 0.5)) +
  geom_line(position = position_dodge(width = 0.5)) +
  scale_color_cosmic() +
  xlab("Mitigation Steps") +
  ylab("Similarity / Distance") +
  guides(color = guide_legend(title = "Comparison")) +
  facet_wrap(~metric, scales = "free_y") +
  theme_custom()

leg <- get_legend(plot.ratios + theme(legend.position = "right"))

plot.ratios_final <- plot.ratios +
  theme(legend.position = "none") +
  inset_element(
    ggdraw(leg),
    left = 2/3, bottom = 0, right = 1, top = 1/2,
    align_to = "panel"
  )

showtext_auto()
plot.ratios_final %>%
  ggsave(
    filename = "plots/similarity_analysis_plot.pdf",
    width = 7.2,
    height = 4,
    device = cairo_pdf # To support the arrows
  )
