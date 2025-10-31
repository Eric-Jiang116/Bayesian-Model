library(ggplot2)
### Load simulated data
load("~/Downloads/BayesianSILA-main/simulation/out.data.Rdata")
data <- out.data$data # observed subject data
constants <- out.data$constants # other important constants
d <- out.data$d # disease age
theta <- out.data$theta # varying coefficient function coefficients for B-Splines


######################################## Plot subject-level data ########################################
subj_data <- data.frame(data$y_pred, constants$subject, d)
# y is outcome data
# d i 
names(subj_data) <- c("y", "ID", "d")

# Subject-level outcome data (first hundred subjects for visualization purposes)
ggplot(subj_data[which(subj_data$ID < 100),], aes(x = d, y = y, group = ID, color = as.factor(ID))) +
  geom_line(alpha = 0.6, linewidth = 0.7) +
  labs(
    x = "d",
    y = "y",
    color = "Subject ID",
    title = expression("Subject-level trajectories")
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",         # hide if too many subjects
    plot.title = element_text(hjust = 0.5)
  )


######################################## Plot true varying coefficient functions ########################################

# Subject-level predictors - multiplied and summed with varying coefficient functions
# to produce log-scale rate vs. value curve. 
head(constants$Xsub)

# Varying coefficient functions
# simulated vc functions are a linear combination of B-splines
vc_functions <- constants$B_pred %*% theta 

# log scale
matplot(constants$v_pred, vc_functions, type = "l", xlab = "v", 
        ylab = expression(beta[r](v)), main = "Log rate scale varying coefficient functions")

# rate scale
matplot(constants$v_pred, exp(vc_functions), type = "l", xlab = "v", 
        ylab = expression(exp(beta[r](v))), main = "Rate scale varying coefficient functions")


