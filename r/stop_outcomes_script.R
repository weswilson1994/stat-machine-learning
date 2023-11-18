# Packages ----------------------------------------------------------------
library(tidyverse)
library(class)
library(MASS)
library(janitor)
library(caret)
library(gtsummary)
library(glmnet)
library(memoise)

# Functions ---------------------------------------------------------------
find_best_k <- function(k, x_variables, y_variable) {
  
  xtrain = model_data$train[, x_variables] # Our training set X variables
  ytrain = model_data$train[[y_variable]] # Our training set y variable
  xtest = model_data$test[, x_variables] # Our X variables from our test
  ytest = model_data$test[[y_variable]] # Our training set Y variable
  
  yhat <- knn(train = xtrain, test = xtest, cl = ytrain, k = k)
  
  # extract performance metrics 
  accuracy_rate = 1 - mean(yhat != ytest) #Accuracy rate rate --> the prediction is not correct

  #results
  results <- tibble(
    k = k, 
    accuracy_rate = accuracy_rate
  )
  return(results)
}

#metrics <- map_dfr(1:50, ~ find_best_k(k = ., x_variables = c("yards", "week"), y_variable = "success"))

split_train_and_test_data <- function(split_pct, data, seed){
  set.seed(seed)
  z <- sample(nrow(data), split_pct * nrow(data))
  train <- data[z,]
  test <- data[-z,]
  
  return(list(train = train, test = test))
}

# Data --------------------------------------------------------------------
stop_data <- read_csv(file = "data/original/Stop_Data.csv", na = "NULL")
n_cols <- ncol(stop_data)
n_rows <- nrow(stop_data)

# Data Preparation --------------------------------------------------------
stop_data <- clean_names(stop_data)

#based on reading the documentation of what might be potentially useful variables.
stop_data <- stop_data %>%
  dplyr::select(
    #outcomes of the stop
    stop_type,  arrest_charges, tickets_issued,ticket_count, warnings_issued, warning_count,
    # information about the stop
    datetime, stop_district, stop_duration_mins, stop_reason_nonticket, stop_reason_ticket, person_searched = person_search_pat_down, property_searched = property_search_pat_down, traffic_involved, 
    # information about the person involved in the stop
    gender, ethnicity, age) %>% 
  # not interested in nautical infractions, so I will remove harbor stops. 
  filter(stop_type != "Harbor") %>% 
  # create mutually exclusive stop outcome variable 
  mutate(stop_outcome = case_when(
    is.na(warnings_issued) & is.na(tickets_issued) & is.na(arrest_charges) ~ "No Action",
    !is.na(warnings_issued) & is.na(tickets_issued) & is.na(arrest_charges) ~ "Warning", 
    !is.na(arrest_charges) ~ "Arrest",
    !is.na(tickets_issued) ~ "Ticket"), .keep = "unused", .before = 1) %>% 
  # I will combine the separate stop reasons for ticket non-ticket stops into one variable, and drop the other two
  mutate(stop_reason = tolower(coalesce(stop_reason_nonticket, stop_reason_ticket)), .keep = "unused") %>% 
  # now the problem is that stop_reason is very long. It looks like the primary reason is the first string in the variable, and any other reasons are added, I will separate these out. 
  separate(col = stop_reason, into = "primary_stop_reason", sep = ",|;" ) %>% 
  #now i just have to clean up the strings and categorize them. 
  mutate(primary_stop_reason = case_when(
    str_detect(primary_stop_reason, pattern = "bolo") ~ "responding to bolo",
    str_detect(primary_stop_reason, pattern = "for service") ~ "call for service",
    str_detect(primary_stop_reason, pattern = "individual’s") ~ "responding to individual's actions",
    str_detect(primary_stop_reason, pattern = "sources") ~ "information obtained from law enforcement sources",
    str_detect(primary_stop_reason, pattern = "initiated|prior knowledge|observed") ~ "officer initiated",
    str_detect(primary_stop_reason, pattern = "crash|traffic") ~ "traffic response",
    is.na(primary_stop_reason) ~ "Unknown",
    TRUE ~ primary_stop_reason)) %>% 
  # replace na values for ticket counts
  mutate(across(c(ticket_count, warning_count), ~ replace_na(., 0))) %>% 
  # create variables for time of day, day of the week, etcetera. 
  mutate(datetime = ymd_hms(datetime)) %>% 
  mutate(day_of_week = wday(datetime, label = TRUE)) %>% 
  mutate(hour_of_day = hour(datetime)) %>% 
  mutate(time_category = case_when(
    hour_of_day >= 6 & hour_of_day < 12 ~ "Morning",
    hour_of_day >= 12 & hour_of_day < 17 ~ "Afternoon",
    hour_of_day >= 17 & hour_of_day < 20 ~ "Evening",
    TRUE ~ "Night"
  )) %>% 
  # some stop durations are clearly data entry errors (with stops spanning several days!) I will remove them.
  filter(stop_duration_mins < 60 * 8) %>% 
  mutate(age = as.numeric(age)) %>% 
  # select the final variables-->drop na values because modeling wont work with them.  
  filter(stop_duration_mins > 0) %>% 
  dplyr::select(-stop_type, -datetime) %>% 
  drop_na() 


# export in case my colleagues want to use the same data.
write_rds(x = stop_data, file = "data/final/wes_cleaned_stop_data.rds")

stop_data <- stop_data %>% 
  mutate(male = if_else(gender == "Male", 1, 0)) %>% 
  mutate(white_person = if_else(ethnicity == "White", 1, 0)) %>% 
  dplyr::select(-ethnicity, -gender)

# Exploratory Analysis ----------------------------------------------------
stop_summary <- stop_data %>% 
  group_by(stop_outcome) %>% 
  summarise(mean_duration = mean(stop_duration_mins, na.rm = T), 
            sd_duration = sd(stop_duration_mins), 
            mean_age = mean(age, na.rm = TRUE), 
            sd_age = sd(age, na.rm = TRUE),
            n = n()) %>% 
  ungroup() %>% 
  mutate(prop = n / sum(n))

stop_summary

# looking at the SD of the age and duration, I think qda or knn will be better. 
# create a graph to analyze how the x variables are distributed 
ggplot(data = slice_sample(stop_data, n = 1000), mapping = aes(y = stop_duration_mins, x = age, color = stop_outcome), alpha = .5) +
  geom_jitter()

# well the x variables are not normally distributed. 
qqplot(x = stop_data$stop_duration_mins, stop_data$age)


# Model Selection (lasso) -------------------------------------------------
# Model Selection
# given how big the data set is, a 70/30 split seems appropriate. 
model_data <- split_train_and_test_data(split_pct = .7, data = stop_data, seed = 123)

x_train <- model.matrix(lm(ticket_count ~ . -1 - stop_outcome, data = model_data$train))
y_train <- model_data$train$stop_outcome

x_test <- model.matrix(lm(ticket_count ~ . -1 - stop_outcome, data = model_data$test))
y_test <- model_data$test$stop_outcome

# I know I need to remove perfectly linear variables, so I will use lasso .
r_train_cv <- cv.glmnet(x = x_train, y = y_train, alpha = 1, family = "multinomial")

r_train_cv
selected_coef <- coef(r_train_cv, s = "lambda.1se")
cv_plot <- plot(r_train_cv)
ggsave(filename = "./r/plots/lasso_model_selection.png", plot = cv_plot)


## Final x variables -------------------------------------------------------
# lasso selected all of the variables. 

# Models ------------------------------------------------------------------
# first, I will use the lasso model. 
# this returns probabilities, so I have to turn them to estimates. 
pred_test <- predict(r_train_cv, newx = x_test, type = "response") %>% as.data.frame()

outcomes <- c("arrest", "no action", "ticket", "warning")

pred_test$predicted_categories <- outcomes[max.col(pred_test)]

mean(pred_test$predicted_categories != y_test)
# well, this model is 100% accurate, which can't be right. I will just use the variables it selected, which is all of them....
# Models ------------------------------------------------------------------
## KNN ---------------------------------------------------------------------

### KNN -- Non-tuned -- with model matrix approach ------------------------------------------------------
# I have to create a model matrix given how many categorical variables there are. 
# this is the data that will work for knn. I am going to put the others in a separate dataframe to build a model matrix. 
knn_data <- stop_data %>% 
  dplyr::select(stop_outcome, stop_duration_mins, person_searched, property_searched, traffic_involved, age, hour_of_day, day_of_week)

matrix_xs <- stop_data[, setdiff(names(stop_data), names(knn_data))] %>% 
  dplyr::select(!warning_count) %>% 
  drop_na()

#now create the model matrix. 
new_data <- model.matrix(lm(ticket_count ~ . -1, data = matrix_xs)) 

#put it all together
knn_data <- cbind(knn_data, new_data) 

# given how big the data set is, a 70/30 split seems appropriate. 
model_data <- split_train_and_test_data(split_pct = .7, data = knn_data, seed = 123)

x_variables <- setdiff(names(knn_data), c("stop_outcome", "day_of_week"))
outcome_variable <- "stop_outcome"

xtrain <- model_data$train[, x_variables] 
x_test <- model_data$test[, x_variables]
y_train <- model_data$train[[outcome_variable]]
y_test <- model_data$test[[outcome_variable]]

yhat <- knn(train = xtrain, test = x_test, cl = y_train, k = 1)

knn_model_matrix <- 1 - mean(yhat != y_test)
knn_model_matrix

### KNN -- Non-tuned Without model matrix approach --------------------------------------------------------

model_data <- split_train_and_test_data(split_pct = .7, data = stop_data, seed = 123)

x_variables <- c("stop_duration_mins","person_searched", "property_searched", "traffic_involved", "age", "male", "white_person")
x_variables

xtrain <- model_data$train[, x_variables] 
x_test <- model_data$test[, x_variables]
y_train <- model_data$train[[outcome_variable]]
y_test <- model_data$test[[outcome_variable]]

yhat <- knn(train = xtrain, test = x_test, cl = y_train, k = 1)

knn_regular <- 1 - mean(yhat != y_test)
knn_regular

### KNN -- Tuned Without model matrix approach --------------------------------------------------------
metrics <- map_dfr(1:50, ~ find_best_k(k = ., x_variables = x_variables, y_variable = "stop_outcome"))

tuned_knn_no_matrix <- metrics[which.max(metrics$accuracy_rate), ]
tuned_knn_no_matrix_accuracy <- tuned_knn_no_matrix$accuracy_rate
tuned_knn_no_matrix_accuracy
## LDA and QDA -------------------------------------------------------------

### LDA Model ---------------------------------------------------------------

# wow, lda does the matrix automatically...
lqda_data <- stop_data %>% 
  #these variables create linear combinations
  dplyr::select(! c(ticket_count, warning_count, traffic_involved, primary_stop_reason)) 

lda_all <- lda(data = lqda_data, stop_outcome ~ ., CV = TRUE)
lda_all

lda_accuracy <- mean(lda_all$class == stop_data$stop_outcome)

lda_accuracy

### QDA Model ---------------------------------------------------------------
qda_model <- qda(stop_outcome ~ stop_district + stop_duration_mins + person_searched + property_searched + age + hour_of_day + day_of_week + male + white_person, 
                 data = lqda_data, CV = TRUE)

qda_accuracy <- mean(qda_model$class == stop_data$stop_outcome)

qda_accuracy

knn_model_matrix
knn_regular
tuned_knn_no_matrix_accuracy
lda_accuracy
qda_accuracy

# Plots -------------------------------------------------------------------
accuracy <- tibble("Model" = c("KNN (not tuned)", "KNN (model matrix)", "KNN (tuned)", "LDA", "QDA"),
       "Accuracy Rate" = c(knn_regular, knn_model_matrix, tuned_knn_no_matrix_accuracy, lda_accuracy, qda_accuracy)) %>% 
  arrange(desc("Accuracy Rate")) %>% 
  gt::gt(caption = "Accuracy of All Stop Outcome Models")

accuracy

gt::gtsave(data = accuracy, filename = "./r/plots/stop_outcome_accuracy.PNG")

gt::gtsave()