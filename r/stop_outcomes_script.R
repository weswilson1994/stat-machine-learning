# Packages ----------------------------------------------------------------
library(tidyverse)
library(class)
library(MASS)
library(janitor)

# Functions ---------------------------------------------------------------
find_best_k <- function(k, x_variables, y_variable) {
  
  xtrain = model_data$train[, x_variables] # Our training set X variables
  ytrain = model_data$train[[y_variable]] # Our training set y variable
  xtest = model_data$test[, x_variables] # Our X variables from our test
  ytest = model_data$test[[y_variable]] # Our training set Y variable
  
  yhat <- knn(train = xtrain, test = xtest, cl = ytrain, k = k)
  
  # extract performance metrics 
  err_rate = mean(yhat != ytest) #Error rate --> the prediction is not correct
  true_positive_rate = sum(yhat == 1 & ytest == 1) / sum(ytest == 1) # TP/P
  false_positive_rate = sum(yhat == 1 & ytest == 0) / sum(ytest == 0) #FP/N
  
  #results
  results <- tibble(
    k = k, 
    err_rate = err_rate, 
    true_positive_rate = true_positive_rate, 
    false_positive_rate = false_positive_rate
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
calc_confusion_matrix <- function(y_predicted, y_actual) {
  table(predicted = y_predicted, actual = y_actual)
}


# Data --------------------------------------------------------------------
stop_data <- read_csv(file = "data/original/Stop_Data.csv", na = "NULL")

# Data Preparation --------------------------------------------------------
stop_data <- clean_names(stop_data)

#based on reading the documentation of what might be potentially useful variables.
stop_data <- stop_data %>%
  select(
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
  mutate(across(c(ticket_count, warning_count), ~ replace_na(0))) %>% 
  # create variables for time of day, day of the week, etcetera. 
  mutate(datetime = ymd_hms(datetime)) %>% 
  mutate(day_of_week = wday(datetime, label = TRUE)) %>% 
  mutate(hour_of_day = hour(datetime)) %>% 
  mutate(time_category = case_when(
    hour_of_day >= 6 & hour_of_day < 12 ~ "Morning",
    hour_of_day >= 12 & hour_of_day < 17 ~ "Afternoon",
    hour_of_day >= 17 & hour_of_day < 20 ~ "Evening",
    TRUE ~ "Night"
  ), .keep = "unused") %>% 
  # these are binary columns that I will turn to factors. 
  mutate(across(c(person_searched, property_searched, traffic_involved), ~ as.factor(.x))) %>% 
  # some stop durations are clearly data entry errors (with stops spanning several days!) I will remove them.
  filter(stop_duration_mins < 60 * 8)

# export in case my colleagues want to use the same data.
#write_rds(x = stop_data, file = "data/final/wes_cleaned_stop_data.rds")


# Models ------------------------------------------------------------------


# KNN ---------------------------------------------------------------------


## LDA and QDA -------------------------------------------------------------



# Cross-Validation --------------------------------------------------------



# Plots -------------------------------------------------------------------




