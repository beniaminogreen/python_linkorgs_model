library(tidyverse)
library(arrow)

clean_data <- function(data){
    data %>%
        select(name1, to_key1) %>%
        distinct() %>%
        rename(
               name = name1, key = to_key1
               )

}

pos_data <- read_csv("../data/PosMatches_mat.csv") %>%
    clean_data()

neg_data <- read_csv("../data/NegMatches_mat.csv") %>%
    clean_data()

training_data <- bind_rows(pos_data, neg_data) %>%
    distinct()

write_parquet(training_data, "../data/training_data.parquet")
write_csv(training_data, "../data/training_data.csv")
