# Load CSV
df <- read.csv("C:/Users/rache/Desktop/seminar/Kclustering/liver.csv", header = TRUE)

# removing duplicate rows
print(nrow(df))     
df <- unique(df)
print(nrow(df))     

# Set threshold for heavy drinking
threshold <- 1

# 1 = heavy drinker, 0 = not heavy
#if drinks is bigger than 2 then put 1 else put 0
df$target <- ifelse(df$drinks >= threshold, 1, 0)

# Remove drinks and selector columns
df <- subset(df, select = -c(drinks, selector))

head(df)
# Reorder columns with sgot before sgpt
df <- df[, c("mcv", "alkphos", "sgot", "sgpt", "gammagt", "target")]
head(df)

# Shuffle the dataset
set.seed(123)             # Optional: for reproducibility
df <- df[sample(nrow(df)), ]
rownames(df) <- NULL      # Reset row numbers

# Show first rows to verify shuffle
head(df)

# Save the shuffled dataset
write.csv(df, "C:/Users/rache/Desktop/seminar/Kclustering/liver_cleaned.csv", row.names = FALSE, quote = FALSE)
