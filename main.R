# library requirements
# simply install.package("*package_name*") if you do not have these
library(stringr)
library(keras)
library(tensorflow)
install_tensorflow()

# LOADING THE DATA
raw_text <- readLines("01MND.txt", encoding = "UTF-8" )

# PRE PROCESS THE DATA
text <- str_c(raw_text, collapse = "/")
text <- tolower(text)
text <- str_replace_all(text, "[[:punct:]|[:digit:]]", " ")
text <- str_replace_all(text, "=", "")

# word list - unique words in the document
words <- str_split(text," ")
words <- as.factor(words[[1]])
words <- droplevels(words[!words == ''])
word.unique <- (levels(words))
words <- as.character(words)

# split sequences
seq.length = 5

# list of indexes to split with
indexes <- seq(1, length(words) - seq.length, 3)

# list of features (sentences) and labels (next words)
feature <- list()
label <- list()

for (i in 1:length(indexes)){
  x = indexes[i]
  feature <- c(feature, list(words[x:(x + seq.length - 1)]))
  label <- c(label, words[x + seq.length])
}

# ENCODING FEATURE AND LABELS
feat <- array(0, dim = c(length(feature), seq.length, length(word.unique)))
lab <- array(0, dim = c(length(label), length(word.unique)))

for (i in 1:length(feature)){
  
  temp <- feature[i]
  for (t in 1:length(temp)) {
    word <- temp[[t]]
    feat[i, t, match(word,word.unique)] <- 1
  }
  
  
  lab[i,match(label[i],word.unique)] <- 1
}


# MODEL BUILDING
model.1 <- keras_model_sequential() %>%
  layer_lstm(units = 256, input_shape = c(seq.length, length(word.unique))) %>%
  layer_dense(units = length(word.unique), activation = "softmax")

model.1 %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = 'adam'
)

# TRAINING MODEL
model.1 %>% fit(feat,lab, epochs = 60)

# FUNCTION TO RETURN THE MOST LIKELY WORD BASED ON TEMPERATURE
# based on the Keras's official tutorials and examples
sample_next_word <- function(preds, temperature = 1.0) {
  preds <- as.numeric(preds)
  preds <- log(preds) / temperature
  exp_preds <- exp(preds)
  preds <- exp_preds / sum(exp_preds)
  which.max(t(rmultinom(1, 1, preds)))
}

# sampling a random prompt for the text prediction
start_index <- abs(sample(1:(length(words) - seq.length), size = 1))
generated <- list(words[start_index:(start_index + seq.length - 1)])

# PREDICTING TEXT
for (i in 1:400) {
  
  sampled <- array(0, dim = c(1, seq.length, length(word.unique)))
  
  for (t in 1:length(generated)) {
    word <- generated[[t]]
    sampled[1, t, match(word, word.unique)] <- 1
  }
  
  preds <- model.1 %>% predict(sampled, verbose = 0)
  next_index <- sample_next_word(preds[1,], 1)
  next_word <- word.unique[[next_index]]
  
  generated <- paste0(generated, next_word)
  generated <- c(generated[-1], next_word)
  
  cat(" ")
  cat(next_word)
}
