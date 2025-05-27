required_packages <- c(
  "rvest", "dplyr", "readr", "tm", "textclean", "tokenizers", 
  "textstem", "hunspell", "stringr", "topicmodels", "tidytext", 
  "textdata", "broom", "ggplot2", "wordcloud", "RColorBrewer", "slam"
)
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)

invisible(lapply(required_packages, library, character.only = TRUE))

urls <- c(
  "https://www.dhakatribune.com/opinion/longform/372730/bangladesh-s-ai-moment",
  "https://www.dhakatribune.com/opinion/op-ed/378786/what-bangladesh's-ai-policy-must-get-right",
  "https://www.dhakatribune.com/opinion/op-ed/145905/blockchain-trusting-no-one",
  "https://www.dhakatribune.com/business/135149/blockchain-a-game-changing-technology",
  "https://globalcybersecuritynetwork.com/blog/advanced-wireless-connectivity-solutions/"
)

get_type <- function(url) {
  parts <- unlist(strsplit(url, "/"))
  type <- parts[which(parts == "news") - 1]
  if (length(type) == 0) type <- parts[4]
  return(type)
}

articles_df <- data.frame(website_link = character(), heading = character(), description = character(), type_of_article = character(), stringsAsFactors = FALSE)

for (url in urls) {
  cat("Processing:", url, "\n")
  try({
    webpage <- read_html(url)
    heading <- html_text(html_node(webpage, ".row"), trim = TRUE)
    paragraphs <- html_text(html_nodes(webpage, "p"), trim = TRUE)
    description <- paste(paragraphs, collapse = " ")
    type <- get_type(url)
    
    articles_df <- rbind(articles_df, data.frame(
      website_link = url,
      heading = heading,
      description = description,
      type_of_article = type,
      stringsAsFactors = FALSE
    ))
  }, silent = TRUE)
}

write.csv(articles_df, "AI_Sports_Articles.csv", row.names = FALSE)

df <- read_csv("AI_Sports_Articles.csv")

df$clean_text <- tolower(df$description) %>%
  replace_contraction() %>%
  replace_emoji() %>%
  replace_emoticon() %>%
  str_replace_all("http\\S+|www\\S+", "") %>%
  str_replace_all("[^a-z\\s]", " ")

df$tokens <- lapply(df$clean_text, tokenize_words)
stop_words <- stopwords("en")
df$tokens_no_stop <- lapply(df$tokens, function(x) setdiff(x, stop_words))

df$tokens_checked <- lapply(df$tokens_no_stop, function(tokens) {
  sapply(tokens, function(word) {
    if (!all(hunspell_check(word))) {
      suggestions <- hunspell_suggest(word)[[1]]
      if (length(suggestions) > 0) return(suggestions[1]) else return(word)
    } else {
      return(word)
    }
  })
})



df$lemmatized_tokens <- lapply(df$tokens_checked, lemmatize_words)
df$clean_description <- sapply(df$lemmatized_tokens, paste, collapse = " ")

write_csv(df, "Processed_Articles.csv")

df <- df %>% filter(!is.na(clean_text))

custom_stopwords <- c("can", "like", "will", "also", "just", "must")
corpus <- VCorpus(VectorSource(df$clean_text)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, c(stopwords("english"), custom_stopwords)) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus, control = list(wordLengths = c(3, Inf)))
dtm <- removeSparseTerms(dtm, 0.995)
row_totals <- slam::row_sums(dtm)
dtm <- dtm[row_totals > 0, ]
df <- df[row_totals > 0, ]

num_topics <- 5
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 1234))
topic_labels <- c("1" = "Future Vision & Innovation", "2" = "Blockchain & Technology", "3" = "Wireless & Cybersecurity", "4" = "Government & Economy", "5" = "Development & National Policy")

topic_terms <- tidy(lda_model, matrix = "beta")
top_terms <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_terms$topic <- factor(top_terms$topic, levels = names(topic_labels), labels = topic_labels)

ggplot(top_terms, aes(x = reorder(term, beta), y = beta, fill = topic)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Terms per Interpreted Topic", x = "Term", y = "Probability")

topic_proportions <- as.data.frame(posterior(lda_model)$topics)
colnames(topic_proportions) <- paste0("Topic_", 1:num_topics)
df <- cbind(df, topic_proportions)
df$Dominant_Topic <- apply(topic_proportions, 1, which.max)
df$Topic_Name <- topic_labels[as.character(df$Dominant_Topic)]

write_csv(df, "UPDATED_Articles_With_Topics.csv")

tdm <- TermDocumentMatrix(corpus)
word_freqs <- sort(rowSums(as.matrix(tdm)), decreasing = TRUE)
df_freq <- data.frame(word = names(word_freqs), freq = word_freqs) %>% filter(nchar(word) <= 15)

set.seed(1234)
suppressWarnings(
  wordcloud(
    words = df_freq$word,
    freq = df_freq$freq,
    min.freq = 2,
    max.words = 100,
    random.order = FALSE,
    rot.per = 0.35,
    colors = brewer.pal(8, "Dark2")
  )
)

text_output <- mapply(function(link, heading, desc, type) {
  paste0("Website Link: ", link, "\n",
         "Heading: ", heading, "\n",
         "Description: ", desc, "\n",
         "Type of Article: ", type, "\n",
         strrep("-", 40), "\n")
}, df$website_link, df$heading, df$description, df$type_of_article, SIMPLIFY = TRUE)

writeLines(text_output, "Processed_AI_Articles.txt")

