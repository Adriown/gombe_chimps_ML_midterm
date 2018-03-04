library(tidyverse)
library(data.table)
library(corrplot)
library(ggplot2)
library(dplyr)

dummy <- read_csv(file = 'Downloads/gombe_460.csv') %>%
  select('chimpcode', 'ratercode')

joined = inner_join(dummy, dummy, by = 'chimpcode')
joined[joined$chimpcode == 'K262',] 

length(unique(dummy$chimpcode))
# 128 chimps
length(unique(dummy$chimpcode))

# 18 raters
length(unique(dummy$ratercode))



df_fill = as.tibble(expand.grid(ratercode.x = unique(dummy$ratercode), 
                                ratercode.y = unique(dummy$ratercode)))
arrange(df_fill,ratercode.x)

dummy2 = df_fill[1,]
for (X in 2:NROW(df_fill)){
  a = df_fill[X,]
  b = df_fill[X,]
  d = a$ratercode.x
  b$ratercode.x = a$ratercode.y
  b$ratercode.y = d
  if(NROW(inner_join(dummy2, a, by = c('ratercode.x', 'ratercode.y'))) == 0){
    if(NROW(inner_join(dummy2, b, by = c('ratercode.x', 'ratercode.y'))) == 0){
      dummy2 = rbind(dummy2, a)
    }
  }
}
dummy2 = dummy2[dummy2$ratercode.x != dummy2$ratercode.y,]

a = inner_join(joined, dummy2, by = c('ratercode.x', 'ratercode.y'))
a[a$chimpcode=='K262', ]

pairwise_counts <- a %>% 
  count(ratercode.x, ratercode.y)

pairwise_counts$pair <- paste0("[",
                               pairwise_counts$ratercode.x,
                               ", ",
                               pairwise_counts$ratercode.y,
                               "]")
pairwise_counts <- pairwise_counts[order(pairwise_counts$n, decreasing = T),]
pairwise_counts$pair <- factor(pairwise_counts$pair, levels = pairwise_counts$pair)
ggplot(pairwise_counts, aes(pair, n)) + 
  geom_bar(stat = 'identity') + 
  theme_bw(base_size = 16) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

sum(pairwise_counts$n > 10)

