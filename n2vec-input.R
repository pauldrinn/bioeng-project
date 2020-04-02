library(tidygraph)
library(dplyr)

network_table <- read.table('fi.txt', header = FALSE)
networkasgraph <- as_tbl_graph(network_table)

networkasgraph %>%
  activate(edges) %>%
  write.table('numerical_edge_list.csv', row.names = FALSE, col.names = FALSE)
