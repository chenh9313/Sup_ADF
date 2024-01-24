#! /bin/R
#load needed library
library(gridExtra); library(pheatmap)

rm(list = ls())
# Initialize a list to store the results
data1 <- read.table("Rawdata-Cluster-heatmap/Cluster1_TPM.txt")
a <- list(pheatmap(data1,main = "Cluster 1",cluster_rows = TRUE,cluster_cols = FALSE,col = colorRampPalette(c("blue", "white", "red"))(256),
                   scale = "row",fontsize_row = 4,fontsize_col = 4,show_rownames = FALSE)[[4]])

# Define the clusters you want to iterate over
clusters <- c(seq(2,33))

# Set up the for loop
for (cluster in clusters) {
  data <- read.table(paste("Rawdata-Cluster-heatmap/Cluster",cluster,"_TPM.txt", sep=""))
  # Create the pheatmap and save the result to the list
  a[[cluster]] <- pheatmap(
    data,
    main = paste("Cluster", cluster),
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    col = colorRampPalette(c("blue", "white", "red"))(256),
    scale = "row",
    fontsize_row = 4,
    fontsize_col = 4,
    show_rownames = FALSE
  )[[4]]
}

# Print the list
print(a)

#generate figure
z <- do.call(grid.arrange,a)
plot(z)

