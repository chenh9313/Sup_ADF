#! /bin/R

rm(list=ls())
library(cluster)

setwd("/mnt/ufs18/nodr/home/chenhua9/2023_possible_WT/All_24123no0TPMsample_Cluster")
# Load the data
data <- read.table("All_TPM_matrix_TPMno0_with_pubPMID_cluster.txt",header = T)
dim(data)
head(data[,1:3])

# Scale the data
data_scaled <- scale(data)
data_scaled[is.na(data_scaled)] = 0
head(data_scaled[,1:5])

# Calculate within-cluster sum of squares for K values ranging from 1 to 10
print("Doing wss")
wss <- sapply(1:100, function(k) kmeans(data_scaled, k)$tot.withinss)
wss
write.table(wss,file="res_allNo0TPM_wss.txt")
# Plot the within-cluster sum of squares against K
pdf(file="res_allNo0TPM_wss_k10.pdf")
plot(1:10, wss, type = "b", xlab = "Number of Clusters (K)", ylab = "Within-cluster Sum of Squares")
dev.off()

# Use the elbow method to estimate the optimal K value
elbow <- function(x, y) {
  k.values <- 1:length(y)
  delta.y <- c(0, diff(y))
  div <- cumsum(delta.y) / sum(delta.y)
  return(k.values[which.max(div >= x)])
}

elbow(0.8, wss)
opt_k <- elbow(0.8, wss)
write.table(opt_k,file="res_allNo0TPM_optimal_K.txt")

pdf(file="res_allNo0TPM_heatmap.pdf")
heatmap(as.matrix(data))
dev.off()

#find the optimal K value the set k=opt_k
# Perform K-means clustering with K=3
kmeans_output <- kmeans(data, centers = opt_k)
write.table(kmeans_output$cluster,file="res_allNo0TPM_cluster_samplelist.txt")


