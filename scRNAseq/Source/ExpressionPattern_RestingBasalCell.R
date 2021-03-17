##### Clustering gene expression patterns from p7 over p28 to adult samples for resting basal cells 
##### Author: Seungmin Han (sh906@cam.ac.uk)
##### Last Update: 15/03/2021

library(scater)
library(DropletUtils)
library(scran)
library(Seurat)
library(ggplot2)
library(pheatmap)
library(DT)

result_path <-"../data"
setwd(result_path)
load("pn_adult_sce.RData")

tmp_idx <- colData(pn_adult_sce)$cluster %in% c('2', '7', '12')
tmp_sce <- pn_adult_sce[, tmp_idx]
colData(tmp_sce)$cluster <- as.character(colData(tmp_sce)$cluster)
my.clusters <- colData(tmp_sce)$cluster
markers <- findMarkers(tmp_sce, my.clusters, direction="up")

RB_names1 <- names( table(colData(tmp_sce)$cluster) )

pct_thr <- c(0.975)
pv_thr <- c(0.05)

log2fc_all <- vector()
RB_DEG <- vector()

for (RB_name in RB_names1) {
  cluster_of_interest <- RB_name
  marker.set <- markers[[cluster_of_interest]] 
  
  log2fc_all <- c(log2fc_all, marker.set@listData[[4]])
  log2fc_all <- c(log2fc_all, marker.set@listData[[5]])
}

log2fc_thr <- unname( quantile(log2fc_all, pct_thr) )

for (RB_name in RB_names1) {
  cluster_of_interest <- RB_name
  marker.set <- markers[[cluster_of_interest]] 
  
  log2fc_all <- cbind( marker.set@listData[[4]], marker.set@listData[[5]] )
  max_log2fc_all <- rowMaxs(log2fc_all)
  
  tmp_idx <- ( marker.set@listData$FDR < pv_thr ) & (max_log2fc_all > log2fc_thr)
  sum(tmp_idx)
  DEG_RbasalDiff <- marker.set@rownames[tmp_idx]
  RB_DEG <- union(RB_DEG, DEG_RbasalDiff)
}

tmp_idx <- colData(pn_adult_sce)$cluster %in% c('2', '6', '1', '7', '12')
tmp_sce_2 <- pn_adult_sce[, tmp_idx]
colData(tmp_sce_2)$cluster <- as.character(colData(tmp_sce_2)$cluster)
tmp_sce_2 <- tmp_sce_2[RB_DEG,]
tmp_mtx <- t(base::scale(t(assays(tmp_sce_2)$logcounts), center = TRUE, scale = TRUE))

tmp_idx1 <- colData(tmp_sce_2)$cluster %in% c('2', '6')
tmp_idx2 <- colData(tmp_sce_2)$cluster %in% c('1', '7')
tmp_idx3 <- colData(tmp_sce_2)$cluster %in% c('12')

cl2_6_mean <- Matrix::rowMeans( tmp_mtx[,tmp_idx1] )
cl1_7_mean <- Matrix::rowMeans( tmp_mtx[,tmp_idx2] )
cl12_mean <- Matrix::rowMeans( tmp_mtx[,tmp_idx3] )

avg_exp <- cbind(cl2_6_mean, cl1_7_mean)
avg_exp <- cbind(avg_exp, cl12_mean)

ph <- pheatmap(avg_exp, 
               color = colorRampPalette(c("#4575B4","white","#D73027"))(100),
               breaks=seq(-1, 1, length.out=101),
               cluster_cols = F, 
               clustering_distance_rows = "correlation",
               show_colnames = F,
               kmeans_k = 6,
               filename = "Heatmap_genecluster_km6_Rbasal.pdf"
)

exp_df1 <- data.frame(avg_exp, ph$kmeans$cluster)
rownames(exp_df1) <- rownames(avg_exp)
write.csv(exp_df1, file="Rbasal_DEG_GeneExpPattern.csv")
