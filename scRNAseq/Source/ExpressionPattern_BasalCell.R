##### Clustering gene expression patterns from p7 over p28 to adult samples for all basal cells (cycling + resting) 
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

## Filtering out genes that do not show significant change across p7 to adult stage
tmp_idx <- colData(pn_adult_sce)$cluster %in% c(1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17)
basal_sce <- pn_adult_sce[,tmp_idx]

### DEGs between p7 and p28
tmp_idx <- colData(basal_sce)$condition %in% c("p7","p28")
tmp_sce_p7_28 <- basal_sce[,tmp_idx]
my.clusters <- colData(tmp_sce_p7_28)$condition
markers <- findMarkers(tmp_sce_p7_28, my.clusters, direction="any")
marker.set_p7p28 <- markers[["p28"]] 
tmp_idx1 <- marker.set_p7p28$FDR < 0.05
logFCthr <- quantile(abs(marker.set_p7p28$logFC.p7), 0.95)
tmp_idx2 <- abs( marker.set_p7p28$logFC.p7 ) > logFCthr
tmp_idx <- tmp_idx1 & tmp_idx2
DEG_p7p28 <- rownames(marker.set_p7p28)[tmp_idx]

### DEGs between p7 and adult
tmp_idx <- colData(basal_sce)$condition %in% c("p7","adult")
tmp_sce_p7_adult <- basal_sce[,tmp_idx]
my.clusters <- colData(tmp_sce_p7_adult)$condition
markers <- findMarkers(tmp_sce_p7_adult, my.clusters, direction="any")
marker.set_p7adult <- markers[["adult"]] 
tmp_idx1 <- marker.set_p7adult$FDR < 0.05
logFCthr <- quantile(abs(marker.set_p7adult$logFC.p7), 0.95)
tmp_idx2 <- abs( marker.set_p7adult$logFC.p7 ) > logFCthr
tmp_idx <- tmp_idx1 & tmp_idx2
DEG_p7adult <- rownames(marker.set_p7adult)[tmp_idx]

### DEGs between p28 and adult
tmp_idx <- colData(basal_sce)$condition %in% c("p28","adult")
tmp_sce_p28_adult <- basal_sce[,tmp_idx]
my.clusters <- colData(tmp_sce_p28_adult)$condition
markers <- findMarkers(tmp_sce_p28_adult, my.clusters, direction="any")
marker.set_p28adult <- markers[["adult"]] 
tmp_idx1 <- marker.set_p28adult$FDR < 0.05
logFCthr <- quantile(abs(marker.set_p28adult$logFC.p28), 0.95)
tmp_idx2 <- abs( marker.set_p28adult$logFC.p28 ) > logFCthr
tmp_idx <- tmp_idx1 & tmp_idx2
DEG_p28adult <- rownames(marker.set_p28adult)[tmp_idx]

### Union of DEGs
union_DEG <- union(DEG_p7p28, DEG_p7adult)
union_DEG <- union(union_DEG, DEG_p28adult)

### Preparing auto-scaled data
pn_adult_Seu <- as.Seurat(x = pn_adult_sce, counts = "logcounts", data = "logcounts")
all.genes <- rownames(pn_adult_Seu)
pn_adult_Seu <- ScaleData(pn_adult_Seu, features = all.genes)
tmp_scaledata <- GetAssayData(object = pn_adult_Seu, slot = "scale.data")
tmp_normdata <- assays(pn_adult_sce)$logcounts 

tmp_idx <- colData(pn_adult_sce)$cluster %in% c(1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17)
basal_scaledata <- tmp_scaledata[,tmp_idx]
basal_scaledata <- basal_scaledata[union_DEG,]
basal_normdata <- tmp_normdata[,tmp_idx]
basal_normdata <- basal_normdata[union_DEG,]

### Subsetting to specific cell clusters
tmp_idx <- colData(pn_adult_sce)$cluster %in% c(1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17)
basal_sce <- pn_adult_sce[,tmp_idx]
basal_sce <- basal_sce[union_DEG,]

### Computing average expression matrix for the cell type at each time point
tmp_idx1 <- colData(basal_sce)$condition %in% c("p7")
tmp_idx2 <- colData(basal_sce)$condition %in% c("p28")
tmp_idx3 <- colData(basal_sce)$condition %in% c("adult")

p7_mean <- Matrix::rowMeans( basal_scaledata[,tmp_idx1] )
p28_mean <- Matrix::rowMeans( basal_scaledata[,tmp_idx2] )
adult_mean <- Matrix::rowMeans( basal_scaledata[,tmp_idx3] )
avg_exp <- cbind(p7_mean, p28_mean)
avg_exp <- cbind(avg_exp, adult_mean)
avg_exp_p7_zero <- avg_exp - avg_exp[,1]

p7_mean_norm <- Matrix::rowMeans( basal_normdata[,tmp_idx1] )
p28_mean_norm <- Matrix::rowMeans( basal_normdata[,tmp_idx2] )
adult_mean_norm <- Matrix::rowMeans( basal_normdata[,tmp_idx3] )
avg_exp_norm <- cbind(p7_mean_norm, p28_mean_norm)
avg_exp_norm <- cbind(avg_exp_norm, adult_mean_norm)
avg_exp_norm_p7_zero <- avg_exp_norm - avg_exp_norm[,1]

##  Clustering gene expression patterns
annotation_col = data.frame(row.names = c("p7_mean", "p28_mean", "adult_mean"), condition = c("p7", "p28", "adult"))

ann_colors = list("condition" = c("p7"="#F8766D",
                                  "p28"="#00BA38",
                                  "adult"="#619CFF"
))

set.seed(100)
ph <- pheatmap(avg_exp_p7_zero, 
               color = colorRampPalette(c("#4575B4","white","#D73027"))(100),
               breaks=seq(-1, 1, length.out=101),
               cluster_cols = F, 
               clustering_distance_rows = "correlation",
               show_colnames = F,
               annotation_col = annotation_col,              
               annotation_colors = ann_colors,
               kmeans_k = 10,
               filename = "Heatmap_genecluster_km10_basalcell_logFC.pdf"
)
dev.off()
rowData(basal_sce)["GeneExpPattern"] <- ph$kmeans$cluster

avg_exp_p7_zero <- data.frame(X=rownames(avg_exp_p7_zero), avg_exp_p7_zero, cluster=ph$kmeans$cluster, stringsAsFactors = F)

# Calculating Pattern 1
cluster_id <- c(1)

tmp_idx <- avg_exp_p7_zero$cluster %in% cluster_id
tmp_avg_exp_df <- avg_exp_p7_zero[tmp_idx,]
tmp_mtx <- as.matrix( tmp_avg_exp_df[,2:4] ) 
tmp_mtx <- t(tmp_mtx)
tmp_mtx <- cbind(tmp_mtx, rowMeans(tmp_mtx))
colnames(tmp_mtx) <- c(tmp_avg_exp_df$X, "Means")
tmp_df <- data.frame( time = c(1,2,3), tmp_mtx)

library(reshape2)
melt.tmp_df <- melt(tmp_df, id = c("time"))

p1 <- ggplot(melt.tmp_df, aes(x = time, y = value)) + 
  geom_line(aes(color = variable), size = 1) +
  scale_color_manual( values = c(rep(c("#e7e7e7"), times = 148), "#F00D0D") ) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        aspect.ratio = 1)

plot(p1)

# Calculating Pattern 2
cluster_id <- c(2, 3, 4, 8)

tmp_idx <- avg_exp_p7_zero$cluster %in% cluster_id
tmp_avg_exp_df <- avg_exp_p7_zero[tmp_idx,]
tmp_mtx <- as.matrix( tmp_avg_exp_df[,2:4] ) 
tmp_mtx <- t(tmp_mtx)
tmp_mtx <- cbind(tmp_mtx, rowMeans(tmp_mtx))
colnames(tmp_mtx) <- c(tmp_avg_exp_df$X, "Means")
tmp_df <- data.frame( time = c(1,2,3), tmp_mtx)
library(reshape2)
melt.tmp_df <- melt(tmp_df, id = c("time"))

p2 <- ggplot(melt.tmp_df, aes(x = time, y = value)) + 
  geom_line(aes(color = variable), size = 1) +
  scale_color_manual( values = c(rep(c("#e7e7e7"), times = 545), "#F00D0D") ) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        aspect.ratio = 1)

plot(p2)

# Calculating Pattern 3
cluster_id <- c(7)
tmp_idx <- avg_exp_p7_zero$cluster %in% cluster_id
tmp_avg_exp_df <- avg_exp_p7_zero[tmp_idx,]
tmp_mtx <- as.matrix( tmp_avg_exp_df[,2:4] ) 
tmp_mtx <- t(tmp_mtx)
tmp_mtx <- cbind(tmp_mtx, rowMeans(tmp_mtx))
colnames(tmp_mtx) <- c(tmp_avg_exp_df$X, "Means")
tmp_df <- data.frame( time = c(1,2,3), tmp_mtx)
library(reshape2)
melt.tmp_df <- melt(tmp_df, id = c("time"))

p3 <- ggplot(melt.tmp_df, aes(x = time, y = value)) + 
  geom_line(aes(color = variable), size = 1) +
  scale_color_manual( values = c(rep(c("#e7e7e7"), times = 153), "#F00D0D") ) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        aspect.ratio = 1)

plot(p3)

# Calculating Pattern 4
cluster_id <- c(5,6,9,10)
tmp_idx <- avg_exp_p7_zero$cluster %in% cluster_id
tmp_avg_exp_df <- avg_exp_p7_zero[tmp_idx,]
tmp_mtx <- as.matrix( tmp_avg_exp_df[,2:4] ) 
tmp_mtx <- t(tmp_mtx)
tmp_mtx <- cbind(tmp_mtx, rowMeans(tmp_mtx))
colnames(tmp_mtx) <- c(tmp_avg_exp_df$X, "Means")
tmp_df <- data.frame( time = c(1,2,3), tmp_mtx)
library(reshape2)
melt.tmp_df <- melt(tmp_df, id = c("time"))

p4 <- ggplot(melt.tmp_df, aes(x = time, y = value)) + 
  geom_line(aes(color = variable), size = 1) +
  scale_color_manual( values = c(rep(c("#e7e7e7"), times = 892), "#F00D0D") ) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        aspect.ratio = 1)
plot(p4)

multiplot(p1, p3, p2, p4, cols = 2)

p_all <- list()

p_all[[1]] <- p1
p_all[[2]] <- p2
p_all[[3]] <- p3
p_all[[4]] <- p4

p_all <- cowplot::plot_grid(plotlist = p_all, ncol = 2)

ggsave("Expression_4patterns_basalcell.pdf", p_all)




