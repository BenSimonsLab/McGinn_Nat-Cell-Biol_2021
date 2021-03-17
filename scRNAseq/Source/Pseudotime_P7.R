##### Code for calculating expression profiles of Yap and Klf4-related genes for resting basal and differentiated cells at P7 along pseudotime  
##### Author: Seungmin Han (sh906@cam.ac.uk)
##### Last Update: 15/03/2021

library(scater)
library(DropletUtils)
library(scran)
library(Seurat)
library(monocle)
library(pheatmap)
library(zoo)
library(reshape2)

result_path <-"../data"
setwd(result_path)
load("pn_adult_sce.RData")

# Calculating DE genes for resing basal cells and differentiated cells at P7
tmp_idx <- colData(pn_adult_sce)$celltype %in% c('resting_basal_P7', 'differentiated_P7')
tmp_sce <- pn_adult_sce[, tmp_idx]
my.clusters <- colData(tmp_sce)$celltype
markers <- findMarkers(tmp_sce, my.clusters, direction="any")
rm(tmp_sce)
cluster_of_interest <- c("resting_basal_P7")
marker.set <- markers[[cluster_of_interest]] 
log2fc_thr <- unname( quantile(abs( marker.set@listData$logFC.differentiated_P7 ), 0.95) )
tmp_idx <- ( marker.set@listData$FDR < 0.05 ) & (abs( marker.set@listData$logFC.differentiated_P7 ) > log2fc_thr)
DEG_RbasalDiff <- marker.set@rownames[tmp_idx]

# Pseudotime anlaysis using monocle
tmp_idx <- colData(pn_adult_sce)$celltype %in% c('resting_basal_P7', 'differentiated_P7')
tmp_sce <- pn_adult_sce[, tmp_idx]
rowData(tmp_sce)$gene_short_name <- rownames(tmp_sce)
my_colData <- colnames( colData(tmp_sce) )
PND_cds_RBDiff_P7 <- convertTo( tmp_sce, type=("monocle"), row.fields=c("Symbol", "ID", "gene_short_name"), col.fields=my_colData )
rm(tmp_sce)
PND_cds_RBDiff_P7 <- estimateDispersions( PND_cds_RBDiff_P7 )
ordering_genes <- DEG_RbasalDiff
PND_cds_RBDiff_P7 <- setOrderingFilter(PND_cds_RBDiff_P7, ordering_genes)
PND_cds_RBDiff_P7 <- reduceDimension(PND_cds_RBDiff_P7, max_components = 2, reduction_method = 'DDRTree')
PND_cds_RBDiff_P7 <- orderCells(PND_cds_RBDiff_P7)
pData(PND_cds_RBDiff_P7)$sample <- pData(PND_cds_RBDiff_P7)$samplename_batch
pData(PND_cds_RBDiff_P7)$samplename_batch <- NULL
pData(PND_cds_RBDiff_P7)$sample_name <- NULL
plot_cell_trajectory(PND_cds_RBDiff_P7, color_by = "State")
plot_cell_trajectory(PND_cds_RBDiff_P7, color_by = "cluster")
plot_cell_trajectory(PND_cds_RBDiff_P7, color_by = "Pseudotime")
PND_cds_RBDiff_P7 <- orderCells(PND_cds_RBDiff_P7, root_state = '3')
plot_cell_trajectory(PND_cds_RBDiff_P7, color_by = "Pseudotime")
PStime_RBDiff_P7 <- pData(PND_cds_RBDiff_P7)$Pseudotime

# Scaling
tmp_idx <- colData(pn_adult_sce)$celltype %in% c('resting_basal_P7', 'resting_basal_P28Adult', 'differentiated_P7', 'differentiated_P28Adult')
RBDiff_sce <- pn_adult_sce[, tmp_idx]
RBDiff_sce$cluster <- as.character(RBDiff_sce$cluster)
RBDiff_Seu <- as.Seurat(RBDiff_sce, counts = "counts", data = "logcounts")
all.genes <- rownames(RBDiff_Seu)
RBDiff_Seu <- ScaleData(RBDiff_Seu, features = all.genes)
tmp_idx <- RBDiff_Seu@meta.data$celltype %in% c('resting_basal_P7', 'differentiated_P7')
RBDiff_P7_Seu <- RBDiff_Seu[, tmp_idx]
RBDiff_P7_Seu@meta.data$Pseudotime <- PStime_RBDiff_P7
names(PStime_RBDiff_P7) <- colnames(RBDiff_P7_Seu)
cd <- GetAssayData(object = RBDiff_P7_Seu, slot = "scale.data")
cd <- rbind(PStime_RBDiff_P7, cd)
cd <- cd[, order(cd['PStime_RBDiff_P7',])]
cd <- cd[-1,]
cd <- t( scale( t(cd), center = TRUE, scale = FALSE ) )
PStime_RBDiff_P7 <- PStime_RBDiff_P7[order(PStime_RBDiff_P7)]

# Drawing expression profiles along pseudotime order
features.plot <- c("Cyr61", "Ctgf", "Thbs1", "Cav1", "Klf2", "Dcn") # YAP and Mechanics-related genes
# features.plot <- c("Klf4", "Krt4", "Krt13", "Cdkn1a", "Cebpb") # Klf4-related genes
features.plot <- intersect( features.plot, rownames(RBDiff_P7_Seu) )
features.plot
TSmat = matrix(data=NA, nrow=length(features.plot), 
               ncol=length(colnames(cd[,names(PStime_RBDiff_P7)])), 
               dimnames = list(features.plot, colnames(cd[,names(PStime_RBDiff_P7)])))
rownames(TSmat) = features.plot

for(i in features.plot){
  y = cd[i,names(PStime_RBDiff_P7)[order(PStime_RBDiff_P7)]]
  Y_hat = rollapply(y,242, mean, align = 'center',fill = 'extend')
  TSmat[i,] = Y_hat
}

df.TSmat = data.frame(t(TSmat), x = 1:length(colnames(cd[,names(PStime_RBDiff_P7)])))
df.TSmat.long <- melt(df.TSmat, id=c("x"))

plt_1 <- ggplot(data = df.TSmat.long, aes(x=x, y=value, color=variable)) +
  geom_line() +
  ylab("Auto exprs") +
  xlab("Cell order") +
  theme_bw() +
  ylim(-1, 1) +
  theme(text = element_text(size=20),
        aspect.ratio = 0.75,
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=1),
        axis.ticks=element_line(size=1),
        legend.position = "right",
        legend.text=element_text(size=10), 
        legend.title=element_blank(),
        legend.key=element_blank(),
        axis.text.x = element_text(size=20)
  )

plot(plt_1)


