# For Specifically RA protein interaction network for drug treated samples from  the micraoarry data from the GEO database 
#steps followed were - Loading the cel files ,QC,normlaization 
#Loading libraries 

library(oligo)  
library(multtest)
library(limma)
library(pheatmap)


Affy_chip = "hgu133a"
gse_current = "GSE55457"

# covdesc - meta data information  
covdesc <- read.delim("/home/ibab/SEM4_PROJECT/data_analysis_files/covdesc_limma.txt", sep="\t", header=T)

# input and output file directories.
infile_dir = "/home/ibab/SEM4_PROJECT/data_analysis_files/GSE55457/GSE55457_RAW"
outfile_dir = "/home/ibab/SEM4_PROJECT/data_analysis_files/GSE55457/outfile_dir/output_files"
outfile_name = paste("outfile_dir/", gse_current,"_limma_output.csv", sep="") 

test_between = c("Normal","Rheumatoid_arthritis") ## As in covdesc file

## Decide which files are control and which are treatment.
control_cols = which(covdesc$condition == test_between[1])

treatment_cols = which(covdesc$condition == test_between[2])

if(Affy_chip == "hgu133a")
{
  library(hgu133a.db)
  # library(hgu133acdf) not required for oligo
  library(pd.hg.u133a)
}

if(Affy_chip == "hgu133b")
{
  library(hgu133b.db)
  #library(hgu133bcdf) # not required for oligo
  library(pd.hg.u133b)
}
if(Affy_chip == "hugene11st")
{
  library(pd.hugene.1.1.st.v1)
  library(hugene11stprobeset.db)
  library(hugene11sttranscriptcluster.db) # hugene platforms can be analysed at transcript level or probe level
}

# reading the cell files 
celFiles <- list.celfiles(infile_dir,full.names=T,listGzipped = TRUE)
rawData <- read.celfiles(celFiles)

#Generate featureset 
pData(rawData)$gsm =  rownames(pData(rawData))
pData(rawData)$samples =  covdesc$condition
print(rawData)
xlabs = as.character(covdesc$condition)

#boxplot for outlier checking 
boxplot(rawData,target='core', boxwex = 0.5 ,las =2 ,names=xlabs,cex.axis=0.8)

#normalization for the remove the batcheffects and techincal errors using the rma normalization 
normalized_data = rma(rawData)

#boxplot for the normalized data 
xlabs = as.character(covdesc$condition)
boxplot(normalized_data,target='core', boxwex = 0.5 ,las =2 ,names=xlabs,cex.axis=0.8)

#gene expression value  matrix 
expval = exprs(normalized_data)
prbnames = rownames(expval)
clnames = colnames(expval)

#remove the .cel extension form the sample 

nme =  strsplit(colnames(expval),"\\.")
for (i in 1:ncol(expval)) {
  clnames[i] = strsplit(colnames(expval),"\\.")[[i]][1]
}
colnames(expval) = clnames

#PCA for the variancechecking 

pca_data =  prcomp(t(expval),scaler=T)
color = c("olivedrab","goldenrod")
plot(pca_data$x,col=color,type="p",pch=19)

#create a design matrix for anova 
groups = pData(rawData)$sample 
f = factor(groups,levels = test_between)
design = model.matrix(~ 0 + f )
colnames(design) = test_between

#lmfit () calculates mena expression levels in each group 

data_fit = lmFit(expval,design)

#make contrast matrix to show which isnormal and which is RA 

contrast.matrix =  makeContrasts(Normal-Rheumatoid_arthritis,levels=design)
data_fit.cont = contrasts.fit(data_fit,contrast.matrix)

#moderate t-test on limma eBayes 
data_fit.eb = eBayes(data_fit.cont)

#volcano plot 
volcanoplot(data.fit.eb ,coef = 1,col="cornflowerblue",main="volcano plot using Limma")

#multiple hypothesis correction
tab = topTable(data_fit.eb,coef = 1,number = nrow(normalized_data),adjust.method = "BH",sort.by = "P")

#store in the file 
output_limma = tab[order(match(rownames(tab),rownames(fData(normalized_data)))), ,drop=FALSE]
 
 #Arithmetic mean and foldchange 
control_samples =length(control_cols)
treatment_samples = length(treatment_cols)
#
control_data =  rep(0,nrow(expval))
treatment_data =  rep(0,nrow(expval))

# Summing all columns of lean
for(i in control_cols)
{
  control_data = control_data + 2^expval[,i]
}

# summing all columns of obese
for(i in treatment_cols)
{
  treatment_data = treatment_data + 2^expval[,i]
}
control_linear_mean = (control_data/control_samples)
treatment_linear_mean = (treatment_data/treatment_samples)

# Fold change is oa column divided by ra column. We get fold change for all genes.
FC_linear = as.vector(treatment_linear_mean/control_linear_mean)


FC_log2 = log2(FC_linear)

FC_linear = round(FC_linear, 4)
FC_log2 = round(FC_log2,4)

hist(FC_log2, breaks=60, xlim=c(-2,2))

# geomaetric mean

## control.g and treatment.g are mean(log(x))
control.g = apply(expval[, control_cols], 1, mean) # passing 1 means the mean will be calculated along rows)
treatment.g = apply(expval[, treatment_cols], 1, mean) # passing 1 means the mean will be calculated along rows)

## geometric mean is 2^( mean(log(x)) )
control.g.mean = 2^control.g
treatment.g.mean = 2^treatment.g

# Fold change of geometric mean is treatment column divided by control column. We get fold change for all genes.
FC.g_linear = as.vector(treatment.g.mean/control.g.mean)

FC.g_log2 = log2(FC.g_linear)

FC.g_linear = round(FC.g_linear, 4)
FC.g_log2 = round(FC.g_log2,4)
plot(FC_linear, FC.g_linear, xlim=c(0,10), ylim=c(0,10))

#annotation 

annot <- data.frame(ENTREZID=sapply(contents(hgu133aENTREZID),
                                    paste, collapse=", "),
                    ACCNUM=sapply(contents(hgu133aACCNUM), paste, collapse=", "), SYMBOL=sapply(contents(hgu133plus2SYMBOL), 
                                                                                                paste, collapse=","), 
                    DESC=sapply(contents(hgu133aGENENAME), paste, collapse=", "), 
                    ACHROMOSOME=sapply(contents(hgu133aCHR), paste, collapse=", "))
 

fData(normalized_data) = annot[match(rownames(annot),rownames(fData(normalized_data))),]

output_limma = tab[order(match(rownames(tab),rownames(fData(normalized_data)))), ,drop=FALSE]

df = data.frame(prbnames,fData(normalized_data),output_limma,expval)

#differential expression of genes upregulated and downregulated  

df$DE = "NO"

df$DE[df$logFC> 1 & df$adj.P.Val < 0.1] = 'UP'

df$DE[df$logFC < -1 & df$adj.P.Val < 0.1] = 'DOWN'

upregulted_genes = subset(df,df$logFC >1 & df$adj.P.Val < 0.1) 
downregulated_genes = subset(df,df$logFC < -1 & df$adj.P.Val < 0.1)
diff_expressed_genes = rbind(upregulted_genes,downregulated_genes)

dim(diff_expressed_genes)

#NORMALIZATION OF DIFFERENTIALLY EXPRESSED GENES 

remove_cols = c("prbnames", "ENTREZID" ,"ACCNUM",  "DESC"   ,  "ACHROMOSOME"  ,"logFC" ,"AveExpr", "t",  "P.Value" , "adj.P.Val","B")  

heatmap_df = diff_expressed_genes[,!names(diff_expressed_genes) %in% remove_cols]
heatmap_df <- heatmap_df[, sapply(heatmap_df, is.numeric)]

heatmap_matrix = as.matrix(heatmap_df)

# Scale each row (gene-wise normalization)
heatmap_scaled <- t(scale(t(heatmap_matrix)))

# Replace NA values with 0 (optional, to avoid errors)
heatmap_scaled[is.na(heatmap_scaled)] <- 0
pheatmap(heatmap_scaled,  cluster_rows = TRUE,  
         cluster_cols = TRUE,  
         show_rownames = TRUE,  
    
         fontsize_row = 6,  
         width = 10, height = 18,  # Increase plot size
         color = colorRampPalette(c("green", "white", "red"))(50),  )

###################################################################################

#saperate the drug-treated RA asmples from the untreated  samples and reanalysis 
covdesc_df =  data.frame(covdesc) 

#mixed samples -Drug treated RA samples 
RA_WITH_DRUG = c("GSM1337324" ,'GSM1337315','GSM1337318','GSM1337326')

#RA samples without drug 
RA_WO_DRUG = covdesc_df[!covdesc_df$samples %in% RA_WITH_DRUG, ]  

#RA samples with drug 
RA_WITH_DRUG_SAMPLES = covdesc_df[covdesc_df$samples %in% RA_WITH_DRUG |  covdesc_df$condition == "Normal", ]

#reanalyse these samples RA_WO_DRUG

celFiles_all= list.celfiles(infile_dir,full.names=T,listGzipped = TRUE)
sample_names = basename(celFiles_all)  # Get only file names
sample_names = gsub("\\.CEL(\\.GZ)?$", "", sample_names, ignore.case = TRUE)  #

# Remove everything after the first underscore (_)
clean_sample_names <- sub("_.*", "", sample_names)  

# Print cleaned names for debugging
print("Cleaned Sample Names:")
print(clean_sample_names)


filtered_celFiles = celFiles_all[clean_sample_names %in% RA_WO_DRUG$samples]

rawdata = read.celfiles(filtered_celFiles)

pData(rawdata)$gsm <- rownames(pData(rawdata))    #it will add column tothe rawdata as gsm and store about the sample info 
pData(rawdata)$samples <-RA_WO_DRUG$condition

xlabs = RA_WO_DRUG$condition 
boxplot(rawdata,target="core",boxwex=0.4,names=xlabs,las=2,cex.axis=0.7,main='BoxPlot of RA samples wihout Drug')

normalized_data = rma(rawdata)

#boxplot for the normalized data 
xlabs = as.character(RA_WO_DRUG$condition)
boxplot(normalized_data,target='core', boxwex = 0.5 ,las =2 ,names=xlabs,cex.axis=0.8)

#gene expression value  matrix 
expval_RA = exprs(normalized_data)
prbnames = rownames(expval_RA)
clnames = colnames(expval_RA)

#remove the .cel extension form the sample 

nme =  strsplit(colnames(expval_RA),"\\.")
for (i in 1:ncol(expval_RA)) {
  clnames[i] = strsplit(colnames(expval_RA),"\\.")[[i]][1]
}
colnames(expval_RA) = clnames

#PCA for the variancechecking 

pca_data_RA =  prcomp(t(expval_RA),scaler=T)
color = c("olivedrab","goldenrod")

plot(pca_data_RA$x,col=color,type="p",pch=19)

#create a design matrix for anova 
groups = pData(rawdata)$sample 

f = factor(groups,levels = test_between)
design = model.matrix(~ 0 + f )
colnames(design) = test_between

#lmfit () calculates mena expression levels in each group 

data_fit_RA = lmFit(expval_RA,design)

#make contrast matrix to show which isnormal and which is RA 

contrast.matrix =  makeContrasts(Normal-Rheumatoid_arthritis,levels=design)
data_fit.cont_RA = contrasts.fit(data_fit_RA,contrast.matrix)

#moderate t-test on limma eBayes 
data_fit.eb_RA = eBayes(data_fit.cont_RA)

#volcano plot 
volcanoplot(data.fit.eb ,coef = 1,col="cornflowerblue",main="volcano plot using Limma")

#multiple hypothesis correction
tab = topTable(data_fit.eb_RA,coef = 1,number = nrow(normalized_data),adjust.method = "BH",sort.by = "P")

#store in the file 
output_limma_RA_WO_DRUG = tab[order(match(rownames(tab),rownames(fData(normalized_data)))), ,drop=FALSE]


control_cols_RA = which(RA_WO_DRUG$condition == test_between[1])

treatment_cols_RA = which(RA_WO_DRUG$condition == test_between[2])
#Arithmetic mean and foldchange 
control_samples_WO_DRUG =length(control_cols_RA)
treatment_samples_WO_DRUG = length(treatment_cols_RA)
#
control_data_RA =  rep(0,nrow(expval_RA))
treatment_data_RA =  rep(0,nrow(expval_RA))

# Summing all columns of lean
for(i in control_cols_RA)
{
  control_data_RA = control_data_RA + 2^expval_RA[,i]
}

# summing all columns of obese
for(i in treatment_cols_RA)
{
  treatment_data_RA = treatment_data_RA + 2^expval_RA[,i]
}
control_linear_mean_RA = (control_data_RA/control_samples_WO_DRUG)
treatment_linear_mean_RA = (treatment_data_RA/treatment_samples_WO_DRUG)

# Fold change is oa column divided by ra column. We get fold change for all genes.
FC_linear = as.vector(treatment_linear_mean_RA/control_linear_mean_RA)


FC_log2 = log2(FC_linear)

FC_linear = round(FC_linear, 4)
FC_log2 = round(FC_log2,4)

hist(FC_log2, breaks=60, xlim=c(-2,2))

# geomaetric mean

## control.g and treatment.g are mean(log(x))
control.g = apply(expval_RA[, control_cols_RA], 1, mean) # passing 1 means the mean will be calculated along rows)
treatment.g = apply(expval_RA[, treatment_cols_RA], 1, mean) # passing 1 means the mean will be calculated along rows)

## geometric mean is 2^( mean(log(x)) )
control.g.mean = 2^control.g
treatment.g.mean = 2^treatment.g

# Fold change of geometric mean is treatment column divided by control column. We get fold change for all genes.
FC.g_linear = as.vector(treatment.g.mean/control.g.mean)

FC.g_log2 = log2(FC.g_linear)

FC.g_linear = round(FC.g_linear, 4)
FC.g_log2 = round(FC.g_log2,4)
plot(FC_linear, FC.g_linear, xlim=c(0,10), ylim=c(0,10))

#annotation 

annot <- data.frame(ENTREZID=sapply(contents(hgu133aENTREZID),
                                    paste, collapse=", "),
                    ACCNUM=sapply(contents(hgu133aACCNUM), paste, collapse=", "), SYMBOL=sapply(contents(hgu133plus2SYMBOL), 
                                                                                                paste, collapse=","), 
                    DESC=sapply(contents(hgu133aGENENAME), paste, collapse=", "), 
                    ACHROMOSOME=sapply(contents(hgu133aCHR), paste, collapse=", "))


fData(normalized_data) = annot[match(rownames(annot),rownames(fData(normalized_data))),]

output_limma_RA_WO_DRUG = tab[order(match(rownames(tab),rownames(fData(normalized_data)))), ,drop=FALSE]

df_RA = data.frame(prbnames,fData(normalized_data),output_limma,expval_RA)

#differential expression of genes upregulated and downregulated  

df_RA$DE = "NO"

df_RA$DE[df_RA$logFC> 1 & df_RA$adj.P.Val < 0.1] = 'UP'

df_RA$DE[df_RA$logFC < -1 & df_RA$adj.P.Val < 0.1] = 'DOWN'

upregulted_genes = subset(df_RA,df_RA$logFC >1 & df_RA$adj.P.Val < 0.1) 
downregulated_genes = subset(df_RA,df_RA$logFC < -1 & df_RA$adj.P.Val < 0.1)
diff_expressed_genes_RA = rbind(upregulted_genes,downregulated_genes)

dim(diff_expressed_genes_RA)

write.table(diff_expressed_genes_RA ,file = "DIFFERENTIAL_EXPRESSED_GENES_RA_WO_DRUG.csv",sep=",") 
#NORMALIZATION OF DIFFERENTIALLY EXPRESSED GENES 

remove_cols = c("prbnames", "ENTREZID" ,"ACCNUM",  "DESC"   ,  "ACHROMOSOME"  ,"logFC" ,"AveExpr", "t",  "P.Value" , "adj.P.Val","B")  

heatmap_df = diff_expressed_genes_RA[,!names(diff_expressed_genes_RA) %in% remove_cols]
heatmap_df <- heatmap_df[, sapply(heatmap_df, is.numeric)]

heatmap_matrix = as.matrix(heatmap_df)

# Scale each row (gene-wise normalization)
heatmap_scaled <- t(scale(t(heatmap_matrix)))

# Replace NA values with 0 (optional, to avoid errors)
heatmap_scaled[is.na(heatmap_scaled)] <- 0
library(pheatmap)

pheatmap(heatmap_scaled,  cluster_rows = TRUE,  
         cluster_cols = TRUE,  
         show_rownames = TRUE,  
         
         fontsize_row = 6,  
         width = 10, height = 18,  # Increase plot size
         color = colorRampPalette(c("green", "white", "red"))(50), main='Heatmap of DEG for the RA wothout Drug samples ' )


####################################################################### 
library(multtest)
library(limma)

#RA samples with drug 
RA_WITH_DRUG_SAMPLES = covdesc_df[covdesc_df$samples %in% RA_WITH_DRUG |  covdesc_df$condition == "Normal", ]

#reanalyse these samples RA_WO_DRUG

# Load all CEL files
celFiles_all = list.celfiles(infile_dir, full.names = TRUE, listGzipped = TRUE)

# Extract sample names from file paths
sample_names_WITH_DRUG = basename(celFiles_all)  # Get only file names
sample_names_WITH_DRUG = gsub("\\.CEL(\\.GZ)?$", "", sample_names, ignore.case = TRUE)  # Remove extensions

# Clean sample names by removing everything after the first underscore
clean_sample_names <- sub("_.*", "", sample_names)

# Debugging: Print cleaned names
print("Cleaned Sample Names:")
print(clean_sample_names)

# Ensure RA_WITH_DRUG_SAMPLES$sample names are in the same format
RA_WITH_DRUG_GSM = RA_WITH_DRUG_SAMPLES$samples

# Debugging: Print RA_WITH_DRUG GSM IDs
print("RA_WITH_DRUG Samples:")
print(RA_WITH_DRUG_GSM)

# Filter CEL files
filtered_celFiles = celFiles_all[clean_sample_names %in% RA_WITH_DRUG_GSM]

rawdat = read.celfiles(filtered_celFiles)

pData(rawdat)$gsm <- rownames(pData(rawdat))    #it will add column tothe rawdata as gsm and store about the sample info 
pData(rawdat)$samples <-RA_WITH_DRUG_SAMPLES$condition

xlabs = RA_WITH_DRUG_SAMPLES$condition 
boxplot(rawdat,target="core",boxwex=0.4,names=xlabs,las=2,cex.axis=0.7,main='BoxPlot of RA samples with Drug')

normalized_data_RA = rma(rawdat)

#boxplot for the normalized data 
xlabs = as.character(RA_WITH_DRUG_SAMPLES$condition)
boxplot(normalized_data_RA,target='core', boxwex = 0.5 ,las =2 ,names=xlabs,cex.axis=0.8)

#gene expression value  matrix 
expval_RA_WITH_DRUG  = exprs(normalized_data_RA)
prbnames = rownames(expval_RA_WITH_DRUG)
clnames = colnames(expval_RA_WITH_DRUG)
#remove the .cel extension form the sample 

nme =  strsplit(colnames(expval_RA_WITH_DRUG),"\\.")
for (i in 1:ncol(expval_RA_WITH_DRUG)) {
  clnames[i] = strsplit(colnames(expval_RA_WITH_DRUG),"\\.")[[i]][1]
}
colnames(expval_RA_WITH_DRUG) = clnames

#PCA for the variancechecking 

pca_data_RA_WITH_DRUG =  prcomp(t(expval_RA_WITH_DRUG),scaler=T)
color = c("olivedrab","goldenrod")

plot(pca_data_RA_WITH_DRUG$x,col=color,type="p",pch=19)

#create a design matrix for anova 
groups = pData(rawdat)$sample 

f = factor(groups,levels = test_between)
design = model.matrix(~ 0 + f )
colnames(design) = test_between

#lmfit () calculates mena expression levels in each group 

data_fit_RA_WITH_DRUG = lmFit(expval_RA_WITH_DRUG,design)

#make contrast matrix to show which isnormal and which is RA 

contrast.matrix =  makeContrasts(Normal-Rheumatoid_arthritis,levels=design)
data_fit.cont_RA_WITH_DRUG = contrasts.fit(data_fit_RA_WITH_DRUG,contrast.matrix)

#moderate t-test on limma eBayes 
data_fit.eb_RA_WITH_DRUG = eBayes(data_fit.cont_RA_WITH_DRUG)

#volcano plot 
volcanoplot(data.fit.eb ,coef = 1,col="cornflowerblue",main="volcano plot using Limma")

#multiple hypothesis correction
tab_RA = topTable(data_fit.eb_RA_WITH_DRUG,coef = 1,number = nrow(normalized_data),adjust.method = "BH",sort.by = "P")

#store in the file 
output_limma_RA_WITH_DRUG = tab[order(match(rownames(tab_RA),rownames(fData(normalized_data)))), ,drop=FALSE]


control_cols_RA = which(RA_WITH_DRUG_SAMPLES$condition == test_between[1])

treatment_cols_RA = which(RA_WITH_DRUG_SAMPLES$condition == test_between[2])
#Arithmetic mean and foldchange 
control_samples_WITH_DRUG =length(control_cols_RA)
treatment_samples_WITH_DRUG = length(treatment_cols_RA)
#
control_data_RA =  rep(0,nrow(expval_RA_WITH_DRUG))
treatment_data_RA =  rep(0,nrow(expval_RA_WITH_DRUG))

# Summing all columns of lean
for(i in control_cols_RA)
{
  control_data_RA = control_data_RA + 2^expval_RA_WITH_DRUG[,i]
}

# summing all columns of obese
for(i in treatment_cols_RA)
{
  treatment_data_RA = treatment_data_RA + 2^expval_RA_WITH_DRUG[,i]
}
control_linear_mean_RA = (control_data_RA/control_samples_WITH_DRUG)
treatment_linear_mean_RA = (treatment_data_RA/treatment_samples_WITH_DRUG)

# Fold change is oa column divided by ra column. We get fold change for all genes.
FC_linear = as.vector(treatment_linear_mean_RA/control_linear_mean_RA)


FC_log2 = log2(FC_linear)

FC_linear = round(FC_linear, 4)
FC_log2 = round(FC_log2,4)

hist(FC_log2, breaks=60, xlim=c(-2,2))

# geomaetric mean

## control.g and treatment.g are mean(log(x))
control.g = apply(expval_RA_WITH_DRUG[, control_cols_RA], 1, mean) # passing 1 means the mean will be calculated along rows)
treatment.g = apply(expval_RA_WITH_DRUG[, treatment_cols_RA], 1, mean) # passing 1 means the mean will be calculated along rows)

## geometric mean is 2^( mean(log(x)) )
control.g.mean = 2^control.g
treatment.g.mean = 2^treatment.g

# Fold change of geometric mean is treatment column divided by control column. We get fold change for all genes.
FC.g_linear = as.vector(treatment.g.mean/control.g.mean)

FC.g_log2 = log2(FC.g_linear)

FC.g_linear = round(FC.g_linear, 4)
FC.g_log2 = round(FC.g_log2,4)
plot(FC_linear, FC.g_linear, xlim=c(0,10), ylim=c(0,10))

#annotation 

annot <- data.frame(ENTREZID=sapply(contents(hgu133aENTREZID),
                                    paste, collapse=", "),
                    ACCNUM=sapply(contents(hgu133aACCNUM), paste, collapse=", "), SYMBOL=sapply(contents(hgu133plus2SYMBOL), 
                                                                                                paste, collapse=","), 
                    DESC=sapply(contents(hgu133aGENENAME), paste, collapse=", "), 
                    ACHROMOSOME=sapply(contents(hgu133aCHR), paste, collapse=", "))


fData(normalized_data) = annot[match(rownames(annot),rownames(fData(normalized_data))),]

output_limma_RA_WITH_DRUG = tab[order(match(rownames(tab),rownames(fData(normalized_data)))), ,drop=FALSE]

df_RA_WITH_DRUG = data.frame(prbnames,fData(normalized_data),output_limma_RA_WITH_DRUG,expval_RA_WITH_DRUG)

#differential expression of genes upregulated and downregulated  

df_RA$DE = "NO"

df_RA$DE[df_RA$logFC> 1 & df_RA$adj.P.Val < 0.1] = 'UP'

df_RA$DE[df_RA$logFC < -1 & df_RA$adj.P.Val < 0.1] = 'DOWN'

upregulted_genes = subset(df_RA_WITH_DRUG,df_RA_WITH_DRUG$logFC >1 & df_RA_WITH_DRUG$adj.P.Val < 0.1) 
downregulated_genes = subset(df_RA_WITH_DRUG,df_RA_WITH_DRUG$logFC < -1 & df_RA_WITH_DRUG$adj.P.Val < 0.1)
diff_expressed_genes_RA_WITH_DRUG = rbind(upregulted_genes,downregulated_genes)

dim(diff_expressed_genes_RA_WITH_DRUG)

write.table(diff_expressed_genes_RA_WITH_DRUG ,file = "DIFFERENTIAL_EXPRESSED_GENES_RA_WITH_DRUG.csv",sep=",",row.names = FALSE) 
#NORMALIZATION OF DIFFERENTIALLY EXPRESSED GENES 

remove_cols = c("prbnames", "ENTREZID" ,"ACCNUM",  "DESC"   ,  "ACHROMOSOME"  ,"logFC" ,"AveExpr", "t",  "P.Value" , "adj.P.Val","B")  

heatmap_df_RA = diff_expressed_genes_RA_WITH_DRUG[,!names(diff_expressed_genes_RA_WITH_DRUG) %in% remove_cols]
heatmap_df_RA <- heatmap_df_RA[, sapply(heatmap_df_RA, is.numeric)]

heatmap_matrix_RA = as.matrix(heatmap_df_RA)

# Scale each row (gene-wise normalization)
heatmap_scaled_RA <- t(scale(t(heatmap_matrix_RA)))

# Replace NA values with 0 (optional, to avoid errors)
heatmap_scaled_RA[is.na(heatmap_scaled_RA)] <- 0
pheatmap(heatmap_scaled_RA,  cluster_rows = TRUE,  
         cluster_cols = TRUE,  
         show_rownames = TRUE,  
         
         fontsize_row = 6,  
         width = 10, height = 18,  # Increase plot size
         color = colorRampPalette(c("green", "white", "red"))(50), main='Heatmap of DEG for the RA with Drug samples ' )


