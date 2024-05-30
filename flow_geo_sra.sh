#! /bin/sh

#Step1: Find papers which are related to my reserach goal; get the GSE ID which is linked to Bioproject
esearch -db gds -query "cold response" | efilter -query '"Arabidopsis thaliana"[orgn]' | efetch -format txt | grep Series | awk '{print $4}' | sort | uniq | grep GSE > GSE_id.txt

#Step2: Use GSE ID to find BioProject ID
for i in `cat GSE_id.txt`; do echo $i; esearch -db bioproject -query ${i} | efetch -format txt; done | grep BioProject | sort | uniq | awk '{print $3}' > BioPro_id.txt

#Step3: Download GEO Data from a BioProject Accession; get the FTP link
for i in `cat BioPro_id.txt`; do echo $i; esearch -db gds -query "${i}[ACCN]" | efetch -format docsum | xtract -pattern DocumentSummary -element FTPLink; done

#Step4: Use BioProject to get all the sra under each BioProject
for i in `cat BioPro_id.txt`; do echo $i; esearch -db bioproject -query ${i} | elink -target sra | efetch -format docsum | xtract -pattern DocumentSummary -ACC @acc -block DocumentSummary -element "&ACC"> ${i}_sra.txt; done

#Step5: get sample info of RNA-Seq or not
esearch -db bioproject -query PRJNA538858 | elink -target sra | efetch -format docsum | xtract -pattern DocumentSummary -ACC @acc -element Bioproject,Biosample,"&ACC",Title,Platform,LIBRARY_CONSTRUCTION_PROTOCOL

#Step6: some have wrong Project name, thsu I have to re-extra the new project name
esearch -db bioproject -query PRJNA294785 | elink -target bioproject | efetch -format docsum | xtract -pattern DocumentSummary -element Project_Acc
