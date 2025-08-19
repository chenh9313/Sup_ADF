Author: Huan Chen
Last updated: 12/2023

#!/bin/bash --login
########## Define Resources Needed with SBATCH Lines ##########

#SBATCH --account="Your User Name"
#SBATCH --array=0-Int               # replace Int as a specific number; for example: 500 array threads that will run in parallel
#SBATCH --time=4:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=2           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name  Paired-end_fpkm               # you can give your job a name for easier identification (same as -J)
#SBATCH --output=%x_%j.out # output (%x is the --job-name and %j is the job id); there will be 100 output files, one for each thread
 
########## Command Lines to Run ##########

#Step1: give each sample name to get all SRR info
SAMPLE=( "Your sample name list" )

echo "${SLURM_ARRAY_TASK_ID} ; ${SAMPLE[${SLURM_ARRAY_TASK_ID}]}" # iterate throught the list SAMPLE (length of 500 elements)

cd "Your work directory"

module load Trimmomatic/0.38-Java-1.8
module load hisat2/2.1.0
module load SAMtools/1.9
module load Cufflinks/2.2.1

ADAPTER="adapters under Trimmomatic"
REF="path of your Arabidopsis hisat reference file"
RESULT="path where you want to put your fpkm results"
PAIRED="path where you want to put your process file"

mkdir -p ~/.ncbi
echo '/repository/user/main/public/root = "/scratch/standage/sra-cache"' > ~/.ncbi/user-settings.mkfg

time for i in ${SAMPLE[${SLURM_ARRAY_TASK_ID}]};
do mkdir ${i};
cd ${i};
echo $i;
srr="$(esearch -db sra -query $i | efetch -format runinfo | cut -d "," -f 1 | grep SRR)";
echo $srr >> ${i}_SRR_namelist;
sed -i 's/ /\n/g' ${i}_SRR_namelist; 
wc -l ${i}_SRR_namelist | awk '{if($1<1){print "SRR number is Wrong"}}';

#Step2: download fastq file of each SRR
for j in `cat ${i}_SRR_namelist`;
do echo $j;
fasterq-dump --split-files ${j};
done

#Step3: merge all fastq raw files
touch ${i}_1.fastq
touch ${i}_2.fastq
for m in `cat ${i}_SRR_namelist`;
do cat ${i}_1.fastq ${m}_1.fastq > ${i}_1_temp.fastq
mv ${i}_1_temp.fastq ${i}_1.fastq
cat ${i}_2.fastq ${m}_2.fastq > ${i}_2_temp.fastq
mv ${i}_2_temp.fastq ${i}_2.fastq
done
ls -lth *fastq

#Step4: remove each SRR fastq file
for m in `cat ${i}_SRR_namelist`;
do /bin/rm ${m}_?.fastq;
done

#Step5: Mapping -> FPKM
#guess qulity phred score
VAL=$(head -n 40 ${i}_1.fastq | awk '{if(NR%4==0) printf("%s",$0);}' | od -A n -t u1 | awk 'BEGIN{min=100;max=0;}{for(i=1;i<=NF;i++) {if($i>max) max=$i; if($i<min) min=$i;}}END{if(max<=74 && min<59) print "Phred+33"; else if(max>73 && min>=64) print "Phred+64"; else if(min>=59 && min<64 && max>73) print "Solexa+64"; else print "Unknown score";}' | sed 's/Phred+//g' | sed 's/Solexa+//g')

if [ $VAL -eq 33 ] || [ $VAL -eq 64 ]; then echo "$i qulity phred is $VAL"; else echo "$i qulity phred is Wrong"; fi

#Run Trimmomatic
java -jar $EBROOTTRIMMOMATIC/trimmomatic-0.38.jar PE -phred${VAL} \
${i}_1.fastq ${i}_2.fastq \
trimmed-${i}_1.fastq.gz ${i}_1.unpaired.fastq.gz \
trimmed-${i}_2.fastq.gz ${i}_2.unpaired.fastq.gz \
-threads 10 ILLUMINACLIP:${ADAPTER}/TruSeq-All-PE.fa:2:30:10 \
LEADING:20 TRAILING:20 SLIDINGWINDOW:4:20 MINLEN:20
ls -lth *fastq*
/bin/rm ${i}_?.fastq

#Run hisat2
hisat2 --max-intronlen 30000 \
--summary-file ${i}_summary.txt \
-p 10 \
--rna-strandness RF \
-x ${REF}/Arabidopsis_thaliana \
-1 trimmed-${i}_1.fastq.gz \
-2 trimmed-${i}_2.fastq.gz | samtools sort -@ 10 -o ${i}.bam

#Run cufflinks
cufflinks \
-G ${REF}/Arabidopsis_thaliana.TAIR10.28.gff3 \
-b ${REF}/Arabidopsis_thaliana.TAIR10.28.dna.genome.fa \
-q \
-p 10 \
-o ${i}_cufflinks \
${i}.bam

cp ${i}_cufflinks/genes.fpkm_tracking ${RESULT}/${i}.genes.fpkm_tracking
cp ${i}_cufflinks/isoforms.fpkm_tracking ${RESULT}/${i}.isoforms.fpkm_tracking

cd ${RESULT}
gzip ${i}.genes.fpkm_tracking
gzip ${i}.isoforms.fpkm_tracking

cd ${PAIRED}
if [ -e ${RESULT}/${i}.genes.fpkm_tracking.gz ]; then
 echo "$i FPKM is Done"
 /bin/rm -rf ${PAIRED}/${i}
else
 echo "$i FPKM is Wrong"
fi

echo "${i} Finished!!!!"
echo "\n"

done

scontrol show job $SLURM_JOB_ID # information about the job and each thread

