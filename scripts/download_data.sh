#!/bin/bash

cd data/raw


curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.gz
curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.gz.json

curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz
curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.json

curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv.gz
curl -L -O https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv.json

echo "Download complete."