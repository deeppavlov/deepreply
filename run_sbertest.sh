#!/bin/sh
export IPAVLOV_FTP="ftp://share.ipavlov.mipt.ru"
export EMBEDDINGS_URL="http://share.ipavlov.mipt.ru:8080/repository/embeddings/"
export MODELS_URL="http://share.ipavlov.mipt.ru:8080/repository/models/"
export DATASETS_URL="http://share.ipavlov.mipt.ru:8080/repository/datasets/"

while getopts "k:m:e:" option; do
	case "${option}"
	in
		k) KPI_NAME="-k $OPTARG";;
		m) MODEL_FOLDER="-m $OPTARG";;
		e) EMBEDDING_FILE="-e $OPTARG";;
	esac
done

python3 sbertest.py $KPI_NAME $MODEL_FOLDER $EMBEDDING_FILE