#!/bin/sh
export IPAVLOV_FTP="ftp://share.ipavlov.mipt.ru"
export EMBEDDINGS_URL="http://lnsigo.mipt.ru/export/"
export MODELS_URL="http://lnsigo.mipt.ru/export/"
export DATASETS_URL="http://lnsigo.mipt.ru/export/"

while getopts "k:m:e:i:t:l" option; do
	case "${option}"
	in
		k) KPI_NAME="-k $OPTARG";;
		m) MODEL_FOLDER="-m $OPTARG";;
		e) EMBEDDING_FILE="-e $OPTARG";;
		i) ITER_NUM="-i $OPTARG";;
		t) TASKS_NUMBER="-t $OPTARG";;
		l) LOG_STATE="-l";;
	esac
done

python3 run_test.py $KPI_NAME $MODEL_FOLDER $EMBEDDING_FILE $ITER_NUM $TASKS_NUMBER $LOG_STATE