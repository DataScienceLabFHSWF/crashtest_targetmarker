
PIPELINE="4"

PIPELINE_FOLDER="pipeline${PIPELINE}"
INPUT_FOLDER="../testvideos/"
OUTPUT_FOLDER="${PIPELINE_FOLDER}/output/"
MODELS_FOLDER="../models/"

python ${PIPELINE_FOLDER}/process_video.py \
	--input ${INPUT_FOLDER} \
	--output ${OUTPUT_FOLDER} \
	--models ${MODELS_FOLDER}

	

