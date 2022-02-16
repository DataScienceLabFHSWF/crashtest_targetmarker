MODELS_FOLDER="/media/dagie002/data2/targetmarker/test/bachelor_test_neu_duplicate_prev/models/"
INPUT_FOLDER="/media/dagie002/data2/targetmarker/test/bachelor_test/testvideos/"
OUTPUT_FOLDER="/output/"
python process_video.py \
	--input ${INPUT_FOLDER} \
	--output ${OUTPUT_FOLDER} \
	--models ${MODELS_FOLDER}

#INPUT_FOLDER="./testvideos/yt"
INPUT_FOLDER="/media/dagie002/data2/targetmarker/test/bachelor_test/testvideos/"
OUTPUT_FOLDER="/output/"
#OUTPUT_FOLDER="/media/dagie002/DataScience/test/bachelor_test_neu_duplicate_prev/output_wo_bayes"

python process_video.py \
	--input ${INPUT_FOLDER} \
	--output ${OUTPUT_FOLDER} \
	--models ${MODELS_FOLDER}




	

