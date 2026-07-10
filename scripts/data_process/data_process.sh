WORK_DIR=/home/tato/MarineEVT
LOCAL_DIR=/project/marieninst/an/marineevt

## process multiple dataset search format train file
# DATA=train
# python $WORK_DIR/scripts/data_process/marineevt_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=test
python $WORK_DIR/scripts/data_process/marineevt_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA