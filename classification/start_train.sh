LOGDIR=./training_log
CAFFE=../caffe-skin/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./ResNet-50-model.caffemodel
mkdir snapshot
mkdir $LOGDIR

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

