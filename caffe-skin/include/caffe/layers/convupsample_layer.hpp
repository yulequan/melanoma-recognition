#ifndef CAFFE_CONVUPSAMPLE_LAYER_HPP_
#define CAFFE_CONVUPSAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

//lequan add convupsample layer 
template <typename Dtype>
class ConvupsampleLayer : public Layer<Dtype> {
 public:
  
  explicit ConvupsampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Convupsample"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //   vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
   //   const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int height_, width_;
  int group_;
  int num_output_,num_output_h_,num_output_w_;
  int height_out_, width_out_;
  int kernel_num_;
  bool bias_term_;

  /// M_ is the channel dimension of the output for a single group, which is the
  /// leading dimension of the filter matrix.
  //int M_;
  /// K_ is the dimension of an unrolled input for a single group, which is the
  /// leading dimension of the data matrix.
  //int K_;
  /// N_ is the spatial dimension of the output, the H x W, which are the last
  /// dimensions of the data and filter matrices.
  //int N_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};
}


#endif 