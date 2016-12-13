#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/rearrange.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/convupsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvupsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvupsampleParameter conv_param = this->layer_param_.convupsample_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  if(!conv_param.has_num_output_h()){
    num_output_h_=conv_param.num_output();
    num_output_w_=conv_param.num_output();
  }else{
    num_output_h_=conv_param.num_output_h();
    num_output_w_=conv_param.num_output_w();
  }
  // Configure output channels and groups. we operate convolution per channel
 // channels_ = bottom[0]->channels();
  num_ = bottom[0]->shape(0);
  channels_=bottom[0]->shape(1);
  height_ = bottom[0]->shape(2);
  width_ = bottom[0]->shape(3); 
  
  //channels_=1;
   height_out_ =
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  CHECK_EQ(num_output_h_%height_out_,0);
  CHECK_EQ(num_output_w_%width_out_,0)<<"Number of output_h and output_w should be multiple of width_out and height_out";
  kernel_num_=(num_output_h_/height_out_)*(num_output_w_/width_out_);
  //num_output_ = this->layer_param_.convupsample_param().num_output();
  num_output_ = num_output_h_*num_output_w_;
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convupsample_param().group();
  CHECK_EQ(channels_ % group_, 0);
 // CHECK_EQ(num_output_ % group_, 0)
   //   << "Number of output should be multiples of group.";
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convupsample_param().bias_term();
  //LOG(INFO)<<bias_term_;
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width kernel =1 for our sampling task

    this->blobs_[0].reset(new Blob<Dtype>(
        kernel_num_,1, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convupsample_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, kernel_num_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convupsample_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
   //   LOG(INFO)<<kernel_num_;
    }
  }
  //using bilinear to initialize the weight 
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
int table[num_output_h_/height_out_][num_output_w_/width_out_];
int trace = num_output_h_/height_out_/(kernel_h_-1);
int index=0;

for(int p=0;p<kernel_h_-1;++p)
{
  for(int q=0;q<kernel_w_-1;++q){
    int b= (p>=kernel_h_-2)?trace+(num_output_h_/height_out_%(kernel_h_-1)):trace; 
   // LOG(INFO)<<b<<' '<<p;
   for (int t =0 ;t<b;++t){
    int a = (q>=kernel_w_-2)?trace+(num_output_w_/width_out_%(kernel_w_-1)):trace;
      for(int s=0;s<a;++s){ 
       // LOG(INFO)<<p*trace+t<<' '<<q*trace+s;
        table[p*trace+t][q*trace+s] = index;
      }
    }
    index+=1;
  }
  index+=1;
}
//LOG(INFO)<<index;
//record corresponding index in the latter picture

//LOG(INFO)<<"convupsampleweight setup";
int *recordheight = new int[kernel_h_];
int *recordwidth=new int[kernel_w_];
for(int q=0;q<kernel_h_;++q) recordheight[q]=q*trace;
  recordheight[kernel_h_-1]= num_output_h_/height_out_-1;
for(int q=0;q<kernel_w_;++q) recordwidth[q]=q*trace;
  recordwidth[kernel_w_-1]=num_output_w_/width_out_-1;
//Intitial the weight filler 
Dtype* weight = this->blobs_[0]->mutable_cpu_data();
for(int p=0;p<num_output_h_/height_out_;++p)
{
  for(int q=0;q<num_output_w_/width_out_;++q)
  {
    //LOG(INFO)<<"Initialize weight"<<p<<' '<<q<<' '<<table[p][q];
    int upperleft = table[p][q];

    //x: row,y:column
    int upperleft_x= recordheight[upperleft/kernel_w_];
    int upperleft_y= recordwidth[upperleft%kernel_w_];

    int upperright = upperleft +1;
   // int upperright_x = recordheight[upperright/width_];
   // int upperright_y = recordwidth[upperright%width_];

    int lowleft = upperleft + kernel_w_;
    //int lowleft_x = recordheight[lowleft/width_];
   // int lowleft_y = recordwidth[lowleft%width_];

    int lowright = lowleft+1;
    int lowright_x=recordheight[lowright/kernel_w_];
    int lowright_y=recordwidth[lowright%kernel_w_];

    double diff_x = lowright_x - upperleft_x;
    double diff_y = lowright_y-upperleft_y;
     
   weight[(p*num_output_w_/width_out_+q)*kernel_w_*kernel_h_+upperleft]= ((lowright_y-q)/diff_y)*((lowright_x-p)/diff_x);
   weight[(p*num_output_w_/width_out_+q)*kernel_h_*kernel_w_+upperright]=((q-upperleft_y)/diff_y)*((lowright_x-p)/diff_x);
   weight[(p*num_output_w_/width_out_+q)*kernel_w_*kernel_h_+lowleft]=((lowright_y-q)/diff_y)*((p-upperleft_x)/diff_x);
   weight[(p*num_output_w_/width_out_+q)*kernel_h_*kernel_w_+lowright]=((q-upperleft_y)/diff_y)*((p-upperleft_x)/diff_x);

  }                                                                                                                                                                                                                                                                                                                                                                                                         
}
std::ofstream fout("test/weight.txt");
for(int p=0;p<num_output_h_/height_out_;++p)
{
  for(int q=0;q<num_output_w_/width_out_;++q)
  {
    for (int r=0;r<kernel_h_;++r){
      for(int s=0;s<kernel_w_;++s){
        weight[(p*num_output_w_/height_out_+q)*kernel_h_*kernel_w_+r*kernel_w_+s]+=0.00001;
        fout<<weight[(p*num_output_h_/height_out_+q)*kernel_h_*kernel_w_+r*kernel_w_+s]<<' ';
      }
    }
  //  fout<<std::endl;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConvupsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
 // CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
   // " convupsample kernel.";
  // TODO: generalize to handle inputs of different shapes.
  //for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
   // CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    //CHECK_EQ(channels_, bottom[bottom_id]->channels())
   //     << "Inputs must have same channels.";
    //CHECK_EQ(height_, bottom[bottom_id]->height())
     //   << "Inputs must have same height.";
   // CHECK_EQ(width_, bottom[bottom_id]->width())
       // << "Inputs must have same width.";
  //}
  // Shape the tops.
 
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_,channels_, num_output_h_, num_output_w_);
  }
  //LOG(INFO)<<num_output_h_<<' '<<num_output_w_;
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  //M_ = num_output_ / group_;
 // K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  //N_ = height_out_ * width_out_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.One sample, one channel
  col_buffer_.Reshape(
      1, kernel_h_ * kernel_w_, height_out_, width_out_);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_,channels_, num_output_h_, num_output_w_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1,height_out_*width_out_);
    caffe_set(height_out_*width_out_ ,Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ConvupsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
      Blob<Dtype> tmp;
      tmp.Reshape(num_,channels_,num_output_h_,num_output_w_);
  for (int i = 0; i < bottom.size(); ++i) {
    //const Dtype* bottom_data = bottom[i]->cpu_data();
    //Dtype* top_data = (*top)[0]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    //int weight_offset = M_ * K_;  // number of filter parameters in a group
    //int col_offset = K_ * N_;  // number of values in an input region / column
    //int top_offset = M_ * N_;  // number of values in an output region / column
    Dtype* ptr_bottom_data = bottom[i]->mutable_cpu_data();
    Dtype* ptr_top_data=tmp.mutable_cpu_data();
  //  int bottom_data_offset = height_*width_*channels_;
   // int top_data_offset=num_output_h_*num_output_w_*channels_;
  //  int dim_bottom_data_offset = height_*width_;
   // int dim_top_data_offset=num_output_h_*num_output_w_;
    int KK_=kernel_h_*kernel_w_;
    int NN_=height_out_*width_out_;
    int MM_=kernel_num_;
    /*std::ofstream fout("examples/test_convupsample_layer/test/test.txt");
    for(int p=0;p<3;++p)
      for(int q=0;q<8;++q){
        for(int l=0;l<8;++l )
          fout<<ptr_bottom_data[p*64+q*8+l]<<' ';
        fout<<std::endl;
      }*/
    for (int n = 0; n < num_; ++n) {
      for (int dim =0 ;dim<channels_;++dim){
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_cpu_deeplab(ptr_bottom_data,1,1 , height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,1,1,
          col_data);
     /*std:: ofstream fout1("examples/test_convupsample_layer/test/coldata.txt",ios::app);
      for(int p=0;p<196;++p) fout1<<col_data[p]<<' ';
        fout1<<std::endl;*/
      // Take inner products for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, NN_, MM_,KK_,
          (Dtype)1., col_data,weight , 
          (Dtype)0., ptr_top_data);
    //    LOG(INFO)<<NN_<<' '<<MM_<<' '<<KK_;
      }
      // Add bias.
      if (bias_term_) {

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            height_out_*width_out_, kernel_num_,1, (Dtype)1., 
            bias_multiplier_.cpu_data(),this->blobs_[1]->cpu_data(),
            (Dtype)1., ptr_top_data);
      }
      ptr_bottom_data+=height_*width_;
      ptr_top_data+=num_output_h_*num_output_w_;
    }
    }
  }

  Dtype* ptr_tmp_data = tmp.mutable_cpu_data();
  Dtype* ptr_top_data =top[0]->mutable_cpu_data();
  /////////////////////////
 /* std::ofstream fout4("examples/test_convupsample_layer/test/result0.txt");
   for(int p=0;p<3;++p){
    for(int q=0;q<28;++q){
      for(int r=0;r<28;++r)
        fout4<<ptr_tmp_data[p*28*28+q*28+r]<<' ';
    }
    fout4<<std::endl;
   }*/
  ////////////////////////
  arrange_cpu(ptr_tmp_data,num_,channels_,height_out_,
    width_out_,num_output_h_/height_out_,num_output_w_/width_out_,ptr_top_data);
 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /*const Dtype* weight = this->blobs_[0]->cpu_data();
   std::ofstream fout2("examples/test_convupsample_layer/test/weight.txt");
   for(int p=0;p<16;++p){
    for(int q=0;q<4;++q)
      fout2<<weight[p*4+q]<<' ';
    fout2<<std::endl;
   }
   const Dtype* testbias = this->blobs_[1]->cpu_data();
   const Dtype* testmutibias = bias_multiplier_.cpu_data();
   std::ofstream fout5("examples/test_convupsample_layer/test/bias.txt");
   for(int p=0;p<16;++p) fout5<<testbias[p]<<' ';
    std::ofstream fout6("examples/test_convupsample_layer/test/multi.txt");
  for (int p=0;p<49;++p) fout6<<testmutibias[p]<<' ';
  //  Dtype* ptr_top_data=(*top)[0]->mutable_cpu_data();
   std::ofstream fout3("examples/test_convupsample_layer/test/result.txt");
   for(int p=0;p<3;++p){
    for(int q=0;q<28;++q){
      for(int r=0;r<28;++r)
        fout3<<ptr_top_data[p*28*28+q*28+r]<<' ';
    }
    fout3<<std::endl;
   }
*//////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

template <typename Dtype>
void ConvupsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // LOG(INFO)<<"BEGIN";
   const Dtype* weight = NULL;
   Dtype* weight_diff = NULL;
   //reverse process, reverse the output to the form of input.
   Blob<Dtype> tmp;
   tmp.Reshape(num_,channels_,num_output_h_,num_output_w_);
   Dtype* tmp_diff= tmp.mutable_cpu_diff();
   Dtype* top_diff = top[0]->mutable_cpu_diff();
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /*std::ofstream fout2("examples/test_convupsample_layer/test/top_diff0.txt");
   for(int p=0;p<3;++p){
    for(int q=0;q<784;++q)
    {
      fout2<<top_diff[p*784+q]<<' ';
    }
    fout2<<std::endl;
   }*/
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //simple copy the data to the tmp blob;
   for(int t=0;t<top[0]->count();++t)
   {
    tmp_diff[t]=top_diff[t];
   }
   Dtype* ptr_tmp = tmp.mutable_cpu_diff();
   
  rearrange_cpu(ptr_tmp,num_,channels_,height_out_,width_out_,num_output_h_/height_out_,num_output_w_/width_out_,top_diff);
 /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*std:: ofstream fout1("examples/test_convupsample_layer/test/top_diff.txt");
for(int p=0;p<3;++p)
{
  for(int q=0;q<784;++q) fout1<<top_diff[p*784+q]<<' ';
    fout1<<std::endl;
}*/
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
   // LOG(INFO)<<"true";
  }
  //////////////////////////////////////////////////////////////////////////////////////////
 /*//print bottom data 
  const Dtype* testbias = this->blobs_[1]->cpu_data();
   const Dtype* testmutibias = bias_multiplier_.cpu_data();
   std::ofstream fout5("examples/test_convupsample_layer/test/bias.txt");
   for(int p=0;p<16;++p) fout5<<testbias[p]<<' ';
    std::ofstream fout6("examples/test_convupsample_layer/test/multi.txt");
  for (int p=0;p<49;++p) fout6<<testmutibias[p]<<' ';*/
  /////////////////////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < top.size(); ++i) {
    //const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
    Dtype* ptr_top_data = top[0]->mutable_cpu_diff();
    int outnum = num_output_h_*num_output_w_;
    //  top_diff = top[i]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        for(int dim=0;dim<channels_;++dim){
        caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans, kernel_num_,1,height_out_*width_out_,
            1., ptr_top_data,
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
        ptr_top_data += outnum;
       }
      }
    }
 // std::ofstream fout7("examples/test_convupsample_layer/test/bias_diff.txt");
 // for(int p=0;p<16;++p) fout7<<bias_diff[p]<<' ';
     // LOG(INFO)<<"BackWard";
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->mutable_cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
    //  const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      Dtype* ptr_bottom_data = bottom[i]->mutable_cpu_data();
      Dtype* ptr_top_diff = top[0]->mutable_cpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
       for(int dim=0;dim<channels_;++dim){
         im2col_cpu_deeplab(ptr_bottom_data,1,1 , height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,1,1,
          col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, kernel_num_, kernel_w_*kernel_h_,height_out_*width_out_,
                (Dtype)1., ptr_top_diff,
                col_data , (Dtype)1.,
                weight_diff);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, kernel_w_*kernel_h_, height_out_*width_out_, kernel_num_,
                (Dtype)1., weight ,
                ptr_top_diff,
                (Dtype)0., col_diff);
            //////////////////////////////
        /*    std::ofstream fout10("examples/test_convupsample_layer/test/coldiff.txt",ios::app);
            for(int p=0;p<4;++p){
              for(int q=0;q<49;++q)
              {
                fout10<<col_diff[p*49+q]<<' ';
              }
              fout10<<std::endl;
            }*/
            ////////////////////////////////
          }
          // col2im back to the data
          col2im_cpu_deeplab(col_diff,1, 1, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_,1,1, bottom_diff);
        }
        ptr_bottom_data+=height_*width_;
        ptr_top_diff+=num_output_h_*num_output_w_;
        bottom_diff+=height_*width_;
      }
      }
//check weight diff

    }
  }
  //LOG(INFO)<<"BACKWard";
}

INSTANTIATE_CLASS(ConvupsampleLayer);
REGISTER_LAYER_CLASS(Convupsample);
}  // namespace caffe