#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void  arrange_cpu(Dtype* data_in, const int number,const int dimension,
    const int height, const int width, const int kernel_h, const int kernel_w,
    Dtype* data_out) {
  //height: the height_out_ width:width_out_
  //int offset = kernel_h*kernel_w;
  int dim_offset=height*width*kernel_h*kernel_w;
  int num_offset= dimension*dim_offset;
  int column=kernel_w*width;
  for(int num=0;num<number;++num){
  for (int dim =0;dim<dimension;++dim){
    for(int h=0;h<height;++h){
      for(int w=0;w<width;++w){
        for(int kh=0;kh<kernel_h;++kh){
          for(int kw=0;kw<kernel_w;++kw){
            data_out[num*num_offset+dim*dim_offset+(h*kernel_h+kh)*column+(w*kernel_w)+kw] = 
            data_in[num*num_offset+dim*dim_offset+h*width*kernel_h*kernel_w+w*kernel_h*kernel_w+kh*kernel_w+kw];
         //   LOG(INFO)<<num*num_offset+dim*dim_offset+(h*kernel_h+kh)*column+(w*kernel_w)+kw<<' '<<num*num_offset+dim*dim_offset+h*width*kernel_h*kernel_w+w*kernel_h*kernel_w+kh*kernel_w+kw;
          }
        }
        }
      }
    }
  }
}

// Explicit instantiation
template void arrange_cpu<float>(float* data_in, const int number,const int dimension,
    const int height, const int width, const int kernel_h, const int kernel_w,float* dat_out);
template void arrange_cpu<double>( double* data_in, const int number,const int dimension,
    const int height, const int width, const int kernel_h, const int kernel_w,double* data_out);

template <typename Dtype>
void rearrange_cpu(Dtype* data_in,const int number, const int dimension,
    const int height, const int width, const int kernel_h,const int kernel_w, Dtype* data_out) 
{
  //int offset = kernel_h*kernel_w;
  int dim_offset=height*width*kernel_h*kernel_w;
  int num_offset= dimension*dim_offset;
  int column=kernel_w*width;
  for(int num=0;num<number;++num){
  for (int dim =0;dim<dimension;++dim){
    for(int h=0;h<height;++h){
      for(int w=0;w<width;++w){
        for(int kh=0;kh<kernel_h;++kh){
          for(int kw=0;kw<kernel_w;++kw){
            data_out[num*num_offset+dim*dim_offset+h*width*kernel_h*kernel_w+w*kernel_h*kernel_w+kh*kernel_w+kw]=
            data_in[num*num_offset+dim*dim_offset+(h*kernel_h+kh)*column+(w*kernel_w)+kw];
          }
        }
        }
      }
    }
  }
}

// Explicit instantiation
template void rearrange_cpu<float>(float* data_in,const int number, const int dimension,
    const int height, const int width, const int kernel_h, const int kernel_w, float* data_out);
template void rearrange_cpu<double>(double* data_in,const int number, const int dimension,
    const int height, const int width, const int kernel_h, const int kernel_w, double* data_out);

}  // namespace caffe