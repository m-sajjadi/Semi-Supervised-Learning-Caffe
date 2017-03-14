#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss_ts_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_ts(const int nthreads, const Dtype* in_data,
    const int nt, const int dim, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {        
    const int ind1 = (nt*dim)*(index /(nt*nt*dim)) + index % (nt*dim);
    const int ind2 = dim*(index /(dim*nt)) + index % dim;    
    out_data[index] = in_data[ind1] - in_data[ind2];    
  }
}

template <typename Dtype>
void LossTSLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  const Dtype* dt = bottom[0]->gpu_data();
  Dtype* tmp1_ = tmp1.mutable_gpu_data();
  Dtype* tmp2_ = tmp2.mutable_gpu_data();  
  const int nthreads = tmp1.count();
  Forward_ts<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, dt, nt, shp[1], tmp1_);
    
  caffe_gpu_powx<Dtype>(nthreads, tmp1_, Dtype(2), tmp2_); 
  Dtype loss;
  caffe_gpu_asum(nthreads, tmp2_, &loss); 
  loss = loss/shp[0];

  top[0]->mutable_cpu_data()[0] = loss;
  

}

template <typename Dtype>
__global__ void Backward_ts(const int nthreads, const Dtype* in_data,
    const int nt, const int dim, Dtype* out_data, const Dtype lambda) {
  CUDA_KERNEL_LOOP(index, nthreads) {        
    const int ind1 = (nt*dim)*(index /dim) + index % dim;    
    out_data[index] = 0;
    for (int i=0; i<nt; i++){
      out_data[index] = out_data[index] - lambda*in_data[ind1 + dim*i];
    }
  }
}

template <typename Dtype>
void LossTSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    const Dtype* tmp1_ = tmp1.gpu_data();     
    const int nthreads = bottom[0]->count();
    
    Backward_ts<Dtype>
	  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	  nthreads, tmp1_, nt, shp[1], bottom_diff, lambda); 	  
  }	
  
}

INSTANTIATE_LAYER_GPU_FUNCS(LossTSLayer);

}  // namespace caffe
