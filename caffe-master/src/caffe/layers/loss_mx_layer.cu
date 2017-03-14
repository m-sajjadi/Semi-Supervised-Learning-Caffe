#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss_mx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_tx(const int nthreads, Dtype* in_data,
    const int dim, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {      
    const int ind = dim*index;  
    float p = 1;
    
    Dtype ep = FLT_EPSILON;    
    
    for (int i = 0; i < dim; i++){
      
      in_data[ind+i] = max(in_data[ind+i], ep);
      in_data[ind+i] = min(in_data[ind+i], 1-ep);
      
      p = p * (1 - in_data[ind + i]);
    }
    out_data[index] = 0;    
    for (int i = 0; i < dim; i++){
      out_data[index] = out_data[index] + p * in_data[ind + i] / (1 - in_data[ind + i]);
    }
  }
}

template <typename Dtype>
void LossMXLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  Dtype* dt = bottom[0]->mutable_gpu_data();
  Dtype* tmp1_ = tmp1.mutable_gpu_data();
  const int nthreads = tmp1.count();  
  Forward_tx<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, dt, shp[1], tmp1_);
  
  Dtype loss;
  caffe_gpu_asum(nthreads, tmp1_, &loss);  
  loss = loss/shp[0];

  top[0]->mutable_cpu_data()[0] = loss;
  
}

template <typename Dtype>
__global__ void Backward_tx(const int nthreads, const Dtype* in_data,
    const int dim, Dtype* out_data, const Dtype lambda) {
  CUDA_KERNEL_LOOP(index, nthreads) {    
    
    const int ind = (index / dim) * dim;
    float p = 1;
    for (int i = 0; i<dim; i++){
	p = p * (1 - in_data[ ind + i ]);
    }		    

    float t1 = p/(1 - in_data[index]);	
    
    float t2 = 0;
    for (int i = 0; i<dim; i++){
	t2 = t2 + p * in_data[ ind + i ] / (1 - in_data[ ind + i ]) / (1 - in_data[ index ]);	  
    }
    t2 = -t2;
    float t3 = p*in_data[index]/(1 - in_data[index])/(1 - in_data[index]);
    
    float v = (t1 + t2 + t3);    
    
    out_data[index] = -lambda*v;
  }
}

template <typename Dtype>
void LossMXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    const Dtype* dt = bottom[0]->gpu_data();
    const int nthreads = bottom[0]->count();    
    Backward_tx<Dtype>
	  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	  nthreads, dt, shp[1], bottom_diff, lambda);    
  }	
  
}

INSTANTIATE_LAYER_GPU_FUNCS(LossMXLayer);

}  // namespace caffe
