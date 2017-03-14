#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss_mx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LossMXLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
 
  lambda = (Dtype)this->layer_param_.lambda();  
  
  shp = bottom[0]->shape();  
  int dm = shp[1];
  shp[1] = 1;  
  tmp1.Reshape(shp);
  shp[1] = dm;  
}

template <typename Dtype>
void LossMXLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  vector<int> loss_shape(0);  
  top[0]->Reshape(loss_shape);  
}


template <typename Dtype>
void LossMXLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Not Implemented
}

template <typename Dtype>
void LossMXLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Not Implemented
}



#ifdef CPU_ONLY
STUB_GPU(LossMXLayer);
#endif

INSTANTIATE_CLASS(LossMXLayer);
REGISTER_LAYER_CLASS(LossMX);

}  // namespace caffe
