#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss_ts_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LossTSLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
  
  nt = (int)this->layer_param_.nt();
  lambda = (Dtype)this->layer_param_.lambda();
  
  shp = bottom[0]->shape();  
  int shp0 = shp[0];
  shp[0] = nt*shp0;
  tmp1.Reshape(shp);
  tmp2.Reshape(shp);
  shp[0] = shp0;
  
}

template <typename Dtype>
void LossTSLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {    
  vector<int> loss_shape(0);  
  top[0]->Reshape(loss_shape);    
}


template <typename Dtype>
void LossTSLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Not Implemented  
}

template <typename Dtype>
void LossTSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Not Implemented  
}

#ifdef CPU_ONLY
STUB_GPU(LossTSLayer);
#endif

INSTANTIATE_CLASS(LossTSLayer);
REGISTER_LAYER_CLASS(LossTS);

}  // namespace caffe
