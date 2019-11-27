#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;

int EMD_wrap(int n1,int n2, const double *X, const double *Y,const double *D, double *G, double* alpha, double* beta, double *cost, int maxIter);

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("OTNetwork")

.Input("a : double")
  .Input("b : double")
  .Input("d : double")
  .Input("iter : int64")
  .Output("g : double")
  .Output("cost : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &b_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &d_shape));
        shape_inference::ShapeHandle iter_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iter_shape));

        c->set_output(0, c->Matrix(-1,-1));
        c->set_output(1, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("OTNetworkGrad")

.Input("grad_g : double")
.Input("grad_cost : double")
  .Input("g : double")
  .Input("cost : double")
  .Input("a : double")
  .Input("b : double")
  .Input("d : double")
  .Input("iter : int64")
  .Output("grad_a : double")
  .Output("grad_b : double")
  .Output("grad_d : double")
  .Output("grad_iter : int64");


class OTNetworkOp : public OpKernel {
private:
  
public:
  explicit OTNetworkOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& d = context->input(2);
    const Tensor& iter = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& iter_shape = iter.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 2);
    DCHECK_EQ(iter_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n1 = a_shape.dim_size(0), n2 = b_shape.dim_size(0);
    DCHECK_EQ(d_shape.dim_size(0), n1);
    DCHECK_EQ(d_shape.dim_size(1), n2);

    TensorShape g_shape({n1, n2});
    TensorShape cost_shape({});
            
    // create output tensor
    
    Tensor* g = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, g_shape, &g));
    Tensor* cost = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, cost_shape, &cost));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto g_tensor = g->flat<double>().data();
    auto cost_tensor = cost->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    auto alpha = new double[n1];
    auto beta = new double[n2];
    EMD_wrap(n1, n2, a_tensor, b_tensor, d_tensor, g_tensor, alpha, beta, cost_tensor, *iter_tensor);
    delete [] alpha;
    delete [] beta;

  }
};
REGISTER_KERNEL_BUILDER(Name("OTNetwork").Device(DEVICE_CPU), OTNetworkOp);



class OTNetworkGradOp : public OpKernel {
private:
  
public:
  explicit OTNetworkGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_g = context->input(0);
    const Tensor& grad_cost = context->input(1);
    const Tensor& g = context->input(2);
    const Tensor& cost = context->input(3);
    const Tensor& a = context->input(4);
    const Tensor& b = context->input(5);
    const Tensor& d = context->input(6);
    const Tensor& iter = context->input(7);
    
    
    const TensorShape& grad_g_shape = grad_g.shape();
    const TensorShape& grad_cost_shape = grad_cost.shape();
    const TensorShape& g_shape = g.shape();
    const TensorShape& cost_shape = cost.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& iter_shape = iter.shape();
    
    
    DCHECK_EQ(grad_g_shape.dims(), 2);
    DCHECK_EQ(grad_cost_shape.dims(), 0);
    DCHECK_EQ(g_shape.dims(), 2);
    DCHECK_EQ(cost_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 2);
    DCHECK_EQ(iter_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_d_shape(d_shape);
    TensorShape grad_iter_shape(iter_shape);
            
    // create output tensor
    int n1 = a_shape.dim_size(0), n2 = b_shape.dim_size(0);
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_d_shape, &grad_d));
    Tensor* grad_iter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_iter_shape, &grad_iter));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto grad_g_tensor = grad_g.flat<double>().data();
    auto grad_cost_tensor = grad_cost.flat<double>().data();
    auto g_tensor = g.flat<double>().data();
    auto cost_tensor = cost.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();
    auto grad_iter_tensor = grad_iter->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    for(int i=0;i<n1*n2;i++)  grad_d_tensor[i] = *grad_cost_tensor * g_tensor[i];
    
  }
};
REGISTER_KERNEL_BUILDER(Name("OTNetworkGrad").Device(DEVICE_CPU), OTNetworkGradOp);

#ifdef USE_GPU
class OTNetworkOpGPU : public OpKernel {
private:
  
public:
  explicit OTNetworkOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& d = context->input(2);
    const Tensor& iter = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& iter_shape = iter.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 2);
    DCHECK_EQ(iter_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape g_shape({-1,-1});
    TensorShape cost_shape({});
            
    // create output tensor
    
    Tensor* g = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, g_shape, &g));
    Tensor* cost = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, cost_shape, &cost));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto g_tensor = g->flat<double>().data();
    auto cost_tensor = cost->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("OTNetwork").Device(DEVICE_GPU), OTNetworkOpGPU);



class OTNetworkGradOpGPU : public OpKernel {
private:
  
public:
  explicit OTNetworkGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_g = context->input(0);
    const Tensor& grad_cost = context->input(1);
    const Tensor& g = context->input(2);
    const Tensor& cost = context->input(3);
    const Tensor& a = context->input(4);
    const Tensor& b = context->input(5);
    const Tensor& d = context->input(6);
    const Tensor& iter = context->input(7);
    
    
    const TensorShape& grad_g_shape = grad_g.shape();
    const TensorShape& grad_cost_shape = grad_cost.shape();
    const TensorShape& g_shape = g.shape();
    const TensorShape& cost_shape = cost.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& iter_shape = iter.shape();
    
    
    DCHECK_EQ(grad_g_shape.dims(), 2);
    DCHECK_EQ(grad_cost_shape.dims(), 0);
    DCHECK_EQ(g_shape.dims(), 2);
    DCHECK_EQ(cost_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 2);
    DCHECK_EQ(iter_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_d_shape(d_shape);
    TensorShape grad_iter_shape(iter_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_d_shape, &grad_d));
    Tensor* grad_iter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_iter_shape, &grad_iter));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto grad_g_tensor = grad_g.flat<double>().data();
    auto grad_cost_tensor = grad_cost.flat<double>().data();
    auto g_tensor = g.flat<double>().data();
    auto cost_tensor = cost.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();
    auto grad_iter_tensor = grad_iter->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("OTNetworkGrad").Device(DEVICE_GPU), OTNetworkGradOpGPU);

#endif