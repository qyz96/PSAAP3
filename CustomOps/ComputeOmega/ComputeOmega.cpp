#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeOmega.h"


REGISTER_OP("ComputeOmega")
.Input("temp : double")
.Input("y : double")
.Output("omega : double")
.Output("domega : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle temp_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &temp_shape));
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("ComputeOmegaGrad")
.Input("grad_omega : double")
.Input("grad_domega : double")
.Input("omega : double")
.Input("domega : double")
.Input("temp : double")
.Input("y : double")
.Output("grad_temp : double")
.Output("grad_y : double");

/*-------------------------------------------------------------------------------------*/

class ComputeOmegaOp : public OpKernel {
private:
  
public:
  explicit ComputeOmegaOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& temp = context->input(0);
    const Tensor& y = context->input(1);
    
    
    const TensorShape& temp_shape = temp.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(temp_shape.dims(), 0);
    DCHECK_EQ(y_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape omega_shape({gd.N});
    TensorShape domega_shape({gd.N,gd.N+1}); // row-major 
            
    // create output tensor
    
    Tensor* omega = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, omega_shape, &omega));
    Tensor* domega = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, domega_shape, &domega));
    
    // get the corresponding Eigen tensors for data access
    
    auto temp_tensor = temp.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto omega_tensor = omega->flat<double>().data();
    auto domega_tensor = domega->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward_ComputeOmega( omega_tensor, domega_tensor, y_tensor, *temp_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeOmega").Device(DEVICE_CPU), ComputeOmegaOp);



class ComputeOmegaGradOp : public OpKernel {
private:
  
public:
  explicit ComputeOmegaGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_omega = context->input(0);
    const Tensor& grad_domega = context->input(1);
    const Tensor& omega = context->input(2);
    const Tensor& domega = context->input(3);
    const Tensor& temp = context->input(4);
    const Tensor& y = context->input(5);
    
    
    const TensorShape& grad_omega_shape = grad_omega.shape();
    const TensorShape& grad_domega_shape = grad_domega.shape();
    const TensorShape& omega_shape = omega.shape();
    const TensorShape& domega_shape = domega.shape();
    const TensorShape& temp_shape = temp.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(grad_omega_shape.dims(), 1);
    DCHECK_EQ(grad_domega_shape.dims(), 2);
    DCHECK_EQ(omega_shape.dims(), 1);
    DCHECK_EQ(domega_shape.dims(), 2);
    DCHECK_EQ(temp_shape.dims(), 0);
    DCHECK_EQ(y_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_temp_shape(temp_shape);
    TensorShape grad_y_shape(y_shape);
            
    // create output tensor
    
    Tensor* grad_temp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_temp_shape, &grad_temp));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    
    // get the corresponding Eigen tensors for data access
    
    auto temp_tensor = temp.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_omega_tensor = grad_omega.flat<double>().data();
    auto grad_domega_tensor = grad_domega.flat<double>().data();
    auto omega_tensor = omega.flat<double>().data();
    auto domega_tensor = domega.flat<double>().data();
    auto grad_temp_tensor = grad_temp->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward_ComputeOmega(
        grad_y_tensor, grad_temp_tensor, grad_omega_tensor,y_tensor,*temp_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeOmegaGrad").Device(DEVICE_CPU), ComputeOmegaGradOp);
