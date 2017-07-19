/// \file correlation.cc
/// \author Jgorgen
/// \brief Implementation of a pixel-wise correlation
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "correlation_param.h"

using namespace tensorflow;

REGISTER_OP("Correlation")
  .Input("a: float")
  .Input("b: float")
  .Output("correlation: float")
  .Attr("stride: int = 2")
  .Attr("max_displacement: int = 20")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context) {
    shape_inference::ShapeHandle input_shape = context->input(0);
    shape_inference::ShapeHandle output_shape;
    int max_displacement;
    int stride;
    context->GetAttr("stride", &stride);
    context->GetAttr("max_displacement", &max_displacement);
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &input_shape));
    int num_steps = 2*(max_displacement/stride) + 1;
    int num_outputs = num_steps*num_steps;
    context->ReplaceDim(input_shape, 3, context->MakeDim(shape_inference::DimensionOrConstant(num_outputs)), &output_shape);
    if(num_outputs > CORRELATION_OPERATOR_MAX_OFFSETS)
    {
        ::tensorflow::Status _status; 
        _status.Update(errors::InvalidArgument("Number of offsets (2*(max_displacement/stride)+1)**2, must be < ", CORRELATION_OPERATOR_MAX_OFFSETS, " got ", num_outputs));
        return _status;
    }
    

    context->set_output(0, output_shape);
    return Status::OK();
  });

/// \brief Implementation of a correlation operation.
/// \param context
/// \author Jgorgen
class CorrelationOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit CorrelationOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the stride to
    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    // Check that stride is positive
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Need stride > 0, got ",
                                        stride_));
        // Get the index of the max_displacement to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement_));
    // Check that max_displacement is positive
    OP_REQUIRES(context, max_displacement_ > 0,
                errors::InvalidArgument("Need max_displacement > 0, got ",
                                        max_displacement_));
    int row_offsets = (int(max_displacement_/stride_) *2) + 1;
    int num_offsets = row_offsets*row_offsets;
    OP_REQUIRES(context, num_offsets < CORRELATION_OPERATOR_MAX_OFFSETS,
                errors::InvalidArgument(( "Need num offsets < "  MACRO_STR(CORRELATION_OPERATOR_MAX_OFFSETS)  ", got "),  num_offsets));
  }
  int stride_;
  int max_displacement_;
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the left tensor
    const Tensor& a = context->input(0);
    
    // get the right tensor
    const Tensor& b = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    
    // check inputs are both (batch_size,height,width,num_channels)
    DCHECK_EQ(a_shape.dims(), 4);
    DCHECK_EQ(b_shape.dims(), 4);
    DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
    DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
    DCHECK_EQ(a_shape.dim_size(2), b_shape.dim_size(2));
    DCHECK_EQ(a_shape.dim_size(3), b_shape.dim_size(3));
                


    // create output shape
    TensorShape output_shape;
    int num_steps = 2*(max_displacement_/stride_) + 1;
    int num_outputs = num_steps*num_steps;

    output_shape.AddDim(a_shape.dim_size(0));
    output_shape.AddDim(a_shape.dim_size(1));
    output_shape.AddDim(a_shape.dim_size(2));
    output_shape.AddDim(num_outputs);
    std::vector<std::pair<int,int> > offsets(num_outputs);
    size_t offset_index = 0;
    for(int j = -this->max_displacement_; j<= this->max_displacement_;  j+= this->stride_)
    {
        for(int k= -this->max_displacement_; k <= this->max_displacement_; k+= this->stride_)
        {
            offsets.at(offset_index).first = j;
            offsets.at(offset_index).second = k;
            offset_index++;
        }
    }
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto a_tensor = a.tensor<float,4>();
    auto b_tensor = b.tensor<float,4>();
    auto output_tensor = output->tensor<float,4>();

    int batch_size = output->shape().dim_size(0);
    int num_rows = output->shape().dim_size(1);
    int num_cols = output->shape().dim_size(2);
    int max_m = a.shape().dim_size(3);
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_rows; j++) {
          for (int k = 0; k < num_cols; k++) {
              for (int l = 0; l < num_outputs; l++) {
                int j_offset = offsets[l].first;
                int k_offset = offsets[l].second;
                int min_j = 0;
                int max_j = num_rows;
                int min_k = 0;
                int max_k = num_cols;
                if(j_offset < 0){
                    min_j = -1*j_offset;
                }else{
                    max_j -= j_offset;
                }
                if(k_offset < 0){
                    min_k = -1*k_offset;
                }else{
                    max_k -= k_offset;
                }
                output_tensor(i, j,k,l) =0 ;
                if( j >= min_j && j < max_j  && k >= min_k && k < max_k)
                {
                    for( int m = 0 ; m < max_m; m++)
                    {
                         output_tensor(i,j,k,l)+= a_tensor(i,j,k,m)*b_tensor(i,j+j_offset,k+k_offset,m);
                    }
                    output_tensor(i,j,k,l)/= max_m;
	    
                }
              }
            }
        }
      }
    
  }
};



void CorrelationKernelLauncher(const float* a, const float*b,float* out, const int batch_size,const int num_rows, const int num_cols, const int depth,const int num_offsets, const int* offset_list);

class CorrelationGpuOp : public OpKernel {
 public:
  /// \brief Constructor.
  /// \param context
  explicit CorrelationGpuOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the stride to
    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    // Check that stride is positive
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Need stride > 0, got ",
                                        stride_));
        // Get the index of the max_displacement to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement_));
    // Check that max_displacement is positive
    OP_REQUIRES(context, max_displacement_ > 0,
                errors::InvalidArgument("Need max_displacement > 0, got ",
                                        max_displacement_));
  }
  int stride_;
  int max_displacement_;

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);


    // check shapes of input and weights
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    
    // check inputs are both (batch_size,height,width,num_channels)
    DCHECK_EQ(a_shape.dims(), 4);
    DCHECK_EQ(b_shape.dims(), 4);
    DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
    DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
    DCHECK_EQ(a_shape.dim_size(2), b_shape.dim_size(2));
    DCHECK_EQ(a_shape.dim_size(3), b_shape.dim_size(3));
                


    // create output shape
    TensorShape output_shape;
    int num_steps = 2*(max_displacement_/stride_) + 1;
    int num_outputs = num_steps*num_steps;

    output_shape.AddDim(a_shape.dim_size(0));
    output_shape.AddDim(a_shape.dim_size(1));
    output_shape.AddDim(a_shape.dim_size(2));
    output_shape.AddDim(num_outputs);

    TensorShape offset_shape;
    offset_shape.AddDim(num_outputs);
    std::vector<int > offsets(2*num_outputs);
    size_t offset_index = 0;
    for(int j = -this->max_displacement_; j<= this->max_displacement_;  j+= this->stride_)
    {
        for(int k= -this->max_displacement_; k <= this->max_displacement_; k+= this->stride_)
        {
            offsets.at(offset_index+0)=j;
            offsets.at(offset_index+1)=k;
            offset_index+=2;
        }
    }


            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));


    
    // get the corresponding Eigen tensors for data access
    int batch_size = output->shape().dim_size(0);
    int num_rows = output->shape().dim_size(1);
    int num_cols = output->shape().dim_size(2);
    int depth = a.shape().dim_size(3);

    auto input_a = a.flat<float>();
    auto input_b = b.flat<float>();

    auto out = output->template flat<float>();

    // Call the cuda kernel launcher
    CorrelationKernelLauncher(input_a.data(),input_b.data(), out.data(), batch_size,num_rows,num_cols,depth,num_outputs,&offsets[0]);
  }
};

REGISTER_KERNEL_BUILDER(Name("Correlation").Device(DEVICE_GPU), CorrelationGpuOp);


void CorrelationFlatKernel(const float* a, const float*b,float* out, const int batch_size,const int num_rows, const int num_cols, const int depth,const int num_offsets, const int* offset_list)  {
    int one_d_size   = depth;
    int two_d_size   = one_d_size*num_cols;
    int three_d_size = two_d_size*num_rows;

    int out1 = num_offsets;
    int out2 = num_cols * out1;
    int out3 = num_rows * out2;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_rows; j += 1) {
          for (int k = 0; k < num_cols; k++) {
              for (int l =0; l < num_offsets; l++ ) {
                int j_offset = offset_list[2*l];
                int k_offset = offset_list[2*l+1];
                int min_j = 0;
                int max_j = num_rows;
                int min_k = 0;
                int max_k = num_cols;
                if(j_offset < 0){
                    min_j = -1*j_offset;
                }else{
                    max_j -= j_offset;
                }
                if(k_offset < 0){
                    min_k = -1*k_offset;
                }else{
                    max_k -= k_offset;
                }
                int a_root = three_d_size*i + two_d_size*j+one_d_size * k;
                int out_index = out3*i + out2*j+out1*k + l;
                out[out_index] =0 ;
                if( j >= min_j && j < max_j  && k >= min_k && k < max_k)
                {
                    int b_j = j+j_offset;
                    int b_k = k+k_offset;
                    int b_root = three_d_size*i + two_d_size*b_j+one_d_size * b_k;
                    for( int m = 0 ; m < depth; m++)
                    {
                         out[out_index]+= a[a_root+m]*b[b_root+m];
                    }
                    out[out_index]/= depth;
	    
                }
              }
            }
        }
      }


}


void CorrelationFlatKernelLauncher(const float* a, const float*b,float* out, const int batch_size,const int num_rows, const int num_cols, const int depth,const int num_offsets, const int* offset_list) {
  CorrelationFlatKernel(a, b, out,batch_size,num_rows,num_cols,depth,num_offsets,offset_list);
}

class CorrelationFlatOp : public OpKernel {
 public:
  /// \brief Constructor.
  /// \param context
  explicit CorrelationFlatOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the stride to
    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    // Check that stride is positive
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Need stride > 0, got ",
                                        stride_));
        // Get the index of the max_displacement to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement_));
    // Check that max_displacement is positive
    OP_REQUIRES(context, max_displacement_ > 0,
                errors::InvalidArgument("Need max_displacement > 0, got ",
                                        max_displacement_));
    int row_offsets = (int(max_displacement_/stride_) *2) + 1;
    int num_offsets = row_offsets*row_offsets;
    OP_REQUIRES(context, num_offsets < CORRELATION_OPERATOR_MAX_OFFSETS,
                errors::InvalidArgument(( "Need num offsets < "  MACRO_STR(CORRELATION_OPERATOR_MAX_OFFSETS)  ", got "),  num_offsets));
  }
  int stride_;
  int max_displacement_;

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);


    // check shapes of input and weights
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    
    // check inputs are both (batch_size,height,width,num_channels)
    DCHECK_EQ(a_shape.dims(), 4);
    DCHECK_EQ(b_shape.dims(), 4);
    DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
    DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
    DCHECK_EQ(a_shape.dim_size(2), b_shape.dim_size(2));
    DCHECK_EQ(a_shape.dim_size(3), b_shape.dim_size(3));
                


    // create output shape
    TensorShape output_shape;
    int num_steps = 2*(max_displacement_/stride_) + 1;
    int num_outputs = num_steps*num_steps;

    output_shape.AddDim(a_shape.dim_size(0));
    output_shape.AddDim(a_shape.dim_size(1));
    output_shape.AddDim(a_shape.dim_size(2));
    output_shape.AddDim(num_outputs);
    std::vector<int > offsets(2*num_outputs);
    size_t offset_index = 0;
    for(int j = -this->max_displacement_; j<= this->max_displacement_;  j+= this->stride_)
    {
        for(int k= -this->max_displacement_; k <= this->max_displacement_; k+= this->stride_)
        {
            offsets.at(offset_index+0)=j;
            offsets.at(offset_index+1)=k;
            offset_index+=2;
        }
    }
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    int batch_size = output->shape().dim_size(0);
    int num_rows = output->shape().dim_size(1);
    int num_cols = output->shape().dim_size(2);
    int depth = a.shape().dim_size(3);

    auto input_a = a.flat<float>();
    auto input_b = b.flat<float>();

    auto out = output->template flat<float>();

    // Call the cuda kernel launcher
    CorrelationFlatKernelLauncher(input_a.data(),input_b.data(), out.data(), batch_size,num_rows,num_cols,depth,num_outputs,&offsets[0]);
  }
};

//REGISTER_KERNEL_BUILDER(Name("Correlation").Device(DEVICE_CPU), CorrelationOp);
REGISTER_KERNEL_BUILDER(Name("Correlation").Device(DEVICE_CPU), CorrelationFlatOp);
