#include <cuda_runtime_api.h>
#include <algorithm>
#include "resize-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

static const unsigned MAX_BLOCK_SIZE = 512U;
static unsigned getNumThreads(int nElem) {
    unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
    for (int i = 0; i < 5; ++i) {
        if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
            return threadSizes[i];
        }
    }
    return MAX_BLOCK_SIZE;
}

template <typename xpu, typename DType, typename AccReal>
__global__ void _BilinearKernelForward(const int N,
                                       const AccReal scale_h,
                                       const AccReal scale_w,
                                       const Tensor<xpu, 4, DType> in_data,
                                       Tensor<xpu, 4, DType> out_data) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int batchSize = in_data.size(0);
    const int channels = in_data.size(1);
    const int in_height = in_data.size(2);
    const int in_width  = in_data.size(3);
    const int out_height = out_data.size(2);
    const int out_width  = out_data.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;

    // Identity copy
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < channels; ++c) {
                const DType val = in_data[n][c][out_h][out_w];
                out_data[n][c][out_h][out_w] = val;
            }
        }
        return;
    }

    const AccReal in_hr = scale_h * out_h;
    const AccReal in_wr = scale_w * out_w;
    const int in_h0 = in_hr;
    const int in_w0 = in_wr;
    const int in_h1 = (in_h0 < in_height - 1) ? (in_h0 + 1) : in_h0;
    const int in_w1 = (in_w0 < in_width  - 1) ? (in_w0 + 1) : in_w0;
    const AccReal h1_weight = in_hr - in_h0;
    const AccReal w1_weight = in_wr - in_w0;
    const AccReal h0_weight = (AccReal)1 - h1_weight;
    const AccReal w0_weight = (AccReal)1 - w1_weight;

    for (int n = 0; n < batchSize; ++n) {
        for (int  c = 0; c < channels; ++c) {
            const AccReal val = h0_weight * (w0_weight * in_data[n][c][in_h0][in_w0] +
                                             w1_weight * in_data[n][c][in_h0][in_w1]) +
                                h1_weight * (w0_weight * in_data[n][c][in_h1][in_w0] +
                                             w1_weight * in_data[n][c][in_h1][in_w1]);
            out_data[n][c][out_h][out_w] = (DType)val;
        }
    }
}

template <typename xpu, typename DType, typename AccReal>
__global__ void _BilinearKernelBackward(const int N,
                                        const AccReal scale_h,
                                        const AccReal scale_w,
                                        Tensor<xpu, 4, DType> in_grad,
                                        const Tensor<xpu, 4, DType> out_grad) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int batchSize = in_grad.size(0);
    const int channels = in_grad.size(1);
    const int in_height = in_grad.size(2);
    const int in_width  = in_grad.size(3);
    const int out_height = out_grad.size(2);
    const int out_width  = out_grad.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;
    
    // Identity copy
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < batchSize; ++n) {
            for (int c= 0; c < channels; ++c) {
                const DType val = out_grad[n][c][out_h][out_w];
                in_grad[n][c][out_h][out_w] += val;
            }
        }
        return;
    }

    const AccReal in_hr = scale_h * out_h;
    const AccReal in_wr = scale_w * out_w;
    const int in_h0 = in_hr;
    const int in_w0 = in_wr;
    const int in_h1 = (in_h0 < in_height - 1) ? (in_h0 + 1) : in_h0;
    const int in_w1 = (in_w0 < in_width  - 1) ? (in_w0 + 1) : in_w0;
    const AccReal h1_weight = in_hr - in_h0;
    const AccReal w1_weight = in_wr - in_w0;
    const AccReal h0_weight = (AccReal)1 - h1_weight;
    const AccReal w0_weight = (AccReal)1 - w1_weight;

    for (int n = 0; n < batchSize; ++n) {
        for (int c = 0; c < channels; ++c) {
            const DType val = out_grad[n][c][out_h][out_w];
            atomicAdd(&in_grad[n][c][in_h0][in_w0], (DType)(h0_weight * w0_weight * val));
            atomicAdd(&in_grad[n][c][in_h0][in_w1], (DType)(h0_weight * w1_weight * val));
            atomicAdd(&in_grad[n][c][in_h1][in_w0], (DType)(h1_weight * w0_weight * val));
            atomicAdd(&in_grad[n][c][in_h1][in_w1], (DType)(h1_weight * w1_weight * val));
        }
    }
}

template <typename xpu, typename DType, typename AccReal>
__global__ void _NearestKernelForward(const int N,
                                      const AccReal scale_h,
                                      const AccReal scale_w,
                                      const Tensor<xpu, 4, DType> in_data,
                                      Tensor<xpu, 4, DType> out_data) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int nBatch = in_data.size(0);
    const int nChannel = in_data.size(1);
    const int in_height = in_data.size(2);
    const int in_width  = in_data.size(3);
    const int out_height = out_data.size(2);
    const int out_width  = out_data.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;
    
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannel; ++c) {
                const DType val = in_data[n][c][out_h][out_w];
                out_data[n][c][out_h][out_w] = val;
            }
        }
        return;
    }

    const AccReal in_hr = scale_h * out_h;
    const AccReal in_wr = scale_w * out_w;
    int in_hf = in_hr;
    int in_wf = in_wr;
    const int in_h = (in_hr - in_hf < 0.5) ? in_hf :
                     ((in_hf < in_height - 1) ? (in_hf + 1) : in_hf);
    const int in_w = (in_wr - in_wf < 0.5) ? in_wf :
                     ((in_wf < in_width - 1)  ? (in_wf + 1) : in_wf);

    for (int n = 0; n < nBatch; ++n) {
        for (int c = 0; c < nChannel; ++c) {
            const DType val = in_data[n][c][in_h][in_w];
            out_data[n][c][out_h][out_w] = val;
        }
    }
}

template <typename xpu, typename DType, typename AccReal>
__global__ void _NearestKernelBackward(const int N,
                                       const AccReal scale_h,
                                       const AccReal scale_w,
                                       Tensor<xpu, 4, DType> in_grad,
                                       const Tensor<xpu, 4, DType> out_grad) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int nBatch = in_grad.size(0);
    const int nChannel = in_grad.size(1);
    const int in_height = in_grad.size(2);
    const int in_width  = in_grad.size(3);
    const int out_height = out_grad.size(2);
    const int out_width  = out_grad.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;
    
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannel; ++c) {
                const DType val = out_grad[n][c][out_h][out_w];
                in_grad[n][c][out_h][out_w] += val;
            }
        }
        return;
    }

    const AccReal in_hr = scale_h * out_h;
    const AccReal in_wr = scale_w * out_w;
    int in_hf = in_hr;
    int in_wf = in_wr;
    const int in_h = (in_hr - in_hf < 0.5) ? in_hf :
                     ((in_hf < in_height - 1) ? (in_hf + 1) : in_hf);
    const int in_w = (in_wr - in_wf < 0.5) ? in_wf :
                     ((in_wf < in_width - 1)  ? (in_wf + 1) : in_wf);

    for (int n = 0; n < nBatch; ++n) {
        for (int c = 0; c < nChannel; ++c) {
            const DType val = out_grad[n][c][out_h][out_w];
            atomicAdd(&in_grad[n][c][in_h][in_w], (DType)val);
        }
    }
}

template <typename xpu, typename DType, typename AccReal>
__global__ void _AreaKernelForward(const int N,
                                   const AccReal scale_h,
                                   const AccReal scale_w,
                                   const Tensor<xpu, 4, DType> in_data,
                                   Tensor<xpu, 4, DType> out_data) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int nBatch = in_data.size(0);
    const int nChannel = in_data.size(1);
    const int in_height = in_data.size(2);
    const int in_width  = in_data.size(3);
    const int out_height = out_data.size(2);
    const int out_width  = out_data.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;
    
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannel; ++c) {
                const DType val = in_data[n][c][out_h][out_w];
                out_data[n][c][out_h][out_w] = val;
            }
        }
        return;
    }

    const AccReal in_hr0 = scale_h * out_h;
    const AccReal in_hr1 = in_hr0 + scale_h;
    const int in_h0 = in_hr0;
    const int in_h1 = ((in_hr1 - (int)in_hr1 > 0.f) && (in_hr1 < in_height)) ? ((int)in_hr1 + 1) : (int)in_hr1;
    const AccReal scale_h0 = ((in_hr1 < in_h0 + 1) ? in_hr1 : (in_h0 + 1)) - in_hr0;
    const AccReal scale_h1 = in_hr1 - ((in_h1 - 1 > in_hr0) ? (in_h1 - 1) : in_hr0);

    const AccReal in_wr0 = scale_w * out_w;
    const AccReal in_wr1 = in_wr0 + scale_w;
    const int in_w0 = in_wr0;
    const int in_w1 = ((in_wr1 - (int)in_wr1 > 0) && (in_wr1 < in_width)) ? ((int)in_wr1 + 1) : (int)in_wr1;
    const AccReal scale_w0 = ((in_wr1 < in_w0 + 1) ? in_wr1 : (in_w0 + 1)) - in_wr0;
    const AccReal scale_w1 = in_wr1 - ((in_w1 - 1 > in_wr0) ? (in_w1 - 1) : in_wr0);

    const AccReal scale_t = scale_h * scale_w;
    //const int len_cache = (in_h1 - in_h0 > 0) ? (in_h1 - in_h0) : 1;
    const int bound_h = in_h1 - in_h0 - 1;
    const int bound_w = in_w1 - in_w0 - 1;
    //std::vector<AccReal> cache(len_cache);
    for (int n = 0; n < nBatch; ++n) {
        for (int c = 0; c < nChannel; ++c) {
            AccReal val = scale_h0 * scale_w0 * in_data[n][c][in_h0][in_w0];
            for (int _w = 1; _w < bound_w; ++_w) {
                val += scale_h0 * in_data[n][c][in_h0][in_w0 + _w];
                for (int _h = 1; _h < bound_h; ++_h) {
                    val += in_data[n][c][in_h0 + _h][in_w0 + _w];
                }
            }
            for (int _h = 1; _h < bound_h; ++_h) {
                val += scale_w0 * in_data[n][c][in_h0 + _h][in_w0];
            }
            if (bound_w > 0) {
                val += scale_w1 * scale_h0 * in_data[n][c][in_h0][in_w1 - 1];
                for (int _h = 1; _h < bound_h; ++_h) {
                    val += scale_w1 * in_data[n][c][in_h0 + _h][in_w1 - 1];
                }
            }
            if (bound_h > 0) {
                val += scale_h1 * scale_w0 * in_data[n][c][in_h1 - 1][in_w0];
                for (int _w = 1; _w < bound_w; ++_w) {
                    val += scale_h1 * in_data[n][c][in_h1 - 1][in_w0 + _w];
                }
            }
            if ((bound_w > 0) && (bound_h > 0)) {
                val += scale_h1 * scale_w1 * in_data[n][c][in_h1 - 1][in_w1 - 1];
            }
            
            /*
            for (int _h = 0; _h < len_cache; ++_h) {
                cache[_h] = scale_w0 * in_data[n][c][in_h0+_h][in_w0];
                for (int _w = 1; _w < in_w1 - in_w0 - 1; ++_w) {
                    cache[_h] += in_data[n][c][in_h0+_h][in_w0+_w];
                }
                if (in_w1 > in_w0 + 1) {
                    cache[_h] += scale_w1 * in_data[n][c][in_h0+_h][in_w1-1];
                }
            }
            AccReal val = scale_h0 * cache[0];
            for (int _h = 1; _h < in_h1 - in_h0 - 1; ++_h) {
                val += cache[_h];
            }
            if (in_h1 > in_h0 + 1) {
                val += scale_h1 * cache[in_h1 - in_h0 - 1];
            }
            */
            out_data[n][c][out_h][out_w] = (DType)(val / scale_t);
        }
    }
}


template <typename xpu, typename DType, typename AccReal>
__global__ void _AreaKernelBackward(const int N,
                                    const AccReal scale_h,
                                    const AccReal scale_w,
                                    Tensor<xpu, 4, DType> in_grad,
                                    const Tensor<xpu, 4, DType> out_grad) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx >= N) {
        return;
    }

    const int nBatch = in_grad.size(0);
    const int nChannel = in_grad.size(1);
    const int in_height = in_grad.size(2);
    const int in_width  = in_grad.size(3);
    const int out_height = out_grad.size(2);
    const int out_width  = out_grad.size(3);

    const int out_h = Idx / out_width;
    const int out_w = Idx % out_width;
    
    if (in_height == out_height && in_width == out_width) {
        for (int n = 0; n < nBatch; ++n) {
            for (int c = 0; c < nChannel; ++c) {
                const DType val = out_grad[n][c][out_h][out_w];
                in_grad[n][c][out_h][out_w] += val;
            }
        }
        return;
    }

    const AccReal in_hr = scale_h * out_h;
    const AccReal in_wr = scale_w * out_w;
    int in_hf = in_hr;
    int in_wf = in_wr;
    const int in_h = (in_hr - in_hf < 0.5) ? in_hf :
                     ((in_hf < in_height - 1) ? (in_hf + 1) : in_hf);
    const int in_w = (in_wr - in_wf < 0.5) ? in_wf :
                     ((in_wf < in_width - 1)  ? (in_wf + 1) : in_wf);

    for (int n = 0; n < nBatch; ++n) {
        for (int c = 0; c < nChannel; ++c) {
            const DType val = out_grad[n][c][out_h][out_w];
            atomicAdd(&in_grad[n][c][in_h][in_w], (DType)val);
        }
    }
}


template <typename xpu, typename DType, typename AccReal>
void _ResizeForward(mshadow::Stream<gpu>* s,
                    const std::vector<TBlob>& input,
                    const std::vector<TBlob>& output,
                    const ResizeParam& param) {
                    //const bool align_corners) {
    Tensor<xpu, 4, DType> in_data = input[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out_data = output[0].get<xpu, 4, DType>(s);

    const int in_height = in_data.size(2);
    const int in_width  = in_data.size(3);
    const int out_height = out_data.size(2);
    const int out_width  = out_data.size(3);

    const int offset = param.align_corners ? 1 : 0;
    const AccReal scale_h = (out_height > 1) ? (AccReal)(in_height - offset) /
                            (out_height - offset) : (AccReal)0;
    const AccReal scale_w = (out_width > 1) ? (AccReal)(in_width - offset) / 
                            (out_width - offset) : (AccReal)0;

    const int num_kernels = out_height * out_width;
    const int num_threads = getNumThreads(num_kernels);
    dim3 blocks(static_cast<int>((num_kernels + num_threads - 1) / num_threads));
    dim3 threads(num_threads);
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

    if (param.method == resize_enum::kBilinear) {
        _BilinearKernelForward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_data, out_data);
    } else if (param.method == resize_enum::kNearest) {
        _NearestKernelForward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_data, out_data);
    } else if (param.method == resize_enum::kArea) {
        _AreaKernelForward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_data, out_data);
    } else {
        LOG(FATAL) << "unknown sample type";
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(_ResizeForward);
}

template <typename xpu, typename DType, typename AccReal>
void _ResizeBackward(mshadow::Stream<gpu>* s,
                     const std::vector<TBlob>& input,
                     const std::vector<TBlob>& output,
                     const ResizeParam& param) {
    Tensor<xpu, 4, DType> in_grad = output[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out_grad = input[0].get<xpu, 4, DType>(s);

    const int in_height = in_grad.size(2);
    const int in_width  = in_grad.size(3);
    const int out_height = out_grad.size(2);
    const int out_width  = out_grad.size(3);

    const int offset = param.align_corners ? 1 : 0;
    const AccReal scale_h = (out_height > 1) ? (AccReal)(in_height - offset) /
                            (out_height - offset) : (AccReal)0;
    const AccReal scale_w = (out_width > 1) ? (AccReal)(in_width - offset) / 
                            (out_width - offset) : (AccReal)0;

    const int num_kernels = out_height * out_width;
    const int num_threads = getNumThreads(num_kernels);
    dim3 blocks(static_cast<int>((num_kernels + num_threads - 1) / num_threads));
    dim3 threads(num_threads);
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

    if (param.method == resize_enum::kBilinear) {
        _BilinearKernelBackward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_grad, out_grad);
    } else if (param.method == resize_enum::kNearest) {
        _NearestKernelBackward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_grad, out_grad);
    } else if (param.method == resize_enum::kArea) {
        _AreaKernelBackward<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(
                num_kernels, scale_h, scale_w, in_grad, out_grad);
    } else {
        LOG(FATAL) << "unknown sample type";
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(_ResizeBackward);
}

NNVM_REGISTER_OP(_contrib_Resize)
.set_attr<FCompute>("FCompute<gpu>", ResizeOpForward<gpu>);

NNVM_REGISTER_OP(_contrib_Resize_backward)
.set_attr<FCompute>("FCompute<gpu>", ResizeOpBackward<gpu>);

}   // namespace op
}   // namespace mxnet
