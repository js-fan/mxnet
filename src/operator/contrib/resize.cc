#include "resize-inl.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

using namespace mshadow;

template <typename xpu, typename DType, typename AccReal>
void _ResizeForward(mshadow::Stream<cpu>* s,
                           const std::vector<TBlob>& input,
                           const std::vector<TBlob>& output,
                           const ResizeParam& param) {
    Tensor<xpu, 4, DType> itensor = input[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> otensor = output[0].get<xpu, 4, DType>(s);
    const int nBatch = itensor.size(0);
    const int nChannel = itensor.size(1);
    const int out_height = otensor.size(2);
    const int out_width  = otensor.size(3);
    const int in_height = itensor.size(2);
    const int in_width  = itensor.size(3);

    DType* in_data = itensor.dptr_;
    DType* out_data = otensor.dptr_;
    const int nKernel = nBatch * nChannel;
    const int step_in = in_width * in_height;
    const int step_out = out_width * out_height;

    // copy
    if (in_height == out_height && in_width == out_width) {
        for (int _h = 0; _h < out_height; ++_h) {
            for (int _w = 0; _w < out_width; ++_w) {
                const DType* pos_in = &in_data[_h * in_width + _w];
                DType* pos_out = &out_data[_h * out_width + _w];
                for (int k = 0; k < nKernel; ++k) {
                    pos_out[0] = pos_in[0];
                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
        return;
    }

    const int offset = param.align_corners ? 1 : 0;
    const float scale_h = (out_height > 1) ? static_cast<float>(in_height - offset) /
                          (out_height - offset) : 0.f;
    const float scale_w = (out_width  > 1) ? static_cast<float>(in_width  - offset) /
                          (out_width  - offset) : 0.f;

    if (param.method == resize_enum::kNearest) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            const float in_hr = scale_h * out_h;
            int in_hf = in_hr;
            const int in_h = (in_hr - in_hf < 0.5) ? in_hf :
                             ((in_hf < in_height - 1) ? (in_hf + 1) : in_hf);
            for (int out_w = 0; out_w < out_width; ++out_w) {
                const float in_wr = scale_w * out_w;
                int in_wf = in_wr;
                const int in_w = (in_wr - in_wf < 0.5) ? in_wf :
                                 ((in_wf < in_width - 1) ? (in_wf + 1) : in_wf);
                const DType* pos_in = &in_data[in_h * in_width + in_w];
                DType* pos_out = &out_data[out_h * out_width + out_w];
                for (int k = 0; k < nKernel; ++k) {
                    pos_out[0] = pos_in[0];
                    pos_out += step_out;
                    pos_in  += step_in;
                }
            }
        }
    } else if (param.method == resize_enum::kBilinear) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            const float in_hr = scale_h * out_h;
            const int in_h = in_hr;
            const int in_hp = (in_h < in_height - 1) ? 1 : 0;
            const DType h1_weight = in_hr - in_h;
            const DType h0_weight = (DType)1 - h1_weight;

            for (int out_w = 0; out_w < out_width; ++out_w) {
                const float in_wr = scale_w * out_w;
                const int in_w = in_wr;
                const int in_wp = (in_w < in_width - 1) ? 1 : 0;
                const DType w1_weight = in_wr - in_w;
                const DType w0_weight = (DType)1 - w1_weight;

                const DType* pos_in = &in_data[in_h * in_width + in_w];
                DType* pos_out = &out_data[out_h * out_width + out_w];

                for (int k = 0; k < nKernel; ++k) {
                    pos_out[0] = h0_weight * (w0_weight * pos_in[0] + w1_weight * pos_in[in_wp]) +
                                 h1_weight * (w0_weight * pos_in[in_hp * in_width] +
                                              w1_weight * pos_in[in_hp * in_width + in_wp]);
                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
    } else if (param.method == resize_enum::kArea) {
        const float scale_total = scale_h * scale_w;
        for (int out_h = 0; out_h < out_height; ++out_h) {
            const float in_hr0 = scale_h * out_h;
            const float in_hr1 = in_hr0 + scale_h;
            const int in_h0 = in_hr0;
            const int in_h1 = ((in_hr1 - (int)in_hr1 > 0.f) && (in_hr1 < in_height)) ?
                              ((int)in_hr1 + 1) : (int)in_hr1;
            const DType scale_h0 = ((in_hr1 < in_h0 + 1) ? in_hr1 : (in_h0 + 1)) - in_hr0;
            const DType scale_h1 = in_hr1 - ((in_h1 - 1 > in_hr0) ? (in_h1 - 1) : in_hr0);

            const int step_cache = (in_h1 - in_h0 > 0) ? (in_h1 - in_h0) : 1;
            std::vector<DType> cache_sum(step_cache);
            for (int out_w = 0; out_w < out_height; ++out_w) {
                const float in_wr0 = scale_w * out_w;
                const float in_wr1 = in_wr0 + scale_w;
                const int in_w0 = in_wr0;
                const int in_w1 = ((in_wr1 - (int)in_wr1 > 0.f) && (in_wr1 < in_width)) ?
                                  ((int)in_wr1 + 1) : (int)in_wr1;
                const DType scale_w0 = ((in_wr1 < in_w0 + 1) ? in_wr1 : (in_w0 + 1)) - in_wr0;
                const DType scale_w1 = in_wr1 - ((in_w1 - 1 > in_wr0) ? (in_w1 - 1) : in_wr0);

                const DType* pos_in = &in_data[in_h0 * in_width + in_w0];
                DType* pos_out = &out_data[out_h * out_width + out_w];
                for (int k = 0; k < nKernel; ++k) {
                    for (int _h = 0; _h < step_cache; ++_h) {
                        cache_sum[_h] = scale_w0 * pos_in[in_width * _h];
                        for (int _w = 1; _w < in_w1 - in_w0 - 1; ++_w) {
                            cache_sum[_h] += pos_in[in_width * _h + _w];
                        }
                        if (in_w1 > in_w0 + 1) {
                            cache_sum[_h] += scale_w1 * pos_in[in_width * _h + in_w1 - in_w0 - 1];
                        }
                    }
                    pos_out[0] = scale_h0 * cache_sum[0];
                    for (int _h = 1; _h < in_h1 - in_h0 - 1; ++_h) {
                        pos_out[0] += cache_sum[_h];
                    }
                    if (in_h1 > in_h0 + 1) {
                        pos_out[0] += scale_h1 * cache_sum[in_h1 - in_h0 - 1];
                    }
                    pos_out[0] /= scale_total;

                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
    } else {
        LOG(FATAL) << "unknown sample type";
    }
}

template <typename xpu, typename DType, typename AccReal>
void _ResizeBackward(mshadow::Stream<cpu>* s,
                            const std::vector<TBlob>& input,
                            const std::vector<TBlob>& output,
                            const ResizeParam& param) {
    Tensor<xpu, 4, DType> itensor = output[0].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> otensor = input[0].get<xpu, 4, DType>(s);
    const int nBatch = itensor.size(0);
    const int nChannel = itensor.size(1);
    const int out_height = otensor.size(2);
    const int out_width = otensor.size(3);
    const int in_height = itensor.size(2);
    const int in_width = itensor.size(3);

    DType* in_grad = itensor.dptr_;
    DType* out_grad = otensor.dptr_;
    const int nKernel = nBatch * nChannel;
    const int step_in = in_width * in_height;
    const int step_out = out_width * out_height;

    if (in_height == out_height && in_width == out_width) {
        for (int _h = 0; _h < out_height; ++_h) {
            for (int _w = 0; _w < out_width; ++_w) {
                DType* pos_in = &in_grad[_h * in_width + _w];
                const DType* pos_out = &out_grad[_h * out_width + _w];
                for (int k = 0; k < nKernel; ++k) {
                    pos_in[0] += pos_out[0];
                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
        return;
    }

    const int offset = param.align_corners ? 1 : 0;
    const float scale_h = (out_height > 1) ? static_cast<float>(in_height - offset) /
                          (out_height - offset) : 0.f;
    const float scale_w = (out_width  > 1) ? static_cast<float>(in_width  - offset) /
                          (out_width  - offset) : 0.f;

    if (param.method == resize_enum::kNearest || param.method == resize_enum::kArea) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            const float in_hr = scale_h * out_h;
            int in_hf = in_hr;
            const int in_h = (in_hr - in_hf < 0.5) ? in_hf :
                             ((in_hf < in_height - 1) ? (in_hf + 1) : in_hf);
            for (int out_w = 0; out_w < out_width; ++out_w) {
                const float in_wr = scale_w * out_w;
                int in_wf = in_wr;
                const int in_w = (in_wr - in_wf < 0.5) ? in_wf :
                                 ((in_wf < in_width - 1) ? (in_wf + 1) : in_wf);
                DType* pos_in = &in_grad[in_h * in_width + in_w];
                const DType* pos_out = &out_grad[out_h * out_width + out_w];
                for (int k = 0; k < nKernel; ++k) {
                    pos_in[0] += pos_out[0];
                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
    } else if (param.method == resize_enum::kBilinear) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            const float in_hr = scale_h * out_h;
            const int in_h = in_hr;
            const int in_hp = (in_h < in_height - 1) ? 1 : 0;
            const DType h1_weight = in_hr - in_h;
            const DType h0_weight = (DType)1 - h1_weight;

            for (int out_w = 0; out_w < out_width; ++out_w) {
                const float in_wr = scale_w * out_w;
                const int in_w = in_wr;
                const int in_wp = (in_w < in_width - 1) ? 1 : 0;
                const DType w1_weight = in_wr - in_w;
                const DType w0_weight = (DType)1 - w1_weight;

                DType* pos_in = &in_grad[in_h * in_width + in_w];
                const DType* pos_out = &out_grad[out_h * out_width + out_w];

                for (int k = 0; k < nKernel; ++k) {
                    pos_in[0] += w0_weight * h0_weight * pos_out[0];
                    pos_in[in_wp] += w1_weight * h0_weight * pos_out[0];
                    pos_in[in_hp * in_width] += w0_weight * h1_weight * pos_out[0];
                    pos_in[in_hp * in_width + in_wp] += w1_weight * h1_weight * pos_out[0];

                    pos_in += step_in;
                    pos_out += step_out;
                }
            }
        }
    } else {
        LOG(FATAL) << "unknown sample type";
    }
}


DMLC_REGISTER_PARAMETER(ResizeParam);

NNVM_REGISTER_OP(_contrib_Resize)
.describe(R"code(
Perform resize on 4D Tensor.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ResizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ResizeOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", ResizeOpInferType)
.set_attr<FInferStorageType>("FInferStorageType", ResizeOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", ResizeOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_Resize_backward"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(ResizeParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_Resize_backward)
.set_attr_parser(ParamParser<ResizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", ResizeOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", ResizeOpBackward<cpu>);


}   // namespace op
}   // namespace mxnet
