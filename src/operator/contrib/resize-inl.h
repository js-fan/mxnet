#ifndef MXNET_OPERATOR_CONTRIB_RESIZE_COMMON_INL_H_
#define MXNET_OPERATOR_CONTRIB_RESIZE_COMMON_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>

#include "../../ndarray/ndarray_function.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace resize_enum {
enum ResizeMethod {kNearest, kBilinear, kArea};
}


struct ResizeParam : public dmlc::Parameter<ResizeParam> {
    int height;
    int width;
    int method;
    bool align_corners;
    DMLC_DECLARE_PARAMETER(ResizeParam) {
        DMLC_DECLARE_FIELD(height).set_range(1, 65536)
            .describe("Output height (required)");
        DMLC_DECLARE_FIELD(width).set_range(1, 65536)
            .describe("Output width (required)");
        DMLC_DECLARE_FIELD(align_corners).set_default(true)
            .describe("Whether minus size by 1 before computing the scale (default True)");
        DMLC_DECLARE_FIELD(method)
            .add_enum("nearest", resize_enum::kNearest)
            .add_enum("bilinear", resize_enum::kBilinear)
            .add_enum("area", resize_enum::kArea)
            .describe("Interpolation method, {nearest, bilinear, area} (required)");
    }
};

template <typename xpu, typename DType, typename AccReal>
void _ResizeForward(mshadow::Stream<cpu>* s,
                            const std::vector<TBlob>& input,
                            const std::vector<TBlob>& output,
                            const ResizeParam& param);

template <typename xpu, typename DType, typename AccReal>
void _ResizeBackward(mshadow::Stream<cpu>* s,
                            const std::vector<TBlob>& input,
                            const std::vector<TBlob>& output,
                            const ResizeParam& param);

#if MXNET_USE_CUDA
template <typename xpu, typename DType, typename AccReal>
void _ResizeForward(mshadow::Stream<gpu>* s,
                            const std::vector<TBlob>& input,
                            const std::vector<TBlob>& output,
                            const ResizeParam& param);

template <typename xpu, typename DType, typename AccReal>
void _ResizeBackward(mshadow::Stream<gpu>* s,
                             const std::vector<TBlob>& input,
                             const std::vector<TBlob>& output,
                             const ResizeParam& param);
#endif


template <typename xpu>
void ResizeOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
    MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
            _ResizeForward<xpu, DType, AccReal>(
                    s, inputs, outputs, param);
    });
}

template <typename xpu>
void ResizeOpBackward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
    // Zero grad
    if (req[0] == kWriteTo || req[0] == kWriteInplace) {
        MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
                Fill<false>(s, outputs[0], kWriteTo, 0);
        })
    }
    MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
            _ResizeBackward<xpu, DType, AccReal>(
                    s, inputs, outputs, param);
    });
}


static bool ResizeOpInferShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_shape,
                               std::vector<TShape>* out_shape) {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    CHECK_EQ(out_shape->size(), 1U) << "Output:[data]";
    const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);

    TShape dshape(in_shape->at(0));
    if (dshape.ndim() == 0)
        return false;

    dshape[2] = param.height;
    dshape[3] = param.width;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
}

static bool ResizeOpInferType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_type,
                              std::vector<int>* out_type) {
    using namespace mshadow;
    CHECK_EQ(in_type->size(), 1U);

    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";

    int dtype_param = 0;
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
            dtype_param = mshadow::DataType<AccRealX>::kFlag;
    });
    out_type->clear();
    out_type->push_back(dtype_param);
    return true;
}

static inline bool ResizeOpStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1);
    CHECK_EQ(out_attrs->size(), 1);
    *dispatch_mode = DispatchMode::kFCompute;
    for (int& v : *in_attrs) {
        if (v == -1) {
            v = kDefaultStorage;
        }
    }
    for (size_t i = 0; i < out_attrs->size(); ++i) {
        (*out_attrs)[i] = kDefaultStorage;
    }
    return true;
}

}   // namespace op
}   // namespace mxnet
#endif      // MXNET_OPERATOR_CONTRIB_RESIZE_COMMON_INL_H_
