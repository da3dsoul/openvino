// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {

class CustomOp : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomOp", "gpu_opset");

    CustomOp() = default;

    CustomOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", m_alpha);
        visitor.on_attribute("beta", m_beta);
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOp>(inputs[0], m_alpha, m_beta);
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());
        for (size_t i = 0; i < out.get_size(); i++) {
            out.data<float>()[i] = in.data<float>()[i] * m_alpha + m_beta;
        }
        return true;
    }
};

static std::shared_ptr<ov::Model> get_simple_model_with_custom_op() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto op = std::make_shared<CustomOp>(param, 1.0f, 2.0f);
    auto result = std::make_shared<ov::op::v0::Result>(op);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_custom_op");
}

TEST(CustomOp, CanReadValidCustomOpConfig) {
    ov::Core core;
    core.set_property(ov::test::utils::DEVICE_GPU, {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}});
}

TEST(CustomOp, NoRedundantReordersInserted) {
    ov::Core core;
    auto model = get_simple_model_with_custom_op();
    ov::AnyMap config = { ov::hint::inference_precision(ov::element::f32), {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);

    auto runtime_graph = compiled_model.get_runtime_model();

    auto ops = runtime_graph->get_ordered_ops();
    ASSERT_EQ(ops.size(), 3);
    ASSERT_STREQ(ops[0]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Input");
    ASSERT_STREQ(ops[1]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "CustomGPUPrimitive");
    ASSERT_STREQ(ops[2]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Result");
}

class CustomOpIntBuf : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpIntBuf", "gpu_opset");  // must match XML layer name

    CustomOpIntBuf() = default;

    CustomOpIntBuf(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpIntBuf>(inputs[0]);
    }

    bool has_evaluate() const override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());

        // minimal test: copy input to output
        for (size_t i = 0; i < out.get_size(); i++)
            out.data<float>()[i] = in.data<float>()[i];

        return true;
    }
};

static std::shared_ptr<ov::Model> get_simple_model_with_internal_buffer_static() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto op = std::make_shared<CustomOpIntBuf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_internal_buffer");
}

static std::shared_ptr<ov::Model> get_simple_model_with_internal_buffer_dynamic() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 2, 3, 4});
    auto op = std::make_shared<CustomOpIntBuf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_internal_buffer");
}

TEST(CustomOpIntBuf, InternalBufferKernelStaticRuns) {
    ov::Core core;

    ov::AnyMap config = { {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH} };
    auto model = get_simple_model_with_internal_buffer_static();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(1*2*3*4);
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = float(i);

    ov::Tensor input_tensor(ov::element::f32, {1,2,3,4}, input_data.data());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i], 1e-5);
    }
}

TEST(CustomOpIntBuf, InternalBufferKernelDynamicRuns) {
    ov::Core core;

    ov::AnyMap config = { {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH} };
    auto model = get_simple_model_with_internal_buffer_dynamic();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(1*2*3*4);
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = float(i);

    ov::Tensor input_tensor(ov::element::f32, {1,2,3,4}, input_data.data());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i], 1e-5);
    }
}

// Regression test for format="ANY" on a CustomLayer OUTPUT tensor.
//
// format::any is a sentinel (format.hpp: `any = -1`), and cldnn::layout's tensor
// constructor deliberately skips the logical-order remap for it:
//     sizes = fmt == format::any ? size.sizes()
//                                : size.sizes(get_default_format(fmt.dimension(), ...));
// so a format::any output layout was built from the RAW internal tensor order
// (b,f,x,y,z,...) instead of the format's logical order (b,f,y,x). Nothing reported an
// error: CreateCustomOp's WorkSizes B/F/Y/X resolution simply read the wrong extents,
// and downstream nodes saw a shape with its spatial axes transposed.
//
// The shape is deliberately non-square in Y and X (Y=3, X=4), the WorkSizes formula
// dispatches over the individual X and Y extents rather than their product, and the
// kernel writes a value derived from its dispatch coordinates and the declared output
// width. All three are required for the defect to be observable: with a product-form
// WorkSizes and a flat copy kernel, a transposed layout cancels out exactly and the
// test cannot fail.
class CustomOpAnyFmt : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmt", "gpu_opset");

    CustomOpAnyFmt() = default;

    explicit CustomOpAnyFmt(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmt>(inputs[0]);
    }
};

TEST(CustomOpAnyFmt, OutputFormatAnyPreservesAxisOrder) {
    ov::Core core;

    constexpr size_t kY = 3;
    constexpr size_t kX = 4;
    const ov::Shape shape{1, 1, kY, kX};

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpAnyFmt>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_format_output");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(ov::shape_size(shape), 0.0f);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);

    const float* out = output_tensor.data<const float>();
    for (size_t y = 0; y < kY; y++) {
        for (size_t x = 0; x < kX; x++) {
            ASSERT_NEAR(out[y * kX + x], static_cast<float>(y * 1000 + x), 1e-5f)
                << "at y=" << y << " x=" << x;
        }
    }
}


// Same defect as above, on a rank-3 output. A 3D output resolves through
// get_default_format(3) rather than (4) and is the shape the defect was originally
// found on, so it is covered separately rather than assumed equivalent.
class CustomOpAnyFmt3D : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmt3D", "gpu_opset");

    CustomOpAnyFmt3D() = default;

    explicit CustomOpAnyFmt3D(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmt3D>(inputs[0]);
    }
};

TEST(CustomOpAnyFmt3D, OutputFormatAnyPreservesAxisOrder3D) {
    ov::Core core;

    constexpr size_t kB = 2;
    constexpr size_t kF = 3;
    constexpr size_t kY = 4;
    const ov::Shape shape{kB, kF, kY};

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpAnyFmt3D>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_format_3d_output");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(ov::shape_size(shape), 0.0f);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);

    const float* out = output_tensor.data<const float>();
    for (size_t b = 0; b < kB; b++) {
        for (size_t f = 0; f < kF; f++) {
            for (size_t y = 0; y < kY; y++) {
                ASSERT_NEAR(out[(b * kF + f) * kY + y],
                            static_cast<float>(b * 100 + f * 10 + y), 1e-5f)
                    << "at b=" << b << " f=" << f << " y=" << y;
            }
        }
    }
}


// Regression test for the pre-reorder primitive id collision. The reorder inserted for
// a custom layer input is named from the producer primitive id plus the custom op
// friendly name, with no port index, so two input ports fed by the SAME producer used
// to generate identical ids and compilation failed with
// "Different primitive with id ... exists already".
class CustomOpTwoIn : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpTwoIn", "gpu_opset");

    CustomOpTwoIn() = default;

    CustomOpTwoIn(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) : Op({a, b}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpTwoIn>(inputs[0], inputs[1]);
    }
};

TEST(CustomOpTwoIn, SameProducerOnTwoPortsCompiles) {
    ov::Core core;

    const ov::Shape shape{1, 2, 3, 4};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    // Both ports deliberately read the same producer tensor.
    auto op = std::make_shared<CustomOpTwoIn>(param, param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_shared_producer_on_two_ports");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    ov::CompiledModel compiled_model;
    ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));

    auto infer_request = compiled_model.create_infer_request();
    std::vector<float> input_data(ov::shape_size(shape));
    for (size_t i = 0; i < input_data.size(); i++)
        input_data[i] = static_cast<float>(i);

    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();
    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i] * 2.0f, 1e-5f) << "element " << i;
    }
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
