#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("GreedyAssignment")
    .Input("similarities: float")
    .Input("threshold: float")
    .Output("true_positives: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle similarities_shape;
        ::tensorflow::shape_inference::ShapeHandle threshold_shape;
        TF_RETURN_IF_ERROR(
                c->WithRankAtLeast(c->input(0), 2, &similarities_shape));
        TF_RETURN_IF_ERROR(
                c->WithRankAtLeast(c->input(1), 0, &threshold_shape));
        return Status::OK();
    });
