#include "tensorflow/core/framework/op_kernel.h"

#include <limits>
#include <vector>

using namespace tensorflow;

class GreedyAssignmentOp : public OpKernel {
public:
    explicit GreedyAssignmentOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const auto& similarities_tensor = context->input(0);
        const auto& threshold_tensor = context->input(1);
        
        const auto num_targets = similarities_tensor.dim_size(0);
        const auto num_predictions = similarities_tensor.dim_size(1);

        Tensor* true_positives_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                                         {num_predictions},
                                                         &true_positives_tensor));

        const auto& similarities = similarities_tensor.matrix<float>();
        const auto& threshold = threshold_tensor.scalar<float>()();
        auto true_positives = true_positives_tensor->vec<bool>();
        
        // Initialize `true_positives` with false, therefore loop over
        // predictions can end early if all targets have been matched
        for (std::remove_const<decltype(num_predictions)>::type prediction{0};
             prediction < num_predictions;
             ++prediction)
            true_positives(prediction) = false;
        
        // Keep track of matched targets, the total number of true entries is
        // used to stop the loop over predictions
        std::vector<bool> target_matched(num_targets, false);
        size_t num_matched{0};
        for (std::remove_const<decltype(num_predictions)>::type prediction{0};
             prediction < num_predictions && num_matched < num_targets;
             ++prediction) {
            // Fine the max similarity that has not yet been matched
            auto best_similarity = -std::numeric_limits<float>::infinity();
            auto best_target = num_targets;
            for (std::remove_const<decltype(num_targets)>::type target{0};
                 target < num_targets;
                 ++target) {
                // Ignore matched targets
                if (target_matched[target])
                    continue;
                auto current_similarity = similarities(target, prediction);
                // Ignore candidate that does not exceed similarity threshold
                if (current_similarity < threshold)
                    continue;
                // Save target of better match
                if (best_target == num_targets
                        || best_similarity < current_similarity) {
                    best_target = target;
                    best_similarity = current_similarity;
                }
            }

            if (best_target != num_targets) {
                true_positives(prediction) = true;
                target_matched[best_target] = true;
                ++num_matched;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("GreedyAssignment").Device(DEVICE_CPU),
        GreedyAssignmentOp);
