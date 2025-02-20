#include <fstream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/runtime_op_utility.h>

class NncaseModel
{
public:
    NncaseModel(const std::string &kmodel_file) {
        std::ifstream ifs(kmodel_file, std::ios::binary);
        interpreter_.load_model(ifs).unwrap_or_throw();
        entry_function_ = interpreter_.entry_function().unwrap_or_throw();
    }

    ~NncaseModel() {}

    nncase::value_t run(std::vector<nncase::value_t> &inputs) {
        return entry_function_->invoke(inputs).unwrap_or_throw();
    }

    const nncase::runtime::runtime_function *entry() {
        return entry_function_;
    }

private:
    nncase::runtime::interpreter interpreter_;
    nncase::runtime::runtime_function *entry_function_;
};