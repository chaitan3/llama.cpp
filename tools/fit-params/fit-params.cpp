#include "llama.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (!params.fit_params_print) {
        const common_params_fit_status status = common_fit_params(params.model.get_paths(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                nullptr,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("%s\n", common_fit_args_string(&mparams, &cparams).c_str());
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.get_paths(), &mparams, &cparams);
    }

    return 0;
}
