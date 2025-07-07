import numpy as np

from . import metrics


def run_method_multiple_seeds(method, data, seeds):
    estimates = []
    runtimes = []

    for seed in seeds:
        estimate, runtime = method(data, seed)
        estimates.append(estimate)
        runtimes.append(runtime)
    return estimates, runtimes


def run_methods_on_image_pair(methods, data, seeds):
    results = {}

    for method_name, function_handle in methods.items():
        estimates, runtimes = run_method_multiple_seeds(function_handle, data, seeds)
        image_pair_errors = metrics.calculate_multipose_errors(estimates, data)

        # Create dict entries for method
        method_results = results.setdefault(method_name, {})
        method_results['average_errors'] = {}
        method_results['trials'] = {}

        # Store averages
        method_results['average_runtime'] = np.mean(runtimes)
        for k, v in image_pair_errors.items():
            method_results['average_errors'][k] = np.mean(v)

        # Store estimates, errors, and runtimes per trial
        for i, s in enumerate(seeds):
            trial_results = method_results['trials'].setdefault(str(s), {})
            trial_results['estimate'] = estimates[i]
            trial_results['runtime'] = runtimes[i]

            # Store errors
            trial_results.setdefault('errors', {'R': [], 't': [], 'max': []})
            for k, v in image_pair_errors.items():
                trial_results['errors'][k] = v[i]

    return results

