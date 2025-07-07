import matplotlib.pyplot as plt
import numpy as np

from . import metrics


PLOT_STYLES = {
    'ours_CCC': {'marker': 's', 'markersize': 8, 'color': 'tab:blue', 'linestyle': '', 'fillstyle': 'full'},
    'ours_CCA': {'marker': 'd', 'markersize': 8, 'color': 'tab:red', 'linestyle': '', 'fillstyle': 'full'},
    'ours_CCD': {'marker': 'p', 'markersize': 8, 'color': 'tab:orange', 'linestyle': '', 'fillstyle': 'full'},
    'ours_CAA': {'marker': '^', 'markersize': 8, 'color': 'tab:green', 'linestyle': '', 'fillstyle': 'full'},
    'ours_CAD': {'marker': 'v', 'markersize': 8, 'color': 'tab:purple', 'linestyle': '', 'fillstyle': 'full'}, 
    'dense_baseline': {'marker': 'X', 'markersize': 8, 'color': 'tab:cyan', 'linestyle': '', 'fillstyle': 'full'},
    'dense_subsampled': {'marker': 'x', 'color': 'k', 'linestyle': 'dashed', 'fillstyle': 'none', 'alpha': 0.5},
}
LABELS = {  # Also defines the order of the legend
    'ours_CCC': 'Ours CCC',
    'ours_CCA': 'Ours CCA',
    'ours_CCD': 'Ours CCD',
    'ours_CAA': 'Ours CAA',
    'ours_CAD': 'Ours CAD',
    'dense_baseline': 'Dense Baseline (10k)',
    'dense_subsampled': 'Dense Baseline (subsampled)',
}


def setup_plot(figsize=(6, 4), fontsize=12):
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = fontsize

    # If font not found, install it with:
    # sudo apt install msttcorefonts -qq
    # rm ~/.cache/matplotlib -rf    
    plt.rcParams["font.family"] = "Times New Roman"
    return plt.figure(figsize=figsize)


def sort_handles_by_label(handles):
    # Sort legend handles by label
    correct_order = list(LABELS.values())
    labels = [h.get_label() for h in handles]
    inds = [labels.index(label) for label in correct_order if label in labels]
    return [handles[i] for i in inds]


def plot_auc_vs_runtime(errors, runtimes, auc_threshold=5, error_type='max'):
    fig = setup_plot()

    dense_runtimes = []
    dense_aucs = []

    handles = []
    for method_name in errors:

        if len(errors[method_name][error_type].shape) == 1:
            auc = metrics.compute_auc(errors[method_name][error_type], [auc_threshold])[0]
        elif len(errors[method_name][error_type].shape) == 2:
            aucs, stdev = metrics.compute_average_auc(errors[method_name][error_type][()], [auc_threshold])
            auc = aucs[0]
        runtime = np.median(runtimes[method_name]) * 1000.0

        # if method_name in DENSE_BASELINES:
        #     dense_runtimes.append(runtime)
        #     dense_aucs.append(auc)
        # else:
        h = plt.plot(runtime, auc, label=LABELS[method_name], **PLOT_STYLES[method_name])
        handles.extend(h)

    # Plot dense baselines
    # if len(dense_runtimes) > 0:
    #     dense_runtimes, dense_aucs = zip(*sorted(zip(dense_runtimes, dense_aucs), key=lambda x: x[0]))
    #     h = plt.plot(dense_runtimes, dense_aucs, label=LABELS['dense'], **PLOT_STYLES['dense'])
    #     handles.extend(h)
    #     h = plt.plot(dense_runtimes[-1], dense_aucs[-1], label=LABELS['dense_10k'], **PLOT_STYLES['dense_10k'])
    #     handles.extend(h)

    plt.xlabel('Runtime (ms)')
    plt.ylabel(f'AUC@{auc_threshold}°')
    # plt.title(f'{DATASET} {KEYPOINTS}')
    plt.xscale('log')
    plt.legend(handles=sort_handles_by_label(handles))
    plt.show()


def print_auc_table(errors, runtimes, auc_thresholds=metrics.DEFAULT_AUC_THRESHOLDS, baseline=None, latex=False, decimals=2):
    print('Method                                 AUC@{}            AUC@{}           AUC@{}        runtime (med / avg / speedup)'.format(*auc_thresholds))

    def decode_method_name(m):
        mapping = {'D': 'Dense', 'C': 'Center', 'A': 'Summarized'}
        return [mapping[ch] for ch in m.split('_')[1]]

    if baseline is not None and baseline in runtimes:
        # If we have a baseline, we will calculate speedup
        baseline_runtimes = runtimes[baseline]
        if isinstance(baseline_runtimes, np.ndarray) and len(baseline_runtimes.shape) == 2:
            # Average if we are dealing with multiple trials
            baseline_runtimes = np.mean(baseline_runtimes, axis=0)
        baseline_rt_med = np.median(baseline_runtimes) * 1000.0
        sort_by_speedup = True
        lines = {}
    else:
        baseline_rt_med = 0.0
        sort_by_speedup = False

    for method_name, method_errors in errors.items():
        errs = method_errors['max']

        # Compute AUCs
        if isinstance(errs, list) or len(errs.shape) == 1:
            aucs = metrics.compute_auc(errs, auc_thresholds)
            stddevs_auc = None
        elif len(errs.shape) == 2:
            # Multiple trials -- assume each row is a trial
            all_aucs = [metrics.compute_auc(errs_row, auc_thresholds) for errs_row in errs]
            aucs, stddevs_auc = metrics.compute_average_auc(errs, auc_thresholds)
        else:
            raise ValueError("Invalid shape of errors array")

        aucs = [np.round(100.0 * auc, decimals=decimals) for auc in aucs]

        if stddevs_auc is not None:
            auc_stddevs = [np.round(100.0 * s, decimals=2) for s in stddevs_auc]
            if latex:
                auc_msgs = ['\\tabnum{' + f'{auc:3.{decimals}f}' + '}{' + f'{auc_std:2.{decimals}f}' + '}' for auc, auc_std in zip(aucs, auc_stddevs)]
            else:
                auc_msgs = [f'{auc:8.{decimals}f}±{auc_std:2.{decimals}f}' for auc, auc_std in zip(aucs, auc_stddevs)]
        else:
            auc_msgs = [f'{auc:7.1f}      ' for auc in aucs]

        # Runtime stuff
        method_runtimes = runtimes[method_name]
        method_runtimes_stdev = None
        if isinstance(method_runtimes, np.ndarray) and len(method_runtimes.shape) == 2:
            # Average if we are dealing with multiple trials
            method_runtimes = np.mean(method_runtimes, axis=0)
            method_runtimes_stdev = np.std(method_runtimes, axis=0)

        rt_med = np.median(method_runtimes) * 1000.0
        rt_avg = np.mean(method_runtimes) * 1000.0
        speedup = baseline_rt_med / rt_med

        # Rounding (after calculating speedup)
        rt_med_rounded = np.round(rt_med, decimals=1)
        rt_avg_rounded = np.round(rt_avg, decimals=1)
        speedup_rounded = np.round(speedup, decimals=1)

        if method_runtimes_stdev is None:
            rt_msgs = [f'{rt_med_rounded:.1f}', f'{rt_avg_rounded:.1f}', f'{speedup_rounded:.1f}x']
        else:
            rt_stdev = np.round(np.mean(method_runtimes_stdev) * 1000.0)
            rt_msgs = [f'{rt_med:.1f}', '{rt_avg:.1f}±{rt_stdev:.1f}', '{speedup:.1f}x']            

        if latex:
            method_description = decode_method_name(method_name)
            line = ' && ' + ' & '.join(method_description) + f' && {auc_msgs[0]} & {auc_msgs[1]} & {auc_msgs[2]} && {rt_med:.1f} & {rt_avg:.1f} & {speedup:.1f}x \\\\'
        else:
            line = f'{method_name:35s} {auc_msgs[0]}    {auc_msgs[1]}    {auc_msgs[2]}    {rt_med:.1f} / {rt_avg:.1f} ms / {speedup:.1f}x'

        if sort_by_speedup:
            lines.setdefault(speedup, []).append(line)
        else:
            print(line)

    if sort_by_speedup:
        for speedup in sorted(lines.keys()):
            for l in lines[speedup]:
                print(l)
