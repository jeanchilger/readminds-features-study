# Readminds Experiments
Experiments and their different setups are described here. Every experiment is related to a training strategy for the machine learning model. The features extraction pipeline is executed once, and the generated values remains the same across all setups.

## Running Experiments

The entry point for running experiments is the main script under `experiments/`. For running it use:

```shell
./experiments/run <experiment> [options]
```

Availables choices for `<experiment>` are:

| Experiment Name | Description |
|-----------------|-------------|
| `original`      | Same training setup as in thesis |
| `subject`       | Model trained on group, later specialized on each subject |
| `feature-group` | Same setup as in `subject-model`, but with variations on features used |

And options are as follows:

```shell
-s <samples>, --samples=<samples>
      Specifies the number of samples for <experiment>
      that should be executed. At the end of all
      executions, mean and std of them will also be taken.
      Defaults to 1.

-v, --verbose
      Activates logging messages from training phase.

-h, --help
      Shows descriptions for options.
```

By default, the results will be stored under `results/<experiment>/`. Files named `<experiment>-N.csv` contains the individual results, taken on each execution - which total number is defined by `<samples>`. The file `summary.csv` contains the means and standard deviations, of all metrics used, obtained for each subject across all individual runs.