# Leakage-Aware Early Prediction of Student Academic Risk from Micro-Temporal VLE Behavior

Code accompanying the paper *"Leakage-Aware Early Prediction of Student Academic Risk from Micro-Temporal VLE
Behavior with Explainable Machine Learning"*. Every table and figure in the paper can be regenerated with the code here.

The study uses the Open University Learning Analytics Dataset (OULAD). It predicts whether an enrolment ends in
Fail/Withdrawn, using only data observed up to a cutoff week (2, 4, 6, 9, 12, 16, 20). Models are trained on past
presentations (2013B, 2013J, 2014B) and tested on a later, unseen presentation (2014J).

## Repository contents

| Path | Description |
|---|---|
| `Research.ipynb` | Main pipeline: data loading, enrolment-level target, cutoff-based behavioral features, model tuning (Logistic Regression, Random Forest, XGBoost, LightGBM), temporal test evaluation with bootstrap CIs, feature-set ablation, ROC and AUC-vs-cutoff figures, SHAP explainability, and the Fail-vs-Pass sensitivity analysis. |
| `extra_analyses.py` | Additional analyses that reuse the notebook's pipeline: (A) decomposition of the retrospective gap (RQ1), (B) assessment scores as features, (C) a second temporal test fold (test 2014B), (D) train/test student overlap, (E) per-module performance, (F) calibration, (G) subgroup / fairness audit. |
| `results/` | CSV outputs reported in the paper's tables. |
| `figures/` | Figures used in the paper. |
| `Dataset/` | Place the OULAD CSV files here (not included, see below). |

### Outputs

| Produced by | Files |
|---|---|
| `Research.ipynb` | `results/population_by_cutoff.csv`, `results/model_comparison_by_cutoff.csv`, `results/table_week9.csv`, `results/shap_share_by_cutoff.csv`, `results/sensitivity_fail_vs_pass.csv`, `figures/fig_auc_vs_cutoff.png`, `figures/fig_roc_week9.png`, `figures/fig_shap_bar_week9.png`, `figures/fig_shap_beeswarm_week9.png`, `figures/fig_shap_share_by_cutoff.png` |
| `extra_analyses.py` | `results/rq1_decomposition.csv`, `results/scores_by_cutoff.csv`, `results/second_test_fold_2014B.csv`, `results/student_overlap.csv`, `results/per_module_week9.csv`, `results/calibration_week9.csv`, `results/subgroups_week9.csv`, `figures/fig_calibration_week9.png` |

## Data

OULAD is not redistributed here (the click-stream file `studentVle.csv` alone is about 450 MB). Download it from
<https://analyse.kmi.open.ac.uk/open-dataset> (licensed CC BY 4.0) and put these files directly in `Dataset/`:

```
Dataset/
  assessments.csv
  courses.csv
  studentAssessment.csv
  studentInfo.csv
  studentRegistration.csv
  studentVle.csv
  vle.csv
```

Reference: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
*Scientific Data*, 4, 170171. <https://doi.org/10.1038/sdata.2017.171>

## Setup

Python 3.12 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run from the repository root.

1. Run all cells of the main notebook:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace Research.ipynb
   ```

2. Run the additional analyses (this script executes the notebook's setup cells itself, so it uses the same data,
   features and models):

   ```bash
   python extra_analyses.py
   ```

Results go to `results/` and figures to `figures/`. The random seed is fixed (`SEED = 42`). Hyperparameter search runs
in parallel, so the full run takes a while on a laptop, and tiny numeric differences are possible across library
versions or hardware.

## Main settings

| Setting | Value |
|---|---|
| Unit of analysis | Enrolment (`id_student`, `code_module`, `code_presentation`) |
| Target | 1 = Fail or Withdrawn, 0 = Pass or Distinction |
| Cutoff weeks | 2, 4, 6, 9, 12, 16, 20 (primary: week 9, day 62) |
| Population at a cutoff | Enrolments still registered at the end of the cutoff week |
| Train / test | 2013B, 2013J, 2014B / 2014J |
| Tuning | `RandomizedSearchCV`, 15 iterations, student-grouped stratified 5-fold CV, ROC-AUC |
| Decision threshold | Max F1 on out-of-fold training predictions |
| Uncertainty | 1,000 bootstrap resamples of the test set |

## License

The code is released under the MIT License (see `LICENSE`). The OULAD data is not part of this repository and remains
under its own CC BY 4.0 license.
