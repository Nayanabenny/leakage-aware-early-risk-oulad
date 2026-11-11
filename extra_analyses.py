"""Additional analyses requested in review of the draft paper.

Reuses the data loading, feature construction and model definitions from Research.ipynb
(cells 1, 3, 5, 7, 9, 13) so that every number is produced by the same pipeline.

  A. Decomposition of the retrospective gap (RQ1): features {week 9, full course}
     x population {still enrolled at week 9, everyone} x split {temporal, random}.
  B. Assessment scores as an additional feature set.
  C. Second temporal test fold (train 2013B + 2013J, test 2014B).
  D. Students appearing in both training and test presentations.
  E. Per-module performance at week 9.
  F. Calibration (Brier score, reliability curve) at week 9.
  G. Subgroup performance (fairness audit) at week 9.

Run from this folder:  python extra_analyses.py
"""
import json
import time

nb = json.load(open('Research.ipynb', encoding='utf-8'))
for _i in (1, 3, 5, 7, 9, 13):
    exec(''.join(nb['cells'][_i]['source']), globals())

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

t0 = time.time()
W = PRIMARY_WEEK
Y = labels()
SPLIT_TEST = (enrol.split == 'test').to_numpy()
SPLIT_TRAIN = ~SPLIT_TEST


def log(msg):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


def fit_eval(X, y, tr, te, model_name):
    """Same tuning, threshold and metrics as run() in the notebook, for arbitrary train/test masks."""
    groups = enrol.id_student.to_numpy()[tr]
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    est, dist = model_zoo()[model_name]
    search = RandomizedSearchCV(est, dist, n_iter=N_ITER, scoring='roc_auc', n_jobs=-1,
                                random_state=SEED, cv=cv, refit=True)
    search.fit(X[tr], y[tr], groups=groups)
    oof = cross_val_predict(clone(search.best_estimator_), X[tr], y[tr], groups=groups, cv=cv,
                            method='predict_proba', n_jobs=-1)[:, 1]
    thr = f1_threshold(y[tr], oof)
    p = search.predict_proba(X[te])[:, 1]
    yt = y[te]
    pred = (p >= thr).astype(int)
    row = {'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'at_risk_test': float(yt.mean()),
           'cv_auc': search.best_score_, 'test_auc': roc_auc_score(yt, p),
           'pr_auc': average_precision_score(yt, p), 'precision': precision_score(yt, pred),
           'recall': recall_score(yt, pred), 'f1': f1_score(yt, pred), 'brier': brier_score_loss(yt, p),
           'threshold': thr}
    return row, {'y': yt, 'p': p, 'pred': pred, 'te': te, 'model': search.best_estimator_}


def boot_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng_ = np.random.default_rng(seed)
    idx = rng_.integers(0, len(y), size=(n_boot, len(y)))
    a = np.array([roc_auc_score(y[i], p[i]) for i in idx])
    return np.percentile(a, [2.5, 97.5])


def feats(week, extra=None):
    X = build_features(week).join(demo)
    if extra is not None:
        X = X.join(extra)
    return X


# Random enrolment-level split (the common practice): 70/30, stratified by outcome, all presentations pooled
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
_tr_idx, _te_idx = next(sss.split(np.zeros(len(enrol)), Y))
RAND_TEST = np.zeros(len(enrol), bool)
RAND_TEST[_te_idx] = True
RAND_TRAIN = ~RAND_TEST

# ---------------------------------------------------------------- A. decomposition of the retrospective gap
log('A. decomposition')
rows = []
store_A = {}
for split_name, trm, tem in [('temporal', SPLIT_TRAIN, SPLIT_TEST), ('random', RAND_TRAIN, RAND_TEST)]:
    for feat_name, fw in [('week 9', W), ('full course', FULL_WEEKS)]:
        for pop_name, retro in [('enrolled at week 9', False), ('all enrolments', True)]:
            X = feats(fw)[BEHAV]
            pop = population_mask(W, retrospective=retro)
            row, out = fit_eval(X, Y, pop & trm, pop & tem, 'XGBoost')
            lo, hi = boot_ci(out['y'], out['p'])
            row.update({'split': split_name, 'features': feat_name, 'population': pop_name,
                        'auc_ci_low': lo, 'auc_ci_high': hi})
            rows.append(row)
            store_A[(split_name, feat_name, pop_name)] = out
            log(f'  {split_name:8s} {feat_name:11s} {pop_name:18s} AUC {row["test_auc"]:.3f}')
decomp = pd.DataFrame(rows)
decomp.to_csv(os.path.join(RES_DIR, 'rq1_decomposition.csv'), index=False)

MAIN = store_A[('temporal', 'week 9', 'enrolled at week 9')]

# ---------------------------------------------------------------- B. assessment scores
log('B. assessment scores')


def score_features(week):
    d = 7 * week - 1
    seen = subs[(subs.date_submitted <= d) & (subs.is_banked == 0)].copy()
    seen['score'] = pd.to_numeric(seen['score'], errors='coerce')
    g = seen.groupby('eid')['score']
    return pd.DataFrame({'avg_score': g.mean(), 'min_score': g.min()}).reindex(enrol['eid'])


SCORES = ['avg_score', 'min_score']
FEATURE_SETS['Scores'] = SCORES
FEATURE_SETS['Behavioral + Scores'] = BEHAV + SCORES
rows = []
for w in CUTOFF_WEEKS:
    X = feats(w, score_features(w))
    pop = population_mask(w)
    for fs in ['Scores', 'Behavioral + Scores']:
        row, out = fit_eval(X[FEATURE_SETS[fs]], Y, pop & SPLIT_TRAIN, pop & SPLIT_TEST, 'XGBoost')
        lo, hi = boot_ci(out['y'], out['p'])
        row.update({'cutoff_week': w, 'feature_set': fs, 'auc_ci_low': lo, 'auc_ci_high': hi})
        rows.append(row)
    log(f'  week {w}: scores {rows[-2]["test_auc"]:.3f}  beh+scores {rows[-1]["test_auc"]:.3f}')
pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, 'scores_by_cutoff.csv'), index=False)

# ---------------------------------------------------------------- C. second temporal fold
log('C. second temporal fold (train 2013B+2013J, test 2014B)')
TR2 = enrol.code_presentation.isin(['2013B', '2013J']).to_numpy()
TE2 = (enrol.code_presentation == '2014B').to_numpy()
rows = []
for w in CUTOFF_WEEKS:
    X = feats(w)
    pop = population_mask(w)
    for fs, mn in [('Behavioral', 'XGBoost'), ('Behavioral', 'Logistic Regression'), ('Demographic', 'XGBoost')]:
        row, out = fit_eval(X[FEATURE_SETS[fs]], Y, pop & TR2, pop & TE2, mn)
        lo, hi = boot_ci(out['y'], out['p'])
        row.update({'cutoff_week': w, 'feature_set': fs, 'model': mn, 'auc_ci_low': lo, 'auc_ci_high': hi})
        rows.append(row)
    log(f'  week {w}: ' + ', '.join(f'{r["feature_set"][:4]}/{r["model"][:3]} {r["test_auc"]:.3f}' for r in rows[-3:]))
fold2 = pd.DataFrame(rows)
fold2['modules_in_test'] = enrol.loc[TE2, 'code_module'].nunique()
fold2.to_csv(os.path.join(RES_DIR, 'second_test_fold_2014B.csv'), index=False)

# ---------------------------------------------------------------- D. student overlap train/test
log('D. student overlap')
train_students = set(enrol.loc[SPLIT_TRAIN, 'id_student'])
te_idx = np.where(MAIN['te'])[0]
overlap = enrol.id_student.to_numpy()[te_idx]
overlap = np.array([s in train_students for s in overlap])
y_, p_ = MAIN['y'], MAIN['p']
ov = {'test_n': len(y_), 'overlap_n': int(overlap.sum()),
      'overlap_unique_students': int(enrol.iloc[te_idx[overlap]].id_student.nunique()),
      'auc_all': roc_auc_score(y_, p_), 'auc_new_students': roc_auc_score(y_[~overlap], p_[~overlap]),
      'auc_overlap_students': roc_auc_score(y_[overlap], p_[overlap])}
ov['auc_new_ci_low'], ov['auc_new_ci_high'] = boot_ci(y_[~overlap], p_[~overlap])
pd.DataFrame([ov]).to_csv(os.path.join(RES_DIR, 'student_overlap.csv'), index=False)
log(f'  {ov}')

# ---------------------------------------------------------------- E. per-module performance
log('E. per-module')
te_enrol = enrol.iloc[te_idx].reset_index(drop=True)
rows = []
for mod, g in te_enrol.groupby('code_module'):
    i = g.index.to_numpy()
    lo, hi = boot_ci(y_[i], p_[i])
    rows.append({'module': mod, 'n': len(i), 'at_risk': y_[i].mean(), 'auc': roc_auc_score(y_[i], p_[i]),
                 'auc_ci_low': lo, 'auc_ci_high': hi,
                 'recall': recall_score(y_[i], MAIN['pred'][i]), 'precision': precision_score(y_[i], MAIN['pred'][i])})
permod = pd.DataFrame(rows)
permod.to_csv(os.path.join(RES_DIR, 'per_module_week9.csv'), index=False)
print(permod.round(3).to_string(index=False))

# ---------------------------------------------------------------- F. calibration
log('F. calibration (week 9, behavioral features, all four models)')
X = feats(W)[BEHAV]
pop = population_mask(W)
rows, cal_store = [], {}
base = Y[pop & SPLIT_TRAIN].mean()
for mn in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    out = MAIN if mn == 'XGBoost' else fit_eval(X, Y, pop & SPLIT_TRAIN, pop & SPLIT_TEST, mn)[1]
    cal_store[mn] = out
    rows.append({'model': mn, 'test_auc': roc_auc_score(out['y'], out['p']),
                 'brier': brier_score_loss(out['y'], out['p']),
                 'mean_predicted': out['p'].mean(), 'observed_rate': out['y'].mean()})
rows.append({'model': 'Base rate (training)', 'brier': brier_score_loss(MAIN['y'], np.full(len(MAIN['y']), base)),
             'mean_predicted': base, 'observed_rate': MAIN['y'].mean()})
calib = pd.DataFrame(rows)
calib.to_csv(os.path.join(RES_DIR, 'calibration_week9.csv'), index=False)
print(calib.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(3.5, 3.1))
ax.plot([0, 1], [0, 1], color='#898781', lw=0.8, ls='--', label='Perfect calibration')
for mn, col in [('Logistic Regression', '#e8743b'), ('XGBoost', '#3987e5')]:
    fp, mp = calibration_curve(cal_store[mn]['y'], cal_store[mn]['p'], n_bins=10, strategy='quantile')
    ax.plot(mp, fp, marker='o', ms=3, lw=1.4, color=col, label=mn)
ax.set_xlabel('Mean predicted risk'); ax.set_ylabel('Observed share at risk')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_title(f'Calibration at week {W} (2014J)', loc='left', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_calibration_week9.png'), dpi=300)

# ---------------------------------------------------------------- G. subgroup performance
log('G. subgroups')
pred = MAIN['pred']
rows = []
for attr in ['gender', 'age_band', 'imd_band', 'disability', 'highest_education']:
    for val, g in te_enrol.groupby(te_enrol[attr].fillna('Unknown')):
        i = g.index.to_numpy()
        yy, pp, pr = y_[i], p_[i], pred[i]
        if len(i) < 100 or yy.min() == yy.max():
            continue
        rows.append({'attribute': attr, 'group': val, 'n': len(i), 'at_risk': yy.mean(),
                     'auc': roc_auc_score(yy, pp),
                     'tpr': pr[yy == 1].mean(), 'fpr': pr[yy == 0].mean(), 'flag_rate': pr.mean()})
sub = pd.DataFrame(rows)
sub.to_csv(os.path.join(RES_DIR, 'subgroups_week9.csv'), index=False)
print(sub.round(3).to_string(index=False))
log('done')
