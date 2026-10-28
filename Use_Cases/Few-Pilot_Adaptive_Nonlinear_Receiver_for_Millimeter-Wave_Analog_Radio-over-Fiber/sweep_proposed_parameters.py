#!/usr/bin/env python3
"""Add-on grid search for the existing Subspace + DD receiver.

Selection uses source physical-group cross-validation only. The source train
region fits teachers, source val supplies eight pilots and unlabeled DD rows,
and source test scores pseudo-target validation. No held-out target is loaded
until settings have been selected. All four modulations of a physical group
are excluded together from each validation PCA prior.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from fewshot_equalizer.compat import bits_for_symbol_indices, calculate_ber, qam_demod
from run_receiver import load_prior
from fewshot_equalizer.data import discover_scenarios
from fewshot_equalizer.engine import write_csv
from fewshot_equalizer.hammerstein import (
    calibrate_hammerstein_latent, decision_directed_adapt, fit_hammerstein_teacher,
    hammerstein_complexity, normalize_from_region, parallel_hammerstein_design,
    qam_decisions,
)
from fewshot_equalizer.teacher_calibration import pca_basis

HERE = Path(__file__).resolve().parent
BASE = HERE / 'model/source_prior.npz'
OUT = HERE / 'reproduced/parameter_sweep'
TAPS = (7, 9, 11, 13, 15)
RANKS = (7, 11, 15, 19, 23)
STEPS = (0, 4, 8, 12, 16, 24)


def load(cache, scenario):
    with np.load(cache / f"{scenario.key}.npz", allow_pickle=False) as a:
        return a['rx'].astype(np.complex64), a['tx'].astype(np.complex64), a['bits'].astype(np.uint8)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cp = load_prior(BASE)
    root = HERE.parents[1]
    cache = root / cp['cache_dir']
    records = {s.scenario_id: s for s in discover_scenarios(root)}
    sources = [records[i] for i in cp['source_ids_used']]
    groups = [(s.fiber, s.carrier, s.power) for s in sources]
    regions = {k: tuple(v) for k, v in cp['regions'].items()}
    arrays = [load(cache, s) for s in sources]
    all_rows = []
    for taps in TAPS:
        rows_path = OUT / f"validation_t{taps}.csv"
        if rows_path.exists():
            import csv
            with rows_path.open() as f:
                all_rows.extend(list(csv.DictReader(f)))
            print(f"reuse completed t={taps}", flush=True)
            continue
        teacher_path = OUT / f"teachers_t{taps}.npz"
        if teacher_path.exists():
            vectors = np.load(teacher_path)['vectors']
        elif taps == 11:
            vectors = np.asarray(cp['source_coefficients'])
            np.savez(teacher_path, vectors=vectors)
        else:
            vectors = []
            for n, ((rx, tx, _), s) in enumerate(zip(arrays, sources), 1):
                vectors.append(fit_hammerstein_teacher(rx, tx, regions['train'], taps,
                                                      (1, 3), 25000, 1e-6).coefficient_vector)
                if n % 8 == 0:
                    print(f"t={taps}: fitted {n}/32 teachers; elapsed {time.perf_counter()-started:.1f}s", flush=True)
            vectors = np.stack(vectors)
            np.savez(teacher_path, vectors=vectors)
        priors = {}
        for group in set(groups):
            mask = np.asarray([g != group for g in groups])
            priors[group] = pca_basis(vectors[mask], max(RANKS))[:2]
        rows = []
        half = taps // 2
        for n, (s, (rx, tx, bits), group) in enumerate(zip(sources, arrays, groups), 1):
            norm, _, _ = normalize_from_region(rx, regions['val'])
            lo, hi = regions['val'][0]+1024+half, regions['val'][1]-half
            candidates = np.unique(np.linspace(lo, hi-1, 512, dtype=np.int64))
            pd = parallel_hammerstein_design(norm, candidates, taps, (1, 3))
            ui = np.arange(lo, hi, dtype=np.int64)
            ud = parallel_hammerstein_design(norm, ui, taps, (1, 3))
            # Common test scoring rows across all window lengths.
            ti = np.arange(regions['test'][0]+max(TAPS)//2,
                           regions['test'][1]-max(TAPS)//2, dtype=np.int64)
            td = parallel_hammerstein_design(norm, ti, taps, (1, 3))
            truth = bits_for_symbol_indices(bits, ti, s.order)
            center, basis = priors[group]
            initial = [calibrate_hammerstein_latent(pd, tx, candidates, center,
                       basis[:rank], 8, 1e-3, 'uniform')[0] for rank in RANKS]
            weights = np.column_stack(initial)
            # Same DD objective; cache the constant Gram and solve ranks together.
            penalty = np.full(ud.shape[1], 2e-4)
            penalty[-1] *= .01
            gram = ud.conj().T @ ud / len(ud)
            gram.flat[::len(penalty)+1] += penalty
            factor = cho_factor(gram)
            uh = ud.conj().T / len(ud)
            for step in range(max(STEPS)+1):
                if step in STEPS:
                    predictions = td @ weights
                    for j, rank in enumerate(RANKS):
                        errors, ber = calculate_ber(truth, qam_demod(predictions[:, j], s.order))
                        rows.append({'scenario': s.key, 'physical_group': str(group),
                                     'taps': taps, 'complex_coefficients': 8*taps+1,
                                     'rank': rank, 'dd_updates': step, 'bit_errors': int(errors),
                                     'total_bits': int(truth.size), 'ber': float(ber)})
                if taps == 11 and n == 1 and step == 12:
                    reference = decision_directed_adapt(initial[RANKS.index(15)], ud, s.order,
                                 iterations=12, confidence_fraction=1, pilot_weight=0, ridge=2e-4)
                    discrepancy = float(np.max(np.abs(reference-weights[:, RANKS.index(15)])))
                    if discrepancy > 1e-7:
                        raise RuntimeError(f"Cached DD parity failed: {discrepancy}")
                    (OUT / 'dd_numerical_parity.json').write_text(json.dumps({'max_coefficient_difference': discrepancy}, indent=2)+'\n')
                if step < max(STEPS):
                    predictions = ud @ weights
                    pseudo = np.column_stack([qam_decisions(predictions[:, j], s.order)
                                              for j in range(len(RANKS))])
                    weights = cho_solve(factor, uh @ pseudo + penalty[:, None]*weights)
            if n % 4 == 0:
                print(f"t={taps}: validation {n}/32; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        write_csv(rows, rows_path)
        all_rows.extend(rows)
    totals = {}
    for row in all_rows:
        key = tuple(int(row[k]) for k in ('taps', 'rank', 'dd_updates'))
        errors, count = totals.get(key, (0, 0))
        totals[key] = (errors+int(row['bit_errors']), count+int(row['total_bits']))
    summary = []
    for (taps, rank, step), (errors, count) in totals.items():
        summary.append({'taps': taps, 'complex_coefficients': 8*taps+1, 'rank': rank,
                        'dd_updates': step, 'bit_errors': errors, 'total_bits': count,
                        'ber': errors/count, 'real_macs': hammerstein_complexity(2, taps, (1, 3))})
    summary.sort(key=lambda r: (r['ber'], r['taps'], r['rank'], r['dd_updates']))
    write_csv([{k:v for k,v in r.items() if k != 'real_macs'} for r in summary], OUT / 'validation_summary.csv')
    selected = summary[0]
    fixed89 = min((r for r in summary if r['taps'] == 11), key=lambda r:r['ber'])
    baseline = next(r for r in summary if (r['taps'],r['rank'],r['dd_updates']) == (11,15,12))
    result = {'selection': 'Source-only leave-one-physical-group-out; 8 groups, 32 captures, 28 teachers per fold',
              'target_test_used_for_selection': False,
              'grid': {'taps': TAPS, 'rank': RANKS, 'dd_updates': STEPS},
              'fixed': {'pilots':8, 'orders':[1,3], 'samples_per_symbol':2, 'teacher_labels':25000,
                        'latent_ridge':1e-3, 'dd_ridge':2e-4, 'pilot_weight':0, 'confidence_fraction':1},
              'selected': selected, 'best_with_89_coefficients':fixed89,
              'baseline':baseline, 'all_settings':summary,
              'seconds':time.perf_counter()-started,
              'notes':['Window-dependent pilot positions follow the existing evaluator protocol.',
                       'Source scoring uses common test-region rows with seven-symbol guards.',
                       'Source test regions act as parameter-selection validation; target test remains outside selection.',
                       'Only the best tested grid setting is established, not a global optimum.']}
    (OUT / 'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected','best_with_89_coefficients','baseline','seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
