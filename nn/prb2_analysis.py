"""PRB2 Step 2 analysis (executor's analysis; report is DRAFT (UNVERIFIED)).

Reads nn/results_prb2.jsonl + nn/training_prb2.jsonl only. Never touches
MD1 records except for the cited greedy baselines (frozen in the design
note §8, recomputed here only as a cross-check). Writes
nn/analysis_prb2_out.txt and nn/analysis_prb2_out.json.
"""
import json, collections, random

HERE = "/home/hatch/workspace/time-mind-model/nn"
ARMS = ['wrapped', 'plain', 'memory', 'oracle_ind', 'oracle_new', 'phase_only_new']
TRAINED_W = ['R0', 'R1', 'R3', 'R4', 'R5', 'R6', 'R7', 'R9']
HELD_W = ['R2', 'R8']
ALL_W = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
ACTION_NAMES = ["up", "down", "left", "right", "stay"]
GREEDY = {  # design note §8 (seed-0, from MD1 records)
    'trained_pooled': (61, 960), 'held_pooled': (2, 240),
    'per_agent': {'wrapped': (0, 0), 'plain': (14, 0), 'memory': (20, 0),
                  'oracle_ind': (0, 0), 'oracle_new': (27, 2), 'phase_only_new': (0, 0)},
    'per_world': {'R0': 21, 'R1': 0, 'R3': 4, 'R4': 12, 'R5': 3, 'R6': 13,
                  'R7': 8, 'R9': 0, 'R2': 2, 'R8': 0},
    'control': 20,
}

def main():
    ev = [json.loads(l) for l in open(f"{HERE}/results_prb2.jsonl")]
    tr = [json.loads(l) for l in open(f"{HERE}/training_prb2.jsonl")]
    out = {}
    out['n_eval'] = len(ev)
    out['n_train'] = len(tr)
    out['dup_keys'] = len(ev) - len({(r['arm'], r['seed'], r['world'], r['rep'], r['start_phase']) for r in ev})
    trained = [r for r in ev if r['world'] in TRAINED_W]
    held = [r for r in ev if r['world'] in HELD_W]
    out['pooled_trained'] = [sum(r['reached'] for r in trained), len(trained)]
    out['pooled_held'] = [sum(r['reached'] for r in held), len(held)]
    out['pooled_trained_pct'] = sum(r['reached'] for r in trained) / len(trained) * 100
    out['pooled_held_pct'] = sum(r['reached'] for r in held) / len(held) * 100
    out['per_agent'] = {}
    for a in ARMS:
        t = [r for r in ev if r['arm'] == a and r['world'] in TRAINED_W]
        h = [r for r in ev if r['arm'] == a and r['world'] in HELD_W]
        out['per_agent'][a] = {'trained': [sum(r['reached'] for r in t), len(t)],
                                'held': [sum(r['reached'] for r in h), len(h)]}
    out['per_world'] = {}
    for w in ALL_W:
        ws = [r for r in ev if r['world'] == w]
        out['per_world'][w] = [sum(r['reached'] for r in ws), len(ws)]
    out['per_rep_trained'] = []
    out['per_rep_held'] = []
    for rep in range(5):
        t = [r for r in ev if r['rep'] == rep and r['world'] in TRAINED_W]
        h = [r for r in ev if r['rep'] == rep and r['world'] in HELD_W]
        out['per_rep_trained'].append([sum(r['reached'] for r in t), len(t)])
        out['per_rep_held'].append([sum(r['reached'] for r in h), len(h)])
    ctrl = [r for r in ev if r['arm'] == 'memory' and r['seed'] == 0 and r['world'] == 'R0']
    out['control'] = [sum(r['reached'] for r in ctrl), len(ctrl)]
    out['control_per_rep'] = []
    for rep in range(5):
        c = [r for r in ctrl if r['rep'] == rep]
        out['control_per_rep'].append([sum(r['reached'] for r in c), len(c)])
    # epsilon-action rates
    tot_steps = tot_eps = 0
    eps_reached = eps_unreached = steps_reached = steps_unreached = 0
    eps_per_agent = {}
    for r in ev:
        es = sum(1 for s in r['steps'] if s['epsilon_action'])
        tot_steps += len(r['steps']); tot_eps += es
        if r['reached']:
            eps_reached += es; steps_reached += len(r['steps'])
        else:
            eps_unreached += es; steps_unreached += len(r['steps'])
        a = eps_per_agent.setdefault(r['arm'], [0, 0])
        a[0] += es; a[1] += len(r['steps'])
    out['epsilon'] = {'total_steps': tot_steps, 'epsilon_steps': tot_eps,
                      'rate_pct': tot_eps / tot_steps * 100,
                      'reached_rate_pct': eps_reached / steps_reached * 100 if steps_reached else None,
                      'unreached_rate_pct': eps_unreached / steps_unreached * 100 if steps_unreached else None,
                      'per_agent_pct': {a: v[0] / v[1] * 100 for a, v in eps_per_agent.items()}}
    # RNG stream replay: rebuild each rep stream and check epsilon flags + random actions
    mism = 0; checked_steps = 0; checked_eps = 0
    by_rep = collections.defaultdict(list)
    for r in ev:
        by_rep[(r['arm'], r['seed'], r['world'], r['rep'])].append(r)
    for (arm, seed, world, rep), eps in by_rep.items():
        eps.sort(key=lambda r: r['start_phase'])
        rng = random.Random(eps[0]['rep_rng_seed'])
        for r in eps:
            for s in r['steps']:
                checked_steps += 1
                u = rng.random()
                if u < 0.05:
                    k = rng.randrange(5)
                    checked_eps += 1
                    if not s['epsilon_action'] or s['action'] != ACTION_NAMES[k]:
                        mism += 1
                else:
                    if s['epsilon_action']:
                        mism += 1
    out['rng_replay'] = {'steps_checked': checked_steps, 'epsilon_steps_checked': checked_eps, 'mismatches': mism}
    # wrapped blank audit (f103 known flag)
    def blank_class(r):
        flags = [s.get('f103') for s in r['steps']]
        if not flags:
            return 'none'
        if all(f == 0 for f in flags):
            return 'whole'
        if any(f == 0 for f in flags):
            seen_one = False; scattered = False
            for f in flags:
                if f == 1:
                    seen_one = True
                elif f == 0 and seen_one:
                    scattered = True
            return 'scattered' if scattered else 'prefix'
        return 'none'
    wb = [r for r in ev if r['arm'] == 'wrapped']
    bl = [r for r in wb if blank_class(r) in ('prefix', 'scattered', 'whole')]
    out['blank'] = {'wrapped_episodes': len(wb), 'blank_episodes': len(bl),
                    'classes': dict(collections.Counter(blank_class(r) for r in bl)),
                    'blank_reached': sum(r['reached'] for r in bl)}
    out['blank_prefix_lengths'] = collections.Counter(
        sum(1 for s in r['steps'] if s['f103'] == 0) for r in bl if blank_class(r) == 'prefix')
    # staging / far-side description
    reach = [r for r in ev if r['reached']]
    out['reached_staged_at_4row'] = sum(1 for r in reach if r['arrival']['cell'] == [4, r['gap_row']])
    out['reached_total'] = len(reach)
    out['entered_not_reached'] = {}
    out['modal_final_unreached'] = {}
    out['stall_kinds_unreached'] = {}
    for w in ALL_W:
        ws = [r for r in ev if r['world'] == w]
        ent = [r for r in ws if r['first_entry_index'] is not None]
        out['entered_not_reached'][w] = [len(ent), sum(1 for r in ent if not r['reached'])]
        un = [r for r in ws if not r['reached']]
        finals = collections.Counter((r['steps'][-1]['x'], r['steps'][-1]['y']) for r in un)
        out['modal_final_unreached'][w] = finals.most_common(1)[0] if finals else None
        out['stall_kinds_unreached'][w] = dict(collections.Counter(r['stall_kind'] for r in un))
    out['crushes_noisy'] = sum(r['crushes'] for r in ev)
    # rule computation (COMPUTED, NOT RULED — verdict section stays empty)
    complete = (len(ev) == 6000 and len(tr) == 6 and out['dup_keys'] == 0
                and all(v == 100 for v in collections.Counter((r['arm'], r['seed'], r['world']) for r in ev).values()))
    ctrl_rate = out['control'][0] / out['control'][1] * 100 if out['control'][1] else 0
    g_rate = GREEDY['trained_pooled'][0] / GREEDY['trained_pooled'][1] * 100
    n_rate = out['pooled_trained_pct']
    if not complete:
        computed = 'INCOMPLETE'
    elif ctrl_rate < 75:
        computed = 'UNINTERPRETABLE'
    elif n_rate >= 2 * g_rate and n_rate >= 20:
        computed = 'LIFTED'
    elif n_rate < 1.25 * g_rate:
        computed = 'NO LIFT'
    else:
        computed = 'NEITHER'
    out['computed_outcome'] = computed
    out['held_out_lift_flag'] = out['pooled_held_pct'] >= 20
    out['greedy_baselines'] = GREEDY
    with open(f"{HERE}/analysis_prb2_out.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    lines = []
    lines.append(f"evals: {out['n_eval']}, trainings: {out['n_train']}, dups: {out['dup_keys']}")
    lines.append(f"pooled noisy trained: {out['pooled_trained'][0]}/{out['pooled_trained'][1]} = {out['pooled_trained_pct']:.2f}% (greedy 61/960 = 6.35%)")
    lines.append(f"pooled noisy held: {out['pooled_held'][0]}/{out['pooled_held'][1]} = {out['pooled_held_pct']:.2f}% (greedy 2/240 = 0.83%)")
    lines.append(f"per-agent trained/held: {out['per_agent']}")
    lines.append(f"per-world: {out['per_world']}")
    lines.append(f"per-rep trained: {out['per_rep_trained']}")
    lines.append(f"per-rep held: {out['per_rep_held']}")
    lines.append(f"control memory/0 R0: {out['control'][0]}/{out['control'][1]} per-rep {out['control_per_rep']}")
    lines.append(f"epsilon: {out['epsilon']}")
    lines.append(f"rng replay: {out['rng_replay']}")
    lines.append(f"blank: {out['blank']} prefix lengths: {dict(out['blank_prefix_lengths'])}")
    lines.append(f"reached staged at (4,row): {out['reached_staged_at_4row']}/{out['reached_total']}")
    lines.append(f"entered/not-reached: {out['entered_not_reached']}")
    lines.append(f"modal final unreached: {out['modal_final_unreached']}")
    lines.append(f"stall kinds unreached: {out['stall_kinds_unreached']}")
    lines.append(f"crushes noisy total: {out['crushes_noisy']}")
    lines.append(f"COMPUTED outcome (NOT RULED): {computed}; held-out lift flag: {out['held_out_lift_flag']}")
    with open(f"{HERE}/analysis_prb2_out.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
