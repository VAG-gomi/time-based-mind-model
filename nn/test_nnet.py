#!/usr/bin/env python3
"""A2b + A3 verification for nnet.py. Prints PASS/FAIL per check."""

import os
import platform
import sys
import time

import numpy as np

from nnet import MLP, mse_grad, masked_mse_grad, Adam

# Mixed absolute/relative bar for gradient agreement:
#   |a - n| <= atol + rtol * max(|a|, |n|)
# Justification: central differences carry an absolute roundoff floor of
# ~machine_eps*|loss|/(2*eps) (~6e-10 at eps=1e-7), so a pure relative bar
# misfires on tiny gradients (~1e-4). atol=1e-8 sits ~15x above that floor;
# rtol=1e-5 is far above truncation error O(eps^2)~1e-14 yet 4+ orders of
# magnitude below real-bug signals (the mse_grad bug scored 0.33).
ATOL = 1e-8
RTOL = 1e-5
GRAD_EPS = 1e-7


def grads_ok(a, b, atol=ATOL, rtol=RTOL):
    return bool(np.all(np.abs(a - b)
                       <= atol + rtol * np.maximum(np.abs(a), np.abs(b))))


def worst_margin(a, b, atol=ATOL, rtol=RTOL):
    """Max over params of |a-n| / (atol + rtol*max(|a|,|n|)); <=1 passes."""
    return float(np.max(np.abs(a - b)
                        / (atol + rtol * np.maximum(np.abs(a), np.abs(b)))))


def rel_err(a, b):
    return np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))


def make_case(sizes, seed, batch, masked, data_seed=999):
    """Build (net, loss_fn, analytic_fn) for one gradient-check case."""
    rng = np.random.RandomState(data_seed)
    X = rng.randn(batch, sizes[0])
    Y = rng.randn(batch, sizes[-1])
    actions = rng.randint(0, sizes[-1], size=batch)
    targets = rng.randn(batch)
    net = MLP(sizes, seed=seed)

    def loss_of(n):
        out = n.forward(X)
        if masked:
            return masked_mse_grad(out, actions, targets)[0]
        return mse_grad(out, Y)[0]

    def analytic(n):
        out = n.forward(X)
        if masked:
            _, dout = masked_mse_grad(out, actions, targets)
        else:
            _, dout = mse_grad(out, Y)
        n.backward(dout)
        return n.get_grads()

    return net, loss_of, analytic


def central_numeric(net, loss_of, eps=GRAD_EPS):
    p0 = net.get_params().copy()
    num = np.empty(p0.size)
    for i in range(p0.size):
        pp = p0.copy()
        pp[i] += eps
        net.set_params(pp)
        lp = loss_of(net)
        pp = p0.copy()
        pp[i] -= eps
        net.set_params(pp)
        lm = loss_of(net)
        num[i] = (lp - lm) / (2.0 * eps)
    net.set_params(p0)
    return num


def grad_suite(eps):
    npass = total = 0
    worst = 0.0
    worst_case = ""
    max_abs = 0.0
    fails = []
    for seed in range(20):
        for batch in (1, 5):
            for masked in (False, True):
                net, loss_of, analytic = make_case([3, 4, 2], seed,
                                                   batch, masked)
                ana = analytic(net)
                num = central_numeric(net, loss_of, eps=eps)
                total += 1
                m = worst_margin(ana, num)
                if m > worst:
                    worst = m
                    worst_case = ("seed=%d batch=%d masked=%s"
                                  % (seed, batch, masked))
                max_abs = max(max_abs, float(np.abs(ana - num).max()))
                if m <= 1.0:
                    npass += 1
                else:
                    fails.append((seed, batch, masked, m))
    return npass, total, worst, worst_case, max_abs, fails


def main():
    all_ok = True

    # ---- check 1: gradient check, mixed atol/rtol bar ----
    print("##### check 1: gradient check (mixed bar) #####")
    print("bar: |a-n| <= %.0e + %.0e*max(|a|,|n|); margin<=1 passes"
          % (ATOL, RTOL))
    results = {}
    for eps in (1e-7, 1e-5):
        npass, total, worst, worst_case, max_abs, fails = grad_suite(eps)
        results[eps] = (npass, total)
        print("eps=%.0e: %d/%d passed | worst margin %.3f (%s) | "
              "max abs err %.3e"
              % (eps, npass, total, worst, worst_case, max_abs))
        for f in fails:
            print("  FAILED: seed=%d batch=%d masked=%s margin=%.3f" % f)
    ok = all(n == t for n, t in results.values())
    print("GRADIENT CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 2: negative control (corrupted gradient must fail) ----
    print("##### check 2: negative control #####")
    net, loss_of, analytic = make_case([3, 4, 2], 0, 5, False)
    ana = analytic(net)
    num = central_numeric(net, loss_of)
    corrupt = ana.copy()
    corrupt[3] += 0.5  # deliberately wrong gradient
    m = worst_margin(corrupt, num)
    print("corrupted analytic grads: worst margin = %.3e (must be > 1)"
          % m)
    ok = m > 1.0
    print("NEGATIVE CONTROL: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 2b: backward returns the true dL/dX ----
    print("##### check 2b: backward returns dL/dX #####")
    net = MLP([3, 4, 2], seed=3)
    rng = np.random.RandomState(77)
    X = rng.randn(5, 3)
    Y = rng.randn(5, 2)
    out = net.forward(X)
    _, dout = mse_grad(out, Y)
    dX_ana = net.backward(dout)
    print("returned shape: %s (expected (5, 3))" % (dX_ana.shape,))
    eps = 1e-7
    dX_num = np.zeros_like(X)
    for b in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[b, j] += eps
            Xm = X.copy()
            Xm[b, j] -= eps
            lp = mse_grad(net.forward(Xp), Y)[0]
            lm = mse_grad(net.forward(Xm), Y)[0]
            dX_num[b, j] = (lp - lm) / (2.0 * eps)
    m = worst_margin(dX_ana, dX_num)
    print("dL/dX vs central differences: worst margin %.3e" % m)
    ok = dX_ana.shape == (5, 3) and m <= 1.0
    print("DLDX CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 2c: Adam optimizer ----
    print("##### check 2c: Adam optimizer #####")
    # (i) first-step math: with zero-initialized moments, after one step
    # m = 0.1*g, v = 0.001*g^2, so m_hat = g, v_hat = g^2 exactly, and
    # the update is lr * g / (|g| + eps).
    net = MLP([3, 4, 2], seed=5)
    rng = np.random.RandomState(9)
    X = rng.randn(5, 3)
    Y = rng.randn(5, 2)
    out = net.forward(X)
    _, dout = mse_grad(out, Y)
    net.backward(dout)
    p0 = net.get_params().copy()
    g = net.get_grads()
    opt = Adam(net, lr=0.01)
    opt.step()
    p1 = net.get_params()
    expected = p0 - 0.01 * g / (np.abs(g) + 1e-8)
    m = worst_margin(p1, expected)
    print("first Adam step vs hand computation: worst margin %.3e" % m)
    ok_math = m <= 1.0
    # (ii) Adam minimizes: loss must fall on the smoke-test problem.
    rng = np.random.RandomState(123)
    Xs = rng.randn(40, 4)
    Ys = (Xs[:, [0]] ** 2 + np.sin(Xs[:, [1]])
          - 0.5 * Xs[:, [2]] * Xs[:, [3]])
    net2 = MLP([4, 32, 1], seed=1)
    opt2 = Adam(net2, lr=0.01)
    out = net2.forward(Xs)
    l0, _ = mse_grad(out, Ys)
    for _ in range(500):
        out = net2.forward(Xs)
        _, dout = mse_grad(out, Ys)
        net2.backward(dout)
        opt2.step()
    out = net2.forward(Xs)
    l1, _ = mse_grad(out, Ys)
    print("Adam 500 steps: loss %.6f -> %.6f" % (l0, l1))
    ok_down = l1 < 0.1 * l0
    ok = ok_math and ok_down
    print("ADAM CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 3: determinism ----
    print("##### check 3: determinism #####")
    a = MLP([3, 4, 2], seed=7).get_params()
    b = MLP([3, 4, 2], seed=7).get_params()
    c = MLP([3, 4, 2], seed=8).get_params()
    same = np.array_equal(a, b)
    diff = not np.array_equal(a, c)
    print("same seed -> identical weights: %s" % same)
    print("different seeds -> different weights: %s" % diff)
    ok = same and diff
    print("DETERMINISM CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 4: empty world (zeros init) ----
    print("##### check 4: empty world (all-zero init) #####")
    net = MLP([3, 4, 2], seed=0, init="zeros")
    rng = np.random.RandomState(42)
    X = rng.randn(5, 3)
    Y = rng.randn(5, 2)
    out = net.forward(X)
    _, dout = mse_grad(out, Y)
    net.backward(dout)
    grad_zero = {}
    for k in net._keys():
        g = net.grads[k]
        grad_zero[k] = bool(np.all(g == 0.0))
        print("grad %-3s: exactly zero: %s (max abs %.3e)"
              % (k, grad_zero[k], np.abs(g).max()))
    only_b2_nonzero = (not grad_zero["b2"]
                       and all(grad_zero[k] for k in net._keys()
                               if k != "b2"))
    print("only the output biases have nonzero gradient: %s"
          % only_b2_nonzero)
    p0 = net.get_params().copy()
    net.sgd_step(0.1)
    off = 0
    changed = []
    p1 = net.get_params()
    for k in net._keys():
        n = net.params[k].size
        if not np.array_equal(p0[off:off + n], p1[off:off + n]):
            changed.append(k)
        off += n
    print("blocks changed by one SGD step: %s (expected ['b2'])" % changed)
    ok = only_b2_nonzero and changed == ["b2"]
    print("EMPTY WORLD CHECK: %s" % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 5: learning smoke test (tiny fixed regression) ----
    print("##### check 5: learning smoke test #####")
    rng = np.random.RandomState(123)
    Xs = rng.randn(40, 4)
    Ys = (Xs[:, [0]] ** 2 + np.sin(Xs[:, [1]])
          - 0.5 * Xs[:, [2]] * Xs[:, [3]])
    net = MLP([4, 32, 1], seed=1)
    lr, steps = 0.2, 2000
    out = net.forward(Xs)
    l0, _ = mse_grad(out, Ys)
    for _ in range(steps):
        out = net.forward(Xs)
        _, dout = mse_grad(out, Ys)
        net.backward(dout)
        net.sgd_step(lr)
    out = net.forward(Xs)
    l1, _ = mse_grad(out, Ys)
    print("net 4->32->1, %d full-batch SGD steps, lr=%.2f" % (steps, lr))
    print("loss: %.6f -> %.6f (ratio %.4f)" % (l0, l1, l1 / l0))
    ok = l1 < 0.03 * l0
    print("SMOKE TEST (loss falls to <3%% of start): %s"
          % ("PASS" if ok else "FAIL"))
    all_ok = all_ok and ok

    # ---- check 6: speed ----
    print("##### check 6: speed #####")
    print("python %s | numpy %s" % (sys.version.split()[0],
                                    np.__version__))
    u = platform.uname()
    print("machine: %s %s | cpus: %s | node: %s"
          % (u.system, u.machine, os.cpu_count(), u.node))
    print("this run: Linux VM (hatch) -- not the phone")
    rng = np.random.RandomState(0)
    for H in (32, 1000, 10000):
        net = MLP([105, H, 5], seed=0)
        mem_mib = net.n_params * 8 / 1024.0 ** 2
        for B in (1, 32):
            X = rng.randn(B, 105)
            Y = rng.randn(B, 5)
            for _ in range(20):  # warmup
                out = net.forward(X)
                _, dout = mse_grad(out, Y)
                net.backward(dout)
                net.sgd_step(0.01)
            N = 300
            t0 = time.perf_counter()
            for _ in range(N):
                out = net.forward(X)
                _, dout = mse_grad(out, Y)
                net.backward(dout)
                net.sgd_step(0.01)
            ms = (time.perf_counter() - t0) / N * 1000.0
            print("H=%5d batch=%2d: %7.3f ms/update | params=%7d | "
                  "weights %6.2f MiB" % (H, B, ms, net.n_params, mem_mib))

    # ---- check 7: A3, XOR ----
    print("##### check 7: A3, XOR #####")
    Xxor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Txor = np.array([[0], [1], [1], [0]])
    # lr=0.5 tuned: all 10 xavier seeds reach max abs err < 0.1 without
    # diverging (lr=1.0 diverged on some seeds). "Solved" = every output
    # within 0.1 of its target, not just across the 0.5 threshold.
    lr, max_steps = 0.5, 20000

    def train_xor(net, max_steps):
        for s in range(1, max_steps + 1):
            out = net.forward(Xxor)
            _, dout = mse_grad(out, Txor)
            net.backward(dout)
            net.sgd_step(lr)
            if bool(np.all(np.abs(out - Txor) < 0.1)):
                loss, _ = mse_grad(out, Txor)
                return s, float(np.abs(out - Txor).max()), float(loss)
        out = net.forward(Xxor)
        loss, _ = mse_grad(out, Txor)
        return None, float(np.abs(out - Txor).max()), float(loss)

    print("part 1: xavier init, 2->4->1, lr=%.1f, 10 seeds" % lr)
    solved = 0
    for seed in range(10):
        net = MLP([2, 4, 1], seed=seed)
        s, maxae, loss = train_xor(net, max_steps)
        tag = "solved in %d steps" % s if s else "NOT solved"
        print("  seed %d: %s (final max abs err %.4f, loss %.6f)"
              % (seed, tag, maxae, loss))
        solved += s is not None
    print("xavier: %d/10 seeds solved all four XOR outputs" % solved)

    print("part 2: zero init (prediction: plateau at loss 0.125)")
    net = MLP([2, 4, 1], seed=0, init="zeros")
    s, maxae, loss = train_xor(net, 5000)
    out = net.forward(Xxor).ravel()
    print("  after 5000 steps: loss=%.6f (predicted 0.125), "
          "outputs=%s, max abs err=%.4f, solved=%s"
          % (loss, np.array2string(out, precision=4), maxae, s is not None))
    ok_zero = (s is None and abs(loss - 0.125) < 1e-6
               and np.all(np.abs(out - 0.5) < 1e-6))
    print("ZERO-INIT PLATEAU CHECK: %s" % ("PASS" if ok_zero else "FAIL"))
    all_ok = all_ok and ok_zero

    print("part 3: init scale sweep (how small a seed can still grow)")
    for scale in (1e-8, 1e-6, 1e-3, 1e-1):
        res = []
        for seed in range(5):
            net = MLP([2, 4, 1], seed=seed, init=scale)
            s, _, _ = train_xor(net, 50000)
            res.append(s)
        nsolved = sum(s is not None for s in res)
        steps = sorted(s for s in res if s is not None)
        med = steps[len(steps) // 2] if steps else None
        print("  scale %.0e: %d/5 solved in 50000 steps, median steps: %s"
              % (scale, nsolved, med))
    print("part 3b: extended probe -- stuck or just slow?")
    for scale in (1e-3, 1e-6):
        net = MLP([2, 4, 1], seed=0, init=scale)
        s, maxae, loss = train_xor(net, 200000)
        print("  scale %.0e, 200000 steps: loss=%.6f maxae=%.4f solved=%s"
              % (scale, loss, maxae, s is not None))

    print("=== OVERALL: %s ===" % ("PASS" if all_ok else "FAIL"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
