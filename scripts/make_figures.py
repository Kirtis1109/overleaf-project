"""Reproduce the supporting figures/tables for Theorem 1, Corollary 1 and Sec. IV.
Constants: a=0.01, Ts=1, U=40*sqrt(2), M=17, r=9; outputs = nodal mean and nodes
nearest z=0.25, 0.70; two Dirichlet boundary inputs in [-40,20]^2 (centered at 40 C)."""
import json, numpy as np, mpmath as mp
from pathlib import Path
mp.mp.dps = 60
a, Ts, M, U, r = mp.mpf('0.01'), mp.mpf(1), mp.mpf(17), 40*mp.sqrt(2), 9
c = 4*a*Ts
root = Path(__file__).resolve().parents[1]; out = {}

def lam(k, N): return mp.e**(-c*(N+1)**2*mp.sin(k*mp.pi/(2*(N+1)))**2)
def q(k, N0): return lam(k, N0) if (N0 is not None and k <= N0) else mp.e**(-c*k**2)
def ubound(r, t0, N0, K=400):
    ini = mp.sqrt(mp.fsum(q(k, N0)**(2*t0) for k in range(r+1, K+1)))
    frc = mp.fsum(q(k, N0)/k for k in range(r+1, K+1))
    return float(mp.sqrt(2)*M*ini + 2*mp.sqrt(2)*U/mp.pi*frc)   # tail beyond K < 1e-2000

def modal(N):
    k = np.arange(1, N+1); th = k*np.pi/(N+1)
    lm = np.exp(-float(c)*(N+1)**2*np.sin(th/2)**2)
    bL = (1-lm)/np.tan(th/2)/(np.sqrt(2)*(N+1)); bR = (-1)**(k+1)*bL
    j = np.arange(1, N+1); phi = np.sqrt(2)*np.sin(np.outer(k, j)*np.pi/(N+1))  # phi[k,j]
    W = np.zeros((3, N)); W[0] = 1/N
    for i, z in enumerate((0.25, 0.70)): W[i+1, int(round(z*(N+1)))-1] = 1
    C = W @ phi.T                     # c_{ik}
    return lm, np.stack([bL, bR], 1), C, phi
def worst_case(N, T=300):
    lm, b, C, _ = modal(N); f = slice(r, N); best = 0
    for i in range(3):
        cf, lf, bf = C[i, f], lm[f], b[f]
        forced = np.cumsum([np.linalg.norm((cf*lf**l) @ bf) for l in range(1, T)])
        for t in range(1, T):
            w = float(M)*np.sqrt(np.sum(cf**2*lf**(2*t))) + float(U)*(forced[t-2] if t > 1 else 0)
            best = max(best, w)
    return best
def empirical(N, seeds=20, T=300):
    lm, b, C, phi = modal(N); f = slice(r, N)
    xi0 = (phi @ np.full(N, -17.0))/(N+1)          # flat -17 C deviation profile
    worst = 0
    for s in range(seeds):
        u = np.random.default_rng(1000+s).uniform(-40, 20, (T, 2)); xi = xi0.copy()
        for t in range(T):
            xi_new = lm*xi + b @ u[t]
            d = C[:, f] @ (xi_new[f] - b[f] @ u[t])   # residual after removing newest input
            worst = max(worst, np.abs(d).max()); xi = xi_new
    return worst

# --- checks against the paper's interval values (Table I / fixed-mesh suprema)
out['check'] = {f'N0={n},t0={t}': ubound(r, t, n) for n in (16, 20, 50) for t in (1, 3)}
out['check_W'] = {n: worst_case(n) for n in (16, 50)}

# --- Fig. A: bound vs mesh
Ns = [16, 20, 25, 30, 40, 50, 75, 100, 150, 200]
figA = {'N': Ns, 'bound_t1': [ubound(r, 1, n) for n in Ns], 'bound_t3': [ubound(r, 3, n) for n in Ns],
        'worst': [worst_case(n) for n in Ns], 'emp': [empirical(n) for n in Ns]}
out['figA'] = figA
# --- Fig. B / Table: Corollary 1 (q_k = e^{-ck^2}, valid for every N >= r)
rs = list(range(0, 21))
figB = {'r': rs, 'N16': [ubound(rr, 1, 16) for rr in rs if rr <= 16], 'N50': [ubound(rr, 1, 50) for rr in rs],
        'free': [ubound(rr, 1, None) for rr in rs]}
out['figB'] = figB
out['minr'] = {}
for ds in (1e-1, 1e-2, 1e-3, 1e-4):
    out['minr'][ds] = [next((rr for rr in range(60) if (n is None or rr <= n) and ubound(rr, 1, n) <= ds), None)
                       for n in (16, 50, None)]
json.dump(out, open(root/'scripts'/'results.json', 'w'), indent=1, default=str)

# --- plots (IEEE single column, print-safe: distinct markers + line styles)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 4,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'mathtext.fontset': 'stix'})
B, O, A, Y = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
fig, ax = plt.subplots(figsize=(3.4, 2.0))
ax.semilogy(Ns, figA['bound_t1'], '-o', color=B, label=r'Theorem 1 bound (all meshes $\geq N$)')
ax.semilogy(Ns, figA['worst'], '--^', color=O, label=r'exact worst case at mesh $N$')
ax.set_xlabel(r'interior nodes $N$'); ax.set_ylabel(r'output residual ($^\circ$C)')
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([1e-3, 2e-3, 5e-3, 1e-2, 2e-2]); ax.set_yticklabels(['0.001', '0.002', '0.005', '0.01', '0.02']); ax.set_ylim(1e-3, 2.5e-2)
ax.grid(True, which='major', color='#dddddd', linewidth=0.4); ax.legend(fontsize=7, loc='upper right')
fig.tight_layout(pad=0.3); fig.savefig(root/'figures'/'bound_vs_mesh.pdf')
fig, ax = plt.subplots(figsize=(3.4, 2.1))
r16 = [rr for rr in rs if rr <= 16]
ax.semilogy(r16, figB['N16'], '-o', color=B, label=r'$N\geq16$')
ax.semilogy(rs, figB['N50'], '--s', color=A, label=r'$N\geq50$')
ax.semilogy(rs, figB['free'], ':^', color=O, label=r'$N\geq r$ (Corollary 1)')
ax.axvline(9, color='#888888', linewidth=0.6, linestyle='-', zorder=0)
ax.text(9.25, 2e2, r'$r=9$', fontsize=7, color='#555555')
ax.set_xlabel(r'retained modes $r$ (auxiliary order $\leq r+2$)'); ax.set_ylabel(r'bound $\bar\delta$, $t_0=1$ ($^\circ$C)')
ax.set_xticks(range(0, 21, 2)); ax.set_ylim(1e-8, 1e3)
ax.grid(True, which='major', color='#dddddd', linewidth=0.4); ax.legend(fontsize=7, loc='lower left')
fig.tight_layout(pad=0.3); fig.savefig(root/'figures'/'bound_vs_r.pdf')
print(json.dumps({k: out[k] for k in ('check', 'check_W', 'minr')}, indent=1, default=str))
print(json.dumps(figA, indent=0))
