"""Illustrative normalized calculations for the thesis.

These calculations are deliberately labeled as surrogate/model demonstrations.
They do not replace relativistic differential cross-section integration or Monte
Carlo radiation transport.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def mj_energy_pdf(E: np.ndarray, theta: float) -> np.ndarray:
    """Maxwell-Juttner probability density in kinetic energy E/(m_ec^2)."""
    gamma = 1.0 + E
    beta = np.sqrt(np.maximum(1.0 - 1.0 / gamma**2, 0.0))
    raw = gamma**2 * beta * np.exp(-gamma / theta)
    return raw / trapezoid(raw, E)


def redistributed_pdf(E: np.ndarray, base: np.ndarray, cutoff: float,
                      center: float, width: float) -> np.ndarray:
    keep = base.copy()
    mask = E > cutoff
    tail_n = trapezoid(base[mask], E[mask])
    keep[mask] = 0.0
    g = np.exp(-0.5 * ((E - center) / width) ** 2)
    g /= trapezoid(g, E)
    out = keep + tail_n * g
    return out / trapezoid(out, E)


def moment(E: np.ndarray, f: np.ndarray, q: float) -> float:
    return float(trapezoid((E**q) * f, E))

E = np.linspace(1e-5, 8.0, 12000)
theta = 0.40
base = mj_energy_pdf(E, theta)
cutoff = 1.20  # 3 kT when theta=0.4
redist = redistributed_pdf(E, base, cutoff=cutoff, center=0.30, width=0.055)

plt.figure(figsize=(7.2, 4.8))
plt.semilogy(E, base, label="Maxwell-Juttner reference")
plt.semilogy(E, redist, label="Cutoff plus redistribution")
plt.axvline(cutoff, linestyle="--", label="Cutoff")
plt.xlabel(r"Kinetic energy $E/(m_ec^2)$")
plt.ylabel("Normalized probability density")
plt.xlim(0, 4.0)
plt.ylim(1e-7, None)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "distribution_cutoff.pdf")
plt.savefig(FIG / "distribution_cutoff.png", dpi=220)
plt.close()

cutoffs = np.linspace(0.45, 3.0, 70)
qs = [1.0, 2.0, 3.0]
plt.figure(figsize=(7.2, 4.8))
for q in qs:
    ratios = []
    for ec in cutoffs:
        center = min(0.30, 0.35 * ec)
        f = redistributed_pdf(E, base, cutoff=ec, center=center, width=0.055)
        ratios.append(moment(E, f, q) / moment(E, base, q))
    plt.plot(cutoffs / theta, ratios, label=fr"Moment exponent $q={q:g}$")
plt.xlabel(r"Cutoff energy $E_c/(k_BT_e)$")
plt.ylabel("Radiation-weighted moment ratio")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "moment_reduction.pdf")
plt.savefig(FIG / "moment_reduction.png", dpi=220)
plt.close()

# Surrogate photon spectra: exponential bremsstrahlung-like shape with a
# suppression factor strongest at high photon energy.
Eg = np.linspace(0.002, 3.0, 5000)
S0 = Eg * np.exp(-Eg / theta)
S0 /= trapezoid(S0, Eg)

def spectrum_ratio(ec: float) -> np.ndarray:
    # Smooth high-energy suppression; values remain between 0.18 and 1.
    return 1.0 - 0.82 / (1.0 + np.exp(-(Eg - 0.55 * ec) / 0.10))

weights = {
    "Energy fluence": np.ones_like(Eg),
    "Shallow-tissue weighting": 1.0 / np.sqrt(Eg + 0.035),
    "Penetrating-dose weighting": np.sqrt(Eg + 0.02),
}

power_ratio = []
dose_ratios = {k: [] for k in weights}
for ec in cutoffs:
    r = spectrum_ratio(ec)
    S = S0 * r
    power_ratio.append(trapezoid(S, Eg) / trapezoid(S0, Eg))
    for name, w in weights.items():
        dose_ratios[name].append(trapezoid(S * w, Eg) / trapezoid(S0 * w, Eg))

plt.figure(figsize=(7.2, 4.8))
plt.plot(cutoffs / theta, power_ratio, label="Integrated source-power ratio")
for name, vals in dose_ratios.items():
    plt.plot(cutoffs / theta, vals, label=name)
plt.xlabel(r"Cutoff energy $E_c/(k_BT_e)$")
plt.ylabel("Normalized response ratio")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "dose_vs_power.pdf")
plt.savefig(FIG / "dose_vs_power.png", dpi=220)
plt.close()

# Summary table for selected cutoff values.
selected = [1.5, 2.0, 3.0, 4.0, 5.0]
rows = []
for mult in selected:
    ec = mult * theta
    f = redistributed_pdf(E, base, cutoff=ec, center=min(0.30, 0.35 * ec), width=0.055)
    rows.append([
        mult,
        trapezoid(base[E > ec], E[E > ec]),
        moment(E, f, 1) / moment(E, base, 1),
        moment(E, f, 2) / moment(E, base, 2),
        moment(E, f, 3) / moment(E, base, 3),
    ])
np.savetxt(ROOT / "illustrative_results.csv", np.array(rows), delimiter=",",
           header="Ec_over_kT,tail_number_fraction,M1_ratio,M2_ratio,M3_ratio",
           comments="")
