"""
Hur mycket kostar pulsbortfallet? Alltså: vad skulle en bortfallsdetektor
vara värd, mätt innan man bygger den.

Din intuition stämmer, och konsekvensen är värre än en felaktig punkt. En
tappad puls mitt i en dwell av längd 48 ger:

    (v, 48)   ->   (v, a)  +  (v_dubbel, 1)  +  (v, b)     med a + b = 47

Alltså tre effekter på en gång:
  * dwellen skärs i två kortare körningar, så varaktighetsfördelningen
    smetas ut och blir överdispergerad - vilket BIC svarar på med att dela
    Poissonkomponenten om och om igen
  * ett falskt PRI-värde vid ungefär dubbla nivån föder en EGEN
    värdekomponent, som aldrig dör
  * sekvensen får ett insticksfel som förskjuter kontexten för allt efter

Tre körningar jämförs, allt annat lika:
  hög SNR    bortfallsfrekvensen nära noll
  låg SNR    ca 10 % bortfall
  låg SNR utan bortfall - samma mätbrus, men bortfallet avstängt. Det är
             taket en perfekt bortfallsdetektor skulle nå.
"""

import numpy as np

import emitter_data.Emitter as EM
from emitter_data.Emitter import EmitterConfig, build_emitter
from gaussian_filter import GaussianFilter
from main import Cluster
import emitter_curve as ec


def kör(snr, bortfall=True, n_träning=25, n_test=6, seed=2, längd=60e-3):
    rng = np.random.default_rng(seed)
    cfg = EmitterConfig.generate(pri=3, freq=1, rng=rng)
    cfg.snr = snr

    # Bortfallet läses ur modulens globaler vid anropet, så det går att
    # stänga av utan att röra mätbruset.
    hög, låg = EM.MISSING_PULSE_HIGH, EM.MISSING_PULSE_LOW
    if not bortfall:
        EM.MISSING_PULSE_HIGH = EM.MISSING_PULSE_LOW = 0.0
    try:
        em = build_emitter(cfg)
        test = ec.inspelningar(em, n_test, rng)
        träning = ec.inspelningar(em, n_träning, rng, längd=längd)

        filt = GaussianFilter()
        filt.calibrate_noise(np.concatenate(träning[:8]))
        cs = Cluster(max_depth=6, min_count=1, min_obs_for_split=30,
                     state_filter=filt)
        for o in träning:
            cs.observe(o, 0)
        mät = [m for m in (cs.evaluate(t, 0) for t in test) if m]
    finally:
        EM.MISSING_PULSE_HIGH, EM.MISSING_PULSE_LOW = hög, låg

    rå = [c for lst in cs.counts_params[0].raw.values() for c in lst]
    a = -0.1 / (30 - 7)
    return {
        "bortfallsfrekvens": a * snr + (0 - a * 30) if bortfall else 0.0,
        "nivåträff": float(np.mean([m["value_accuracy"] for m in mät])),
        "symbolträff": float(np.mean([m["accuracy"] for m in mät])),
        "perplexitet": float(np.mean([m["perplexity"] for m in mät])),
        "värdekomp": sum(1 for c in filt.components if c.n >= filt.min_support),
        "antalskomp": len(cs.counts_params[0].active),
        "överdispersion": float(np.var(rå) / max(np.mean(rå), 1e-9)),
        "symboler_per_inspelning": float(np.mean([m["n_symbols"] for m in mät])),
        "mk": cfg.mk,
    }


if __name__ == "__main__":
    fall = [("hög SNR (30)", 30.0, True),
            ("låg SNR (7)", 7.0, True),
            ("låg SNR, bortfall AV", 7.0, False)]

    print("VAD KOSTAR PULSBORTFALLET?   (dwell & switch, 25 inspelningar)")
    print("=" * 92)
    print(f"{'fall':<24}{'bortfall':>9}{'nivåträff':>11}{'symbol':>9}"
          f"{'ppl':>8}{'värdek':>8}{'antalsk':>9}{'överdisp':>10}{'symb/insp':>11}")
    print("-" * 92)
    for namn, snr, bf in fall:
        r = kör(snr, bortfall=bf)
        print(f"{namn:<24}{r['bortfallsfrekvens']*100:>8.1f}%{r['nivåträff']:>11.3f}"
              f"{r['symbolträff']:>9.3f}{r['perplexitet']:>8.2f}{r['värdekomp']:>8}"
              f"{r['antalskomp']:>9}{r['överdispersion']:>10.1f}"
              f"{r['symboler_per_inspelning']:>11.0f}")
    print("-" * 92)
    print("överdisp = varians/medelvärde för körlängderna. En Poisson kräver 1.")
    print("symb/insp = antal RLE-symboler per inspelning. Fler = dwellarna är")
    print("            sönderskurna, eftersom en hel dwell borde bli EN symbol.")
