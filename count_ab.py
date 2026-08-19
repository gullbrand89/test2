"""En delad antalsfordelning, eller en per varde-komponent?"""
import numpy as np
from gaussian_filter import GaussianFilter
from main import Cluster
import emitter_curve as ec
from emitter_data.Emitter import EmitterConfig, build_emitter


def kor(pri_typ, fro, langd, per_value, n_tran=25, n_test=6, max_depth=6):
    rng = np.random.default_rng(fro)
    cfg = EmitterConfig.generate(pri=pri_typ, freq=1, rng=rng)
    em = build_emitter(cfg)
    test = ec.inspelningar(em, n_test, rng)
    tran = ec.inspelningar(em, n_tran, rng, längd=langd)
    filt = GaussianFilter(); filt.calibrate_noise(np.concatenate(tran[:8]))
    cs = Cluster(max_depth=max_depth, min_count=1, min_obs_for_split=30,
                 per_value_counts=per_value, state_filter=filt)
    for o in tran:
        cs.observe(o, 0)
    m = [x for x in (cs.evaluate(t, 0) for t in test) if x]
    ra = [c for lst in cs.counts_params[0].raw.values() for c in lst]
    return {
        "niva": float(np.mean([x["value_accuracy"] for x in m])),
        "symbol": float(np.mean([x["accuracy"] for x in m])),
        "ppl": float(np.mean([x["perplexity"] for x in m])),
        "logp": float(np.mean([x["avg_log_prob"] for x in m])),
        "antalskomp": len(cs.counts_params[0].active),
        "alfabet": len(cs.trees[0].active_vocab),
        "overdisp": float(np.var(ra) / max(np.mean(ra), 1e-9)),
        "mk": cfg.mk,
    }


if __name__ == "__main__":
    fall = [("stagger", 2, 42, 12e-3), ("dwell & switch", 3, 2, 60e-3)]
    print("EN DELAD ANTALSFORDELNING, ELLER EN PER VARDE-KOMPONENT?")
    print("=" * 88)
    print(f"{'signal':<17}{'antalsmodell':<14}{'niva':>8}{'symbol':>9}{'ppl':>8}"
          f"{'logp':>9}{'komp':>7}{'alfabet':>9}{'overdisp':>10}")
    print("-" * 88)
    for namn, typ, fro, langd in fall:
        for per, etikett in ((False, "delad"), (True, "per niva")):
            r = kor(typ, fro, langd, per)
            print(f"{namn:<17}{etikett:<14}{r['niva']:>8.3f}{r['symbol']:>9.3f}"
                  f"{r['ppl']:>8.2f}{r['logp']:>9.3f}{r['antalskomp']:>7}"
                  f"{r['alfabet']:>9}{r['overdisp']:>10.1f}")
        print("-" * 88)
