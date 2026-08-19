"""
Inlärningskurva på riktiga signaler ur emitter_data.

Samma emitter, flera inspelningar. Generatorn ger precis det upplägg som
efterfrågades, och lite till:

  fasförskjutning  signal() börjar varje inspelning på ett slumpmässigt
                   pulsindex (start_index), så samma mönster kommer in med
                   olika utgångspunkt varje gång.
  brus             TOA-brus, som blir PRI-brus eftersom PRI är en differens
                   av två ankomsttider.
  pulsbortfall     upp till 10 % av pulserna försvinner vid låg SNR. Det är
                   hårdare än vanligt brus: när en puls faller bort SLÅS TVÅ
                   PRI IHOP till en, så det uppmätta värdet blir ungefär det
                   dubbla. Sekvensen får alltså inte bara en felaktig punkt
                   utan ett insticksfel som förskjuter allt efter sig.

Emitterns identitet (staggersekvensen, dwell-ordningen) härleds ur seed och
är densamma vid varje inspelning. Det är det som gör "samma signal" väl
definierat.

Signalen som matas in är PRI-serien i mikrosekunder. Värdemodellen blir
PRI-nivåerna, antalsmodellen hur många pulser i rad som ligger på varje nivå.
"""

import numpy as np

from emitter_data.Emitter import EmitterConfig, build_emitter
from gaussian_filter import GaussianFilter
from main import Cluster

KONTROLLPUNKTER = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40)


def inspelningar(emitter, n, rng, längd=None):
    """n inspelningar av SAMMA emitter - ny fas, nytt brus, nytt bortfall.

    `längd` är inspelningstiden i sekunder. TRÄNINGSinspelningarna görs korta
    med flit. Med full längd (100 ms) innehåller EN inspelning ett par hundra
    pulser, alltså tiotals varv av mönstret - och då är uppgiften löst efter
    första inspelningen, inte för att modellen är bra utan för att det inte
    finns något kvar att lära. Korta inspelningar tvingar informationen att
    ackumulera över inspelningar, vilket är det som ska mätas. Det är också
    det realistiska fallet: korta uppfångade snuttar, inte en lång inspelning.
    """
    from emitter_data.my_settings import SAMPLING_TIME
    ut = []
    for _ in range(n):
        df = emitter.signal(signal_length=SAMPLING_TIME if längd is None else längd,
                            noise=True,
                            noise_rng=np.random.default_rng(int(rng.integers(0, 2**62))))
        ut.append(df["pri"].values * 1e6)      # mikrosekunder
    return ut


def kör(pri_typ, namn, seed=42, n_träning=40, n_test=6, max_depth=6,
        träningslängd=4e-3, verbose=True):
    rng = np.random.default_rng(seed)
    cfg = EmitterConfig.generate(pri=pri_typ, freq=1, rng=rng)
    em = build_emitter(cfg)

    test = inspelningar(em, n_test, rng)                       # full längd
    träning = inspelningar(em, n_träning, rng, längd=träningslängd)

    # Brusnivån kalibreras på ihopslagen TRÄNINGSdata innan filtret startar.
    # Korta inspelningar räcker inte var för sig: skattaren måste kunna nå ett
    # tidsavstånd som motsvarar mönstrets period, och en 16-samplers snutt av
    # en stagger med period 14 innehåller inte det avståndet.
    filt = GaussianFilter()
    filt.calibrate_noise(np.concatenate(träning[:8]))

    cs = Cluster(max_depth=max_depth, min_count=1, min_obs_for_split=30,
                 state_filter=filt)

    if verbose:
        längder = [len(t) for t in träning]
        print(f"\n{namn.upper()}  (n={cfg.n}, mk={cfg.mk}, npri={cfg.npri}, "
              f"SNR={cfg.snr:.1f})")
        print(f"  {np.mean(längder):.0f} pulser per inspelning i snitt")
        print("-" * 74)

    kurva = []
    for k, obs in enumerate(träning, start=1):
        cs.observe(obs, cluster_id=0)
        if k not in KONTROLLPUNKTER:
            continue

        mät = [m for m in (cs.evaluate(t, 0) for t in test) if m]
        if not mät:
            continue
        rad = {
            "n_obs": k,
            "accuracy": float(np.mean([m["accuracy"] for m in mät])),
            "value_accuracy": float(np.mean([m["value_accuracy"] for m in mät])),
            "avg_log_prob": float(np.mean([m["avg_log_prob"] for m in mät])),
            "perplexity": float(np.mean([m["perplexity"] for m in mät])),
            "avg_depth": float(np.mean([m["avg_depth"] for m in mät])),
            "symboler_per_obs": float(np.mean([m["n_symbols"] for m in mät])),
            "kontexter": len(cs.trees[0].context_tree),
            "symboler": len(cs.trees[0].active_vocab),
            "värdekomp": sum(1 for c in cs.state_filter.components
                             if c.n >= cs.state_filter.min_support),
            "antalskomp": len(cs.counts_params[0].active),
        }
        kurva.append(rad)
        if verbose:
            print(f"  {k:>2} inspelningar: nivåträff {rad['value_accuracy']:.3f}  "
                  f"symbolträff {rad['accuracy']:.3f}  "
                  f"logp {rad['avg_log_prob']:+.3f}  "
                  f"ppl {rad['perplexity']:7.2f}  djup {rad['avg_depth']:.2f}  "
                  f"kontexter {rad['kontexter']:>5}  "
                  f"värdekomp {rad['värdekomp']:>2}  antalskomp {rad['antalskomp']}")

    return kurva, cs, cfg


if __name__ == "__main__":
    import json

    print("INLÄRNINGSKURVA PÅ RIKTIGA EMITTERSIGNALER")
    print("Hållen-ut utvärdering: 6 inspelningar modellen aldrig tränats på.")

    allt = {}
    # Dwell & switch använder seed 2: med seed 42 råkar generatorn ge EN enda
    # dwell (mk=[13]), alltså konstant PRI utan något att växla mellan - ett
    # degenererat fall som inte säger något om modulationen. Seed 2 ger sex
    # dwells.
    # Inspelningslängderna är valda så att en enskild träningsinspelning
    # täcker ett par varv av mönstret - tillräckligt för att bidra, för lite
    # för att ensam lösa uppgiften.
    for pri_typ, namn, frö, längd in ((2, "stagger", 42, 12e-3),
                                      (3, "dwell & switch", 2, 60e-3)):
        kurva, cs, cfg = kör(pri_typ, namn, seed=frö, träningslängd=längd)
        allt[namn] = kurva
        f, s = kurva[0], kurva[-1]
        print(f"  => nivåträff {f['value_accuracy']:.3f} -> {s['value_accuracy']:.3f}   "
              f"perplexitet {f['perplexity']:.2f} -> {s['perplexity']:.2f}")
        print(f"     skattad brusnivå: {np.sqrt(cs.state_filter.init_var):.5f} us")

    json.dump(allt, open("emitter_kurvor.json", "w"))
    print("\nsparade emitter_kurvor.json")
