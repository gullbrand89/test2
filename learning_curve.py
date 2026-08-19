"""
Inlärningskurva: blir prediktionen bättre ju fler observationer modellen ser?

Flera observationer av SAMMA signal, men med olika fasförskjutning (varje
observation börjar på ett slumpmässigt ställe i cykeln), färskt gaussiskt
brus på nivåerna och färska Poisson-dragningar på varaktigheterna.

MÄTDISCIPLIN
Utvärderingsmängden hålls helt utanför träningen och mäts om efter varje ny
träningsobservation. Filtret körs i inferensläge (update=False) under
mätning - annars lär det sig av testdatan medan den mäts, och kurvan mäter
ingenting.

TVÅ TRÄFFSÄKERHETER
accuracy kräver exakt rätt symbol, alltså rätt nivå OCH rätt
varaktighetsklass. Den har ett tak som ingen mängd träning kan passera:
varaktigheterna är Poissondragna, så vilken antalskomponent nästa körning
hamnar i är delvis ren slump. value_accuracy kräver bara rätt nivå, och det
är där den lärbara strukturen finns. Planar den första ut medan den andra
stiger vidare är taket brus, inte okunskap.
"""

import numpy as np

from gaussian_filter import GaussianFilter
from main import Cluster

# En periodisk "cyklistprofil": (nivå, förväntad varaktighet).
# Varaktigheten hör till positionen i mönstret, så det finns struktur att
# lära sig även i tidsdimensionen - inte bara i ordningen av nivåer.
MÖNSTER = [(0.0, 4), (1.0, 12), (2.0, 4), (1.0, 12), (3.0, 6), (1.0, 4)]
BRUS = 0.2


def generera(rng, n_block=30, fas=None):
    """En observation: samma mönster, ny fas, nytt brus, nya varaktigheter."""
    if fas is None:
        fas = int(rng.integers(len(MÖNSTER)))
    bitar = []
    for i in range(n_block):
        nivå, lam = MÖNSTER[(fas + i) % len(MÖNSTER)]
        d = int(rng.poisson(lam)) + 1
        bitar.append(rng.normal(nivå, BRUS, size=d))
    return np.concatenate(bitar)


def kör(seed=0, n_träning=60, n_test=8, max_depth=4, block_träning=8,
        block_test=30, interpolate=True, verbose=True):
    """Träningsobservationerna är KORTA med flit.

    Med långa observationer (30 block ~ 5 varv av ett mönster med period 6)
    innehåller EN observation redan nästan all struktur som finns att lära,
    och kurvan blir platt från början - inte för att modellen är dålig utan
    för att experimentet inte lämnar något kvar att lära sig. Korta
    observationer (8 block ~ 1.3 varv) tvingar informationen att ackumuleras
    över observationer, vilket är det som ska mätas.

    Testobservationerna är långa, för att ge stabila mätvärden.
    """
    rng = np.random.default_rng(seed)
    test = [generera(rng, n_block=block_test) for _ in range(n_test)]
    träning = [generera(rng, n_block=block_träning) for _ in range(n_träning)]

    cs = Cluster(max_depth=max_depth, min_count=1, min_obs_for_split=30,
                 state_filter=GaussianFilter())

    kurva = []
    n_splittar_innan = 0
    for k, obs in enumerate(träning, start=1):
        cs.observe(obs, cluster_id=0)
        for t in cs.trees.values():
            t.interpolate = interpolate
        n_splittar = len(cs.counts_params[0].retired_ids)
        delade = n_splittar > n_splittar_innan
        n_splittar_innan = n_splittar

        mät = [cs.evaluate(t, cluster_id=0) for t in test]
        mät = [m for m in mät if m]
        rad = {
            "n_obs": k,
            "accuracy": float(np.mean([m["accuracy"] for m in mät])),
            "value_accuracy": float(np.mean([m["value_accuracy"] for m in mät])),
            "avg_log_prob": float(np.mean([m["avg_log_prob"] for m in mät])),
            "perplexity": float(np.mean([m["perplexity"] for m in mät])),
            "avg_depth": float(np.mean([m["avg_depth"] for m in mät])),
            "kontexter": len(cs.trees[0].context_tree),
            "symboler": len(cs.trees[0].active_vocab),
            "antalskomp": len(cs.counts_params[0].active),
            "splittar": n_splittar,
            "split_nu": delade,
        }
        kurva.append(rad)

        if verbose:
            print(f"  {k:>2} obs: träff {rad['accuracy']:.3f}  "
                  f"nivåträff {rad['value_accuracy']:.3f}  "
                  f"logp {rad['avg_log_prob']:+.3f}  "
                  f"ppl {rad['perplexity']:6.2f}  "
                  f"djup {rad['avg_depth']:.2f}  "
                  f"kontexter {rad['kontexter']:>5}  "
                  f"antalskomp {rad['antalskomp']}"
                  + ("   <- BIC-split" if delade else ""))
    return kurva, cs


if __name__ == "__main__":
    print("INLÄRNINGSKURVA (hållen-ut utvärdering, 8 testobservationer)")
    print("-" * 78)
    kurva, cs = kör(seed=0)

    f, s = kurva[0], kurva[-1]
    print("\nFÖRSTA -> SISTA")
    print("-" * 78)
    for nyckel, etikett in (("accuracy", "träffsäkerhet (symbol)"),
                            ("value_accuracy", "träffsäkerhet (nivå)  "),
                            ("avg_log_prob", "log-sannolikhet       "),
                            ("perplexity", "perplexitet           "),
                            ("avg_depth", "kontextdjup           ")):
        print(f"  {etikett}: {f[nyckel]:+8.3f}  ->  {s[nyckel]:+8.3f}")

    # Monotonitet: hur ofta går det bakåt?
    acc = [r["value_accuracy"] for r in kurva]
    bakåt = sum(1 for a, b in zip(acc, acc[1:]) if b < a - 1e-9)
    print(f"\n  nivåträffsäkerheten gick bakåt i {bakåt} av {len(acc)-1} steg")
    print(f"  bästa halvan snitt: {np.mean(acc[len(acc)//2:]):.3f}  "
          f"första halvan snitt: {np.mean(acc[:len(acc)//2]):.3f}")
