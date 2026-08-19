"""
Fasad och kontraktskontroll - det du behöver för att koppla in ditt eget
gaussiska filter och din egen datakälla.

Två saker finns här:

  check_filter_contract()  kör ditt filter och kontrollerar att det uppfyller
                           kontraktet, med begripliga felmeddelanden. Kör den
                           FÖRST. Eftersom jag inte kan se ditt filter är det
                           här din enda automatiska kontroll av att sömmen
                           sitter rätt.

  SignalModel              en modell för EN signal. Träna med observe(),
                           mät med evaluate(), använd med predict_next().
                           Ingen matchning, ingen klustring - det du sa var
                           viktigt i det här steget.

Behöver du gruppering av flera olika signaler senare finns Cluster kvar i
main.py; SignalModel är en tunn inpackning av samma maskineri med
kluster-id 0.
"""

import numpy as np

from main import Cluster


# ----------------------------------------------------------------------
# Kontraktskontroll
# ----------------------------------------------------------------------

def check_filter_contract(state_filter, signal, verbose=True) -> list[str]:
    """Kontrollerar att ett filter uppfyller kontraktet. Returnerar problem.

    Kontraktet, som ditt filter redan följer:

        filter(data, update=True) -> (state_seq, param_vec, out)

        state_seq  (N, 2) heltal   kolumn 0 = komponentindex (rad i param_vec)
                                   kolumn 1 = rå körlängd, minst 1
        param_vec  (K, 6) float    medelvärde, varians, rått sekventiellt
                                   medelvärde, rå sekventiell varians, antal
                                   punkter, första värdet som initierade
                                   komponenten
        out        (T, 4) float    mu för vald komponent, varians, vald
                                   komponent, negativ log-likelihood

    Utöver formen kontrolleras fyra saker som inte syns i formen men som
    modellen är helt beroende av:

      * körlängderna summerar till antalet sampel. Gör de inte det har
        segmenteringen tappat eller dubblerat data, och varje varaktighet
        modellen lär sig blir fel.
      * update=False ändrar ingenting. Utan det lär sig filtret av testdatan
        medan den mäts och all hållen-ut-utvärdering blir meningslös.
      * radordningen i param_vec är stabil mellan anrop. Modellen använder
        RADINDEX som komponentens identitet för att upptäcka omnumrering.
        Sorterar du om raderna mellan anrop kan den inte se skillnad på "ny
        komponent" och "allt bytte plats".
      * antalet komponenter minskar aldrig. En komponent som försvinner tar
        med sig sin betydelse och lämnar trädet pekande på fel nivåer.
    """
    problem = []
    x = np.asarray(signal, dtype=float).ravel()

    try:
        state_seq, param_vec, out = state_filter.filter(x)
    except TypeError as e:
        problem.append(f"filter(data) gick inte att anropa: {e}")
        return _rapportera(problem, verbose)
    except Exception as e:
        problem.append(f"filter(data) kastade {type(e).__name__}: {e}")
        return _rapportera(problem, verbose)

    state_seq = np.asarray(state_seq)
    param_vec = np.asarray(param_vec)
    out = np.asarray(out)

    # -- form --
    if state_seq.ndim != 2 or state_seq.shape[1] != 2:
        problem.append(f"state_seq har formen {state_seq.shape}, väntade (N, 2)")
    if param_vec.ndim != 2 or param_vec.shape[1] != 6:
        problem.append(f"param_vec har formen {param_vec.shape}, väntade (K, 6)")
    if out.ndim != 2 or out.shape[1] != 4:
        problem.append(f"out har formen {out.shape}, väntade (T, 4)")
    if problem:
        return _rapportera(problem, verbose)

    # -- innehåll --
    if not np.all(state_seq[:, 1] >= 1):
        problem.append("state_seq innehåller körlängder < 1")
    summa = int(state_seq[:, 1].sum())
    if summa != len(x):
        problem.append(
            f"körlängderna summerar till {summa} men signalen har {len(x)} "
            f"sampel. Segmenteringen tappar eller dubblerar data.")
    if state_seq[:, 0].max(initial=-1) >= len(param_vec):
        problem.append(
            f"state_seq pekar på komponent {int(state_seq[:, 0].max())} men "
            f"param_vec har bara {len(param_vec)} rader")
    if len(out) != len(x):
        problem.append(f"out har {len(out)} rader, väntade {len(x)} (ett per sampel)")
    if np.any(~np.isfinite(param_vec[:, 0])):
        problem.append("param_vec kolumn 0 (medelvärde) innehåller NaN eller inf - "
                       "den används för rangordningen och måste vara ändlig")

    # -- inferensläge --
    try:
        före = param_vec.copy()
        state_filter.filter(x, update=False)
        _, efter, _ = state_filter.filter(x, update=False)
        efter = np.asarray(efter)
        if efter.shape != före.shape or not np.allclose(före, efter, equal_nan=True):
            problem.append(
                "update=False ändrar filtrets tillstånd. Utvärdering på "
                "hållen-ut data blir då inte hållen ut.")
    except TypeError:
        problem.append(
            "filter() tar inte emot update=False. Utan ett inferensläge lär "
            "sig filtret av testdatan medan den mäts.")

    # -- stabilitet över anrop --
    _, pv2, _ = state_filter.filter(x)
    pv2 = np.asarray(pv2)
    if len(pv2) < len(param_vec):
        problem.append(
            f"antalet komponenter minskade ({len(param_vec)} -> {len(pv2)}). "
            "Komponenter får födas men inte försvinna.")
    else:
        gemensamma = min(len(param_vec), len(pv2))
        flyttat = np.argsort(param_vec[:gemensamma, 0]) \
            .tolist() != np.argsort(pv2[:gemensamma, 0]).tolist()
        if flyttat:
            problem.append(
                "radordningen i param_vec verkar inte vara stabil mellan "
                "anrop. Radindex måste vara komponentens identitet.")

    return _rapportera(problem, verbose)


def _rapportera(problem, verbose):
    if verbose:
        if problem:
            print(f"KONTRAKTET UPPFYLLS INTE - {len(problem)} problem:")
            for p in problem:
                print(f"  * {p}")
        else:
            print("Kontraktet uppfylls. Filtret kan kopplas in.")
    return problem


# ----------------------------------------------------------------------
# Fasad
# ----------------------------------------------------------------------

class SignalModel:
    """En modell för EN signal, tränad på flera observationer av den.

    Typisk användning:

        modell = SignalModel(state_filter=MittFilter())
        modell.calibrate_noise(np.concatenate(träning[:8]))   # om filtret stödjer det
        for obs in träning:
            modell.observe(obs)

        print(modell.evaluate(hållen_ut))
        print(modell.predict_next(pågående_signal))
    """

    def __init__(self, state_filter, max_depth=6, per_value_counts=True,
                 min_obs_for_split=30, min_count=1, interpolate=True) -> None:
        self._cs = Cluster(max_depth=max_depth, min_count=min_count,
                           min_obs_for_split=min_obs_for_split,
                           per_value_counts=per_value_counts,
                           state_filter=state_filter)
        self._interpolate = interpolate
        self.state_filter = state_filter

    # -- uppsättning ---------------------------------------------------

    def calibrate_noise(self, signal):
        """Sätt filtrets brusnivå från representativ TRÄNINGSdata.

        Vidarebefordras till filtret om det har metoden. Har ditt eget filter
        en egen brusparameter ska du sätta den där i stället - men se till
        att den blir satt: fel brusnivå är den enskilt mest förödande
        inställningen i hela kedjan.
        """
        if hasattr(self.state_filter, "calibrate_noise"):
            return self.state_filter.calibrate_noise(signal)
        return None

    # -- träning -------------------------------------------------------

    def observe(self, signal) -> "SignalModel":
        self._cs.observe(signal, cluster_id=0)
        for t in self._cs.trees.values():
            t.interpolate = self._interpolate
        return self

    def observe_many(self, signals) -> "SignalModel":
        for s in signals:
            self.observe(s)
        return self

    # -- användning ----------------------------------------------------

    def evaluate(self, signal) -> dict:
        """Mät på en observation modellen INTE tränats på.

        Filtret körs i inferensläge, ingenting uppdateras.
        """
        return self._cs.evaluate(signal, cluster_id=0)

    def predict_next(self, signal) -> dict:
        """Vad kommer efter den här signalen?

        Returnerar den mest sannolika symbolen, den marginella fördelningen
        över enbart nivå (varaktigheten bortmarginaliserad), och hela
        fördelningen om du vill göra något annat med den.
        """
        pairs, _, _ = self._cs._preprocess(signal, update=False)
        if 0 not in self._cs.trees or not pairs:
            return {}
        model = self._cs.counts_params[0]
        symbols = self._cs._encode(model, pairs)
        probs = self._cs.trees[0].predict_probabilities(symbols)
        if not probs:
            return {}

        per_nivå: dict = {}
        for (v, _c), p in probs.items():
            per_nivå[v] = per_nivå.get(v, 0.0) + p
        bästa = max(probs, key=probs.get)
        return {
            "symbol": bästa,
            "p": probs[bästa],
            "value_id": max(per_nivå, key=per_nivå.get),
            "p_value": max(per_nivå.values()),
            "fördelning": probs,
            "nivåfördelning": per_nivå,
            "historik_symboler": len(symbols),
        }

    def score(self, signal) -> dict:
        """Hur väl passar den här signalen den tränade modellen?"""
        pairs, _, _ = self._cs._preprocess(signal, update=False)
        if 0 not in self._cs.trees or not pairs:
            return {}
        symbols = self._cs._encode(self._cs.counts_params[0], pairs)
        return self._cs.trees[0].score_sequence(symbols)

    # -- insyn ---------------------------------------------------------

    @property
    def tree(self):
        return self._cs.trees.get(0)

    @property
    def counts(self):
        return self._cs.counts_params.get(0)

    def summary(self) -> dict:
        träd = self.tree
        if träd is None:
            return {"tränad": False}
        return {
            "tränad": True,
            "observationer": len(self._cs._sequences.get(0, [])),
            "kontexter": len(träd.context_tree),
            "alfabet": len(träd.active_vocab),
            "antalskomponenter": len(self.counts.active),
            "omnumreringar": self._cs._relabels,
        }


# ----------------------------------------------------------------------

def learning_curve(state_filter_factory, träning, test, kontrollpunkter=None,
                   **modellargs) -> list[dict]:
    """Inlärningskurvan på din egen data.

    träning/test är listor av råa signaler (numpy-arrayer). Ett FÄRSKT filter
    skapas per körning via state_filter_factory, så kurvan inte förorenas av
    ett filter som redan sett datan.
    """
    if kontrollpunkter is None:
        kontrollpunkter = range(1, len(träning) + 1)
    kontrollpunkter = set(kontrollpunkter)

    m = SignalModel(state_filter_factory(), **modellargs)
    if träning:
        m.calibrate_noise(np.concatenate(träning[:min(8, len(träning))]))

    kurva = []
    for k, obs in enumerate(träning, start=1):
        m.observe(obs)
        if k not in kontrollpunkter:
            continue
        mät = [x for x in (m.evaluate(t) for t in test) if x]
        if not mät:
            continue
        kurva.append({
            "n_obs": k,
            **{nyckel: float(np.mean([x[nyckel] for x in mät]))
               for nyckel in ("accuracy", "value_accuracy", "avg_log_prob",
                              "perplexity", "avg_depth")},
            **m.summary(),
        })
    return kurva


if __name__ == "__main__":
    from gaussian_filter import GaussianFilter
    from main import DummyFilter

    rng = np.random.default_rng(0)
    sig = np.concatenate([rng.normal([0., 1., 2., 3.][i % 4], 0.2,
                                     size=rng.poisson(8) + 2) for i in range(40)])

    print("--- referensfilter: GaussianFilter ---")
    check_filter_contract(GaussianFilter(), sig)
    print("\n--- referensfilter: DummyFilter ---")
    check_filter_contract(DummyFilter(), sig)

    print("\n--- ett filter med en trasig segmentering ---")

    class TrasigtFilter(DummyFilter):
        def filter(self, data, update=True):
            ss, pv, out = super().filter(data, update=update)
            ss[0, 1] += 3          # körlängderna summerar inte längre
            return ss, pv, out

    check_filter_contract(TrasigtFilter(), sig)

    print("\n--- fasaden ---")
    träning = [np.concatenate([rng.normal([0., 1., 2., 3.][i % 4], 0.2,
                                          size=rng.poisson(8) + 2)
                               for i in range(8)]) for _ in range(20)]
    test = [np.concatenate([rng.normal([0., 1., 2., 3.][i % 4], 0.2,
                                       size=rng.poisson(8) + 2)
                            for i in range(30)]) for _ in range(4)]

    m = SignalModel(GaussianFilter(), max_depth=4)
    m.calibrate_noise(np.concatenate(träning[:8]))
    m.observe_many(träning)
    print("summary:", m.summary())
    r = m.evaluate(test[0])
    print(f"hållen ut: nivåträff {r['value_accuracy']:.3f}  "
          f"ppl {r['perplexity']:.2f}")
    p = m.predict_next(test[0])
    print(f"nästa: symbol {p['symbol']} (p={p['p']:.2f}), "
          f"nivå {p['value_id']} (p={p['p_value']:.2f})")
