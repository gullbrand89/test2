"""
CustomVMM - en Variable-order Markov Model (VMM) för RLE-symboler.

Symbolerna är odelbara tupler (värde_id, antal_id), t.ex. (5, 1):
  värde_id  - pekar in i värdemodellen. Antas konstant. Idna är sorterade
              i storleksordning efter komponenternas medelvärde.
  antal_id  - pekar in i antalsmodellen (Poisson). Denna är DYNAMISK: mellan
              observationer testas med BIC om 1 eller 2 komponenter passar
              bäst. Blir det två, pensioneras det gamla idt permanent och
              två nya tar över.

Det dynamiska vokabuläret är det som styr designen här. Tre saker följer:

  1. Ett pensionerat id kan aldrig återkomma, men ligger kvar i trädet och
     stjäl sannolikhetsmassa från de levande symbolerna. Därav `retired`
     och filtreringen i _predict_with_depth().

  2. När en komponent delas är den gamla statistiken inte värdelös - de nya
     komponenterna ÄR den gamla, uppdelad. split_count_component() flyttar
     över räknarna proportionellt istället för att kasta dem. Utan det
     nollställs modellen i praktiken vid varje split.

  3. Ett nyskapat id har per definition aldrig setts, och ser därför
     maximalt osannolikt ut i scoringen - trots att det inte är avvikande,
     bara nytt. score_sequence() räknar därför NOVA steg separat.

Räknarna är float, inte int, eftersom en split delar dem proportionellt.
"""

import math
from collections import defaultdict
from itertools import product
from typing import Any, Iterable, Sequence


class CustomVMM:
    def __init__(self, max_depth=5, min_count=1, interpolate=True) -> None:
        # Hur långt bakåt vi som mest tittar
        self.max_depth = max_depth
        # Hur många observationer en kontext måste ha för att få vara kvar
        self.min_count = min_count
        # True = väg samman alla kontextlängder (se _predict_interpolated).
        # False = hård back-off, ta längsta kontext som setts.
        self.interpolate = interpolate
        # context_tree[kontext][nästa_symbol] = vikt (antal observationer)
        # Kontext = tuple av symboler, alltså en tuple av tupler:
        #   ((5, 1), (3, 2)) betyder "efter (5,1) följt av (3,2)".
        self.context_tree = defaultdict(lambda: defaultdict(float))
        # Alla symboler modellen någonsin sett
        self.vocab: set = set()
        # Symboler vars antal_id pensionerats - kan aldrig förekomma igen
        self.retired: set = set()
        # Sätts av calibrate()
        self._baseline_mean: float | None = None
        self._baseline_std: float | None = None

    @property
    def active_vocab(self) -> set:
        """Symboler som fortfarande kan förekomma."""
        return self.vocab - self.retired

    # ------------------------------------------------------------------
    # Indatakontroll
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sequence(sequence, argname="sequence") -> list:
        """Fångar det klassiska misstaget att skicka in EN symbol där en
        sekvens av symboler förväntas.

        (5, 1) är en symbol. [(5, 1)] är en sekvens med en symbol. Utan den
        här kontrollen tolkas (5, 1) tyst som två symboler, 5 och 1, och
        allt fortsätter fungera - men på fel data.
        """
        if isinstance(sequence, tuple) and sequence and not isinstance(sequence[0], tuple):
            raise TypeError(
                f"{argname}={sequence!r} ser ut som EN symbol, inte en sekvens. "
                f"Skicka [{sequence!r}] om det var meningen."
            )
        return list(sequence)

    # ------------------------------------------------------------------
    # Träning
    # ------------------------------------------------------------------

    def fit(self, sequence) -> None:
        # sequence är en lista av symboler: [(5,1), (3,2), (5,1), ...]
        sequence = self._check_sequence(sequence)
        self.vocab.update(sequence)

        for i in range(len(sequence) - 1):
            next_sym = sequence[i + 1]

            # Rot-kontexten: hur ofta varje symbol förekommer överhuvudtaget.
            # Back-off:ens sista utväg när ingen längre kontext matchar.
            self.context_tree[()][next_sym] += 1.0

            # Alla kontexter som slutar på position i, längd 1..max_depth
            for depth in range(self.max_depth):
                if i - depth < 0:
                    break  # slut på historik åt vänster
                context = tuple(sequence[i - depth: i + 1])
                self.context_tree[context][next_sym] += 1.0

    def fit_many(self, sequences) -> None:
        """Träna på flera separata sekvenser (t.ex. en per cyklist/pass)."""
        for seq in sequences:
            self.fit(seq)

    def prune(self) -> None:
        # OBS: räknarna är float. Efter en split kan legitima grenar ha vikt
        # under 1.0 och åker då ut med min_count=1. Vill du behålla dem,
        # sätt min_count lägre eller prune före split.
        for context in list(self.context_tree.keys()):
            if context == ():
                continue  # roten sparas alltid - annars kollapsar back-off
            total_observations = sum(self.context_tree[context].values())
            if total_observations < self.min_count:
                del self.context_tree[context]

    # ------------------------------------------------------------------
    # Dynamiskt vokabulär: pensionering och splittar
    # ------------------------------------------------------------------

    def retire_count_id(self, old_cid) -> set:
        """Markera ett antal_id som pensionerat utan att flytta statistik.

        Symbolerna ligger kvar i trädet men filtreras bort vid prediktion,
        så de stjäl ingen sannolikhetsmassa. Använd split_count_component()
        istället när du vet vilka nya id:n som ersatte det gamla - då
        återanvänds den gamla statistiken.
        """
        dead = {s for s in self.vocab if s[1] == old_cid}
        self.retired |= dead
        return dead

    def split_count_component(self, old_cid, new_cids, weights=None,
                              min_weight=1e-3) -> dict[str, int]:
        """BIC valde två Poisson-komponenter där det förut fanns en.

        Varje symbol (v, old_cid) ersätts av (v, new_cids[0]) och
        (v, new_cids[1]), och räknarna delas enligt `weights` - använd
        blandningsproportionerna från Poisson-anpassningen. Utan vikter
        antas jämn fördelning.

        Delningen sker på båda ställen en symbol kan förekomma: som
        prediktionsmål (inre dicts) och inne i kontextnycklarna.

        En kontext som innehåller det gamla idt på k positioner expanderar
        till 2^k nya kontexter. Vikterna multipliceras längs vägen, så
        fragmenten krymper snabbt - `min_weight` slänger de obetydliga.
        Vid många splittar i rad, kör prune() emellanåt.
        """
        if weights is None:
            weights = [1.0 / len(new_cids)] * len(new_cids)
        if len(weights) != len(new_cids):
            raise ValueError("weights och new_cids måste vara lika långa.")
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Gammal symbol -> [(ny symbol, andel), ...]
        mapping = {
            s: [((s[0], cid), w) for cid, w in zip(new_cids, weights)]
            for s in self.vocab if s[1] == old_cid
        }
        return self.remap_symbols(mapping, min_weight=min_weight)

    def relabel_value_ids(self, permutation) -> dict[str, int]:
        """Värde-idna har bytt betydelse - flytta hela trädet med dem.

        Behövs när värdemodellen numrerar om sig. Med kanoniska id satta av
        rangordning efter medelvärde räcker det att EN ny komponent föds med
        ett medelvärde som sorterar in mellan två befintliga: allt ovanför
        skjuts upp ett steg, och varje symbol trädet lärt sig pekar plötsligt
        på fel fysisk nivå. Tyst och förödande, till skillnad från en
        pensionering som åtminstone syns som ett dött id.

        permutation: {gammalt värde_id: nytt värde_id}. Ren ompermutering,
        inga vikter delas - det är samma komponent med ny etikett.
        """
        mapping = {
            s: [((permutation[s[0]], s[1]), 1.0)]
            for s in self.vocab if s[0] in permutation
            and permutation[s[0]] != s[0]
        }
        return self.remap_symbols(mapping)

    def remap_symbols(self, mapping, min_weight=1e-3) -> dict[str, int]:
        """Skriv om trädet enligt {gammal symbol: [(ny symbol, andel), ...]}.

        Gemensam motor för split (1 -> 2 med vikter) och ompermutering
        (1 -> 1 med vikt 1.0). Symboler byts på BÅDA ställen de förekommer:
        som prediktionsmål i de inre dictarna, och inne i kontextnycklarna.
        """
        if not mapping:
            return {"contexts_expanded": 0, "symbols_replaced": 0}

        new_tree = defaultdict(lambda: defaultdict(float))
        contexts_expanded = 0

        for context, counts in self.context_tree.items():
            # Expandera kontextnyckeln: kryssprodukt över positioner som
            # innehåller en symbol som ska bytas.
            slots = [mapping.get(sym, [(sym, 1.0)]) for sym in context]
            variants = list(product(*slots)) if context else [()]
            if len(variants) > 1:
                contexts_expanded += 1

            for variant in variants:
                new_context = tuple(sym for sym, _ in variant)
                ctx_w = 1.0
                for _, w in variant:
                    ctx_w *= w

                for target, c in counts.items():
                    for new_target, tw in mapping.get(target, [(target, 1.0)]):
                        val = c * ctx_w * tw
                        if val >= min_weight:
                            new_tree[new_context][new_target] += val

        self.context_tree = new_tree
        self.vocab -= set(mapping)
        self.vocab |= {ns for repl in mapping.values() for ns, _ in repl}
        # Ersatta symboler är nu helt borta ur trädet - ingen anledning att
        # fortsätta filtrera på dem.
        self.retired -= set(mapping)

        return {"contexts_expanded": contexts_expanded,
                "symbols_replaced": len(mapping)}

    # ------------------------------------------------------------------
    # Prediktion
    # ------------------------------------------------------------------

    def _predict_with_depth(self, history) -> tuple[dict[Any, float], int]:
        """Som predict_probabilities, men returnerar även vilken kontextlängd
        som matchade. -1 = ingenting matchade.

        Pensionerade symboler filtreras bort FÖRE normaliseringen, så massan
        fördelas bara på utfall som fortfarande är möjliga.
        """
        history = self._check_sequence(history, "history")

        for depth in range(min(len(history), self.max_depth), -1, -1):
            context = () if depth == 0 else tuple(history[-depth:])

            # .get() - annars skapar defaultdict:en tomma poster vid uppslag
            counts = self.context_tree.get(context)
            if not counts:
                continue

            if self.retired:
                counts = {k: v for k, v in counts.items() if k not in self.retired}

            total = sum(counts.values())
            if total > 0:
                return {sym: c / total for sym, c in counts.items()}, depth

        return {}, -1

    def _predict_interpolated(self, history) -> tuple[dict[Any, float], int]:
        """Väg samman ALLA kontextlängder i stället för att välja den längsta.

        Hård back-off tar den längsta kontext som setts och litar blint på
        den. Men den längsta kontexten är också den som setts FÄRRAST gånger.
        Har den bara observerats två gånger blir dess fördelning en punkt-
        skattning ur två stickprov, och modellen uttalar sig tvärsäkert på
        nästan ingenting. Mätbart: träffsäkerheten steg med djupet medan
        log-sannolikheten föll - modellen gissade rätt oftare men var
        systematiskt överkonfident.

        Interpolationen bygger i stället upp fördelningen nedifrån:

            P_d(x) = (n_d(x) + alpha · P_{d-1}(x)) / (n_d + alpha)

        Varje nivå justerar sin förälder i stället för att ersätta den. En
        kontext med gott om data dominerar sin förälder; en med lite data
        lämnar den nästan orörd. Vikten sköter sig själv med datamängden -
        ingen tröskel att ställa in.

        alpha = antalet DISTINKTA symboler i kontexten (Witten-Bell). Tanken:
        har en kontext redan visat många olika fortsättningar är den
        benägen att visa ännu en osedd, och mer massa ska lämnas åt
        föräldern.
        """
        active = self.active_vocab
        if not active:
            return {}, -1

        probs = {s: 1.0 / len(active) for s in active}   # djup -1: likformig
        used = -1

        for d in range(0, min(len(history), self.max_depth) + 1):
            context = () if d == 0 else tuple(history[-d:])
            counts = self.context_tree.get(context)
            if not counts:
                break   # kontexterna är nästlade: saknas denna, saknas längre
            if self.retired:
                counts = {k: v for k, v in counts.items() if k not in self.retired}
            total = sum(counts.values())
            if total <= 0:
                break

            alpha = max(1.0, float(len(counts)))
            denom = total + alpha
            probs = {s: (counts.get(s, 0.0) + alpha * probs.get(s, 0.0)) / denom
                     for s in set(probs) | set(counts)}
            used = d

        return probs, used

    def predict_probabilities(self, history) -> dict[Any, float] | dict[Any, Any]:
        history = self._check_sequence(history, "history")
        if self.interpolate:
            return self._predict_interpolated(history)[0]
        return self._predict_with_depth(history)[0]

    def generate_sequence(self, start_sequence, length_to_generate=10) -> list[Any]:
        """Genererar en LISTA av nästa symboler baserat på högsta sannolikhet."""
        generated = self._check_sequence(start_sequence, "start_sequence")

        for _ in range(length_to_generate):
            probs = self.predict_probabilities(history=generated)
            if not probs:
                break

            # Välj det absolut mest troliga elementet
            next_sym = max(probs, key=probs.get)
            generated.append(next_sym)

        return generated

    # ------------------------------------------------------------------
    # Scoring: hur väl passar en ny sekvens det tränade trädet?
    # ------------------------------------------------------------------

    def score_sequence(self, sequence, epsilon=0.01, top_surprises=5) -> dict[str, Any]:
        """Log-likelihood för en ny sekvens under den tränade modellen.

        För varje position beräknas P(symbol | historik) via back-off och
        blandas med en likformig fördelning:  p = (1-ε)·p_modell + ε/V,
        där V = antal levande symboler + 1. Det gör att p aldrig blir 0
        (log(0) = -inf) och att en osedd symbol får ett litet men ändligt
        värde.

        Steg delas upp i tre sorter, vilket är poängen med det dynamiska
        vokabuläret:
          KÄNT     symbolen finns i active_vocab
          NYTT     symbolen har aldrig setts - troligen ett antal_id som
                   skapats efter träningen. Inte avvikande, bara nytt.
          DÖTT     symbolen är pensionerad men dyker ändå upp i indatan.
                   Det är ett fel uppströms, inte en anomali i mönstret.

        avg_log_prob räknar med allt. avg_log_prob_known hoppar över NYTT
        och DÖTT och är det mått du vill jämföra över tid när vokabuläret
        rört på sig - annars mäter du bara hur många nya id:n som hunnit
        skapas sedan träningen.
        """
        sequence = self._check_sequence(sequence)
        n = len(sequence)
        if n == 0:
            return {"n_symbols": 0, "total_log_prob": 0.0, "avg_log_prob": 0.0,
                    "avg_log_prob_known": 0.0, "perplexity": float("nan"),
                    "avg_depth": 0.0, "n_novel": 0, "n_dead": 0,
                    "novel_symbols": [], "dead_symbols": [],
                    "surprises": [], "z_score": None}

        V = len(self.active_vocab) + 1
        total_log_prob = 0.0
        known_log_prob = 0.0
        n_known = 0
        depths: list[int] = []
        steps: list[dict] = []
        novel: set = set()
        dead: set = set()

        for i in range(n):
            history = sequence[:i]
            symbol = sequence[i]
            if self.interpolate:
                probs, depth = self._predict_interpolated(history)
            else:
                probs, depth = self._predict_with_depth(history)

            p_model = probs.get(symbol, 0.0)
            p = (1.0 - epsilon) * p_model + epsilon / V
            lp = math.log(p)
            total_log_prob += lp
            depths.append(max(depth, 0))

            if symbol in self.retired:
                kind = "dead"
                dead.add(symbol)
            elif symbol not in self.vocab:
                kind = "novel"
                novel.add(symbol)
            else:
                kind = "known"
                known_log_prob += lp
                n_known += 1

            steps.append({"index": i, "symbol": symbol, "prob": p,
                          "matched_depth": depth, "kind": kind})

        avg_log_prob = total_log_prob / n
        avg_known = known_log_prob / n_known if n_known else float("nan")

        # Bara KÄNDA steg är intressanta som överraskningar - ett nytt id är
        # osannolikt per definition och skulle annars alltid toppa listan.
        surprises = sorted((s for s in steps if s["kind"] == "known"),
                           key=lambda s: s["prob"])[:top_surprises]

        z_score = None
        if self._baseline_mean is not None and self._baseline_std:
            z_score = (avg_known - self._baseline_mean) / self._baseline_std

        return {
            "n_symbols": n,
            "total_log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "avg_log_prob_known": avg_known,
            "perplexity": math.exp(-avg_known) if n_known else float("nan"),
            "avg_depth": sum(depths) / n,
            "n_novel": sum(1 for s in steps if s["kind"] == "novel"),
            "n_dead": sum(1 for s in steps if s["kind"] == "dead"),
            "novel_symbols": sorted(novel),
            "dead_symbols": sorted(dead),
            "surprises": surprises,
            "z_score": z_score,
        }

    def calibrate(self, sequences, epsilon=0.01, min_std=0.0) -> dict[str, float]:
        """Ett log-score säger inget i sig - det är bara jämförbart.

        Kör helst på HÅLLEN-UT data som du vet är normal, inte på
        träningsdatan (då blir spridningen orealistiskt liten och z-värdena
        uppblåsta). Efter detta returnerar score_sequence() ett z_score
        baserat på avg_log_prob_known.

        min_std sätter ett golv på spridningen. Vid online-clustering går det
        sällan att hålla ut data, och med få nästan identiska sekvenser blir
        std nära noll - då exploderar z-värdena och tröskeln blir oanvändbar.
        Golvet gör skalan robust. Sätt det till hur stor variation i
        avg_log_prob du anser vara normalt brus.
        """
        scores = [self.score_sequence(s, epsilon=epsilon)["avg_log_prob_known"]
                  for s in sequences if len(s) > 0]
        scores = [s for s in scores if not math.isnan(s)]
        if not scores:
            raise ValueError("Inga användbara sekvenser att kalibrera på.")

        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        self._baseline_mean = mean
        self._baseline_std = max(math.sqrt(var), min_std)
        return {"mean": mean, "std": self._baseline_std, "n": len(scores)}


if __name__ == "__main__":
    # (värde_id, antal_id). Värde-idna är sorterade efter medelvärde,
    # antal_id 1 och 2 kommer från antalsmodellen.
    träning = [
        [(1, 1), (3, 2), (5, 1), (3, 2), (1, 1), (3, 2), (5, 1)],
        [(1, 1), (3, 2), (5, 1), (5, 2), (3, 2), (1, 1), (3, 2)],
        [(3, 2), (5, 1), (3, 2), (1, 1), (3, 2), (5, 1), (3, 2)],
        [(1, 1), (3, 2), (3, 2), (5, 1), (3, 2), (1, 1), (3, 2)],
    ]
    validering = [
        [(1, 1), (3, 2), (5, 1), (3, 2), (1, 1), (3, 2), (3, 2)],
        [(3, 2), (5, 1), (3, 2), (1, 1), (3, 2), (5, 1), (5, 2)],
    ]

    m = CustomVMM(max_depth=3, min_count=1)
    m.fit_many(träning)

    print("Kontexter:", len(m.context_tree))
    print("Vokabulär:", sorted(m.vocab))
    print("Rot:", {k: round(v, 1) for k, v in m.context_tree[()].items()})

    print(f"\nKalibrering (hållen ut): "
          f"{ {k: round(v, 3) for k, v in m.calibrate(validering).items()} }")

    normal = [(1, 1), (3, 2), (5, 1), (3, 2), (1, 1), (3, 2), (5, 1)]
    udda = [(5, 1), (5, 1), (1, 1), (1, 1), (5, 1), (3, 2), (5, 1)]

    for namn, seq in [("normal", normal), ("udda  ", udda)]:
        r = m.score_sequence(seq)
        print(f"\n{namn}: avg_known={r['avg_log_prob_known']:+.3f}  "
              f"ppl={r['perplexity']:.2f}  djup={r['avg_depth']:.2f}  "
              f"z={r['z_score']:+.2f}  nya={r['n_novel']}")
        if r["surprises"]:
            s = r["surprises"][0]
            print(f"  mest oväntat: index {s['index']} = {s['symbol']} "
                  f"(p={s['prob']:.4f}, kontextlängd {s['matched_depth']})")

    # --- BIC säger två komponenter: antal_id 2 delas i 7 och 8 -------------
    print("\n" + "=" * 62)
    print("SPLIT: antal_id 2 -> (7, 8) med proportionerna 0.7 / 0.3")
    info = m.split_count_component(2, [7, 8], weights=[0.7, 0.3])
    print(info)
    print("Vokabulär efter:", sorted(m.vocab))
    print("Rot efter:", {k: round(v, 2) for k, v in m.context_tree[()].items()})

    # Samma mönster, men med de nya idna - statistiken ska ha följt med
    efter_split = [(1, 1), (3, 7), (5, 1), (3, 7), (1, 1), (3, 7), (5, 1)]
    r = m.score_sequence(efter_split)
    print(f"\nsamma mönster, nya id: avg_known={r['avg_log_prob_known']:+.3f}  "
          f"ppl={r['perplexity']:.2f}  djup={r['avg_depth']:.2f}  "
          f"nya={r['n_novel']}  döda={r['n_dead']}")

    # Kontroll: skickar man in EN symbol istället för en sekvens?
    try:
        m.predict_probabilities((5, 1))
    except TypeError as e:
        print(f"\nIndatakontroll: {e}")