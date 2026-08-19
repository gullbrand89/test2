"""
Clustring av RLE-sekvenser med en VLMC per kluster.

Kedjan:

    rådata
      -> state_filter.filter()          segmentering (gaussiskt filter)
      -> _standardize_component_id()    filtrets lokala id -> kanoniska id
      -> [(värde_id, RÅ körlängd), ...] klusteroberoende mellanform
      -> _encode(klustrets antalsmodell)
      -> [(värde_id, antal_id), ...]    symbolerna CustomVMM äter
      -> _match() / _create / _update

Två saker är värda att lägga märke till i designen.

1. _preprocess returnerar RÅA körlängder, inte antal_id.
   Varje kluster har sin EGEN antalsmodell, så samma körlängd blir olika
   antal_id i olika kluster. Symbolerna kan därför inte skapas färdiga före
   matchningen - de måste kodas om per kandidatkluster. Mellanformen
   (värde_id, rå körlängd) är den enda representation som är jämförbar
   mellan kluster.

2. Rangordning och accept är två olika beslut med olika mått.
   RANGORDNING mellan kluster sker på total log-likelihood (symbolstruktur +
   varaktigheter) - det är den generativa modellvalsfrågan "vilket kluster
   förklarar datan bäst".
   ACCEPT ("är detta överhuvudtaget något av mina kluster, eller ett nytt?")
   sker på z-score, eftersom rå log-likelihood inte är jämförbar mellan
   kluster: ett kluster med lågentropiskt mönster ger alltid högre värden
   och skulle systematiskt dra till sig allt.
"""

import numpy as np

from count_model import CountModels, PoissonCountModel
from gaussian_filter import GaussianFilter
from vlmc import CustomVMM


class DummyFilter:
    """Platshållare för det gaussiska filtret.

    Gör det minsta som ger ett testbart flöde: kvantiserar signalen till
    närmaste av ett FAST antal nivåer och körlängdskodar resultatet. Fasta
    nivåer motsvarar antagandet i det här steget - värdekomponenterna är
    stabila, så rangordningen efter medelvärde är samma mellan observationer.

    Returnerar (state_seq, param_vec, out) enligt kontraktet:
      state_seq  (N, 2) int   kolumn 0 = lokalt komponentindex
                              kolumn 1 = rå körlängd
      param_vec  (K, 6) float medelvärde, varians, rå sekventiellt medelvärde,
                              rå sekventiell varians, antal punkter i
                              komponenten, första värdet som initierade den
      out        (T, 4) float mu för vald komponent, varians, vald komponent,
                              negativ log-likelihood
    """

    def __init__(self, levels=(0.0, 1.0, 2.0, 3.0), sigma=0.35) -> None:
        self.levels = np.asarray(levels, dtype=float)
        self.sigma = sigma

    def filter(self, data, update=True):
        # update ignoreras - dummyfiltret har inget tillstånd att uppdatera.
        x = np.asarray(data, dtype=float)
        comp = np.argmin(np.abs(x[:, None] - self.levels[None, :]), axis=1)

        # Körlängdskodning
        rle = []
        start = 0
        for i in range(1, len(comp) + 1):
            if i == len(comp) or comp[i] != comp[start]:
                rle.append((int(comp[start]), i - start))
                start = i
        state_seq = np.asarray(rle, dtype=int).reshape(-1, 2)

        # En rad per komponent - ALLTID alla K, även de som saknas i just
        # den här observationen, så att rangordningen inte förskjuts.
        param_vec = np.zeros((len(self.levels), 6), dtype=float)
        for k in range(len(self.levels)):
            xs = x[comp == k]
            first = xs[0] if len(xs) else np.nan
            param_vec[k] = [self.levels[k],
                            float(xs.var()) if len(xs) else 0.0,
                            float(xs.mean()) if len(xs) else self.levels[k],
                            float(xs.var()) if len(xs) else 0.0,
                            len(xs),
                            first]

        mu = self.levels[comp]
        nll = 0.5 * ((x - mu) / self.sigma) ** 2 + np.log(self.sigma)
        out = np.column_stack([mu, np.full(len(x), self.sigma ** 2), comp, nll])
        return state_seq, param_vec, out


class Cluster:
    """Samling av kluster. Ett kluster = ett träd + en antalsmodell.

    trees          kluster_id -> CustomVMM
    value_params   kluster_id -> param_vec (värdemodellens komponenter)
    counts_params  kluster_id -> PoissonCountModel
                   (.params är listan av (medelvärde, varians)-tupler)
    raw_counts     kluster_id -> antal_id -> råa körlängder
                   (samma objekt som modellens .raw, alias för insyn)
    """

    def __init__(self, max_depth=4, min_count=1, match_threshold=-3.0,
                 z_threshold=-3.0, calibrate_after=3, min_std=0.35,
                 cold_start_relax=0.0, min_obs_for_split=30,
                 per_value_counts=True, state_filter=None) -> None:
        self.trees: dict[int, CustomVMM] = {}
        self.value_params: dict[int, np.ndarray] = {}
        self.counts_params: dict[int, CountModels] = {}
        self.raw_counts: dict[int, dict] = {}
        self.state_filter = state_filter or DummyFilter()

        self.max_depth = max_depth
        self.min_count = min_count
        self.match_threshold = match_threshold
        self.z_threshold = z_threshold
        self.calibrate_after = calibrate_after
        self.min_std = min_std
        # Ett kluster med bara någon enstaka sekvens har ett glest träd och
        # underskattar sannolikheten för ny data - inte för att sekvensen är
        # främmande, utan för att kontexterna inte hunnit fyllas. Detta mjukar
        # upp tröskeln tills klustret sett `calibrate_after` sekvenser.
        # AV som standard: i testerna gjorde den mer skada än nytta genom att
        # suga in den första främmande sekvensen i det enda kluster som fanns.
        # Slå på den om du ser motsatt problem - kluster som splittras i onödan.
        self.cold_start_relax = cold_start_relax
        self.min_obs_for_split = min_obs_for_split
        # True = en Poissonfordelning per varde-komponent. Se CountModels.
        self.per_value_counts = per_value_counts

        # Råa (värde_id, körlängd)-sekvenser per kluster. Sparas RÅA, inte
        # kodade: efter en split är gamla antal_id pensionerade, och en
        # sparad kodad sekvens skulle bestå av döda symboler. Rå form kodas
        # om mot den aktuella modellen när den behövs.
        self._sequences: dict[int, list] = {}
        self._next_id = 0
        # Senaste avbildningen födelseindex -> kanoniskt id, för att upptäcka
        # när värdemodellen numrerar om sig.
        self._prev_canonical = None
        self._relabels = 0

    # ------------------------------------------------------------------
    # Förbehandling
    # ------------------------------------------------------------------

    def _preprocess(self, data, update=True):
        """rådata -> [(värde_id, rå körlängd), ...] (klusteroberoende)."""
        state_seq, param_vec, out = self.state_filter.filter(data, update=update)

        canonical = self._canonical_ids(param_vec)
        if update:
            self._apply_relabel(canonical)  # innan något scoras mot träden
            self._prev_canonical = canonical

        pairs = [(int(canonical[c]), int(n))
                 for c, n in zip(state_seq[:, 0], state_seq[:, 1].astype(int))]
        return pairs, param_vec, out

    @staticmethod
    def _canonical_ids(param_vec):
        """Komponentens födelseindex -> kanoniskt id (rang efter medelvärde)."""
        order = np.argsort(np.asarray(param_vec)[:, 0], kind="stable")
        birth_to_canonical = np.empty(len(order), dtype=int)
        birth_to_canonical[order] = np.arange(len(order))
        return birth_to_canonical

    def _standardize_component_id(self, comp_seq, param_vec):
        """Filtrets komponentindex -> kanoniskt id, för en sekvens."""
        canonical = self._canonical_ids(param_vec)
        return [int(canonical[c]) for c in comp_seq]

    def _apply_relabel(self, canonical) -> dict:
        """Har rangordningen ändrats sedan förra observationen?

        Med ett filter som föder komponenter online räcker det att EN ny
        komponent dyker upp med ett medelvärde som sorterar in mellan två
        befintliga: allt ovanför skjuts upp ett steg och varje symbol i varje
        träd pekar plötsligt på fel fysisk nivå. Inget fel kastas, inget id
        dör - modellen bara slutar tyst att betyda något.

        Detektionen bygger på att komponenternas FÖDELSEINDEX (radordningen i
        param_vec) är stabilt även när deras rang inte är det. Ändras
        avbildningen födelseindex -> rang flyttas alla träd och all sparad
        historik med, via CustomVMM.relabel_value_ids().
        """
        prev = self._prev_canonical
        if prev is None:
            return {}

        perm = {int(prev[j]): int(canonical[j])
                for j in range(min(len(prev), len(canonical)))
                if int(prev[j]) != int(canonical[j])}
        if not perm:
            return {}

        for tree in self.trees.values():
            tree.relabel_value_ids(perm)
        for cid, seqs in self._sequences.items():
            self._sequences[cid] = [[(perm.get(v, v), c) for v, c in seq]
                                    for seq in seqs]
        self._relabels += 1
        return perm

    @staticmethod
    def _pad_rows(arr, n_rows):
        """Fyll ut en (K, 6)-matris till n_rows rader med nollor."""
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= n_rows:
            return arr
        pad = np.zeros((n_rows - len(arr), arr.shape[1]), dtype=float)
        return np.vstack([arr, pad])

    @staticmethod
    def _encode(model, pairs):
        """(värde_id, rå körlängd) -> (värde_id, antal_id) för ETT kluster."""
        return [(v, model.assign(v, c)) for v, c in pairs]

    # ------------------------------------------------------------------
    # Huvudflöde
    # ------------------------------------------------------------------

    def process(self, data) -> dict:
        pairs, param_vec, _ = self._preprocess(data)
        if not pairs:
            return {"cluster_id": None, "action": "tom sekvens"}

        match, diagnostics = self._match(pairs)
        if match is None:
            cid = self._create_new_cluster(pairs, param_vec)
            action = "nytt kluster"
        else:
            cid = match
            self._update_cluster(cid, pairs, param_vec)
            action = "uppdaterat"

        return {"cluster_id": cid, "action": action,
                "n_symbols": len(pairs), "candidates": diagnostics}

    def _match(self, pairs):
        """Vilket kluster förklarar sekvensen bäst - och räcker det?"""
        diagnostics = []
        for cid, tree in self.trees.items():
            model = self.counts_params[cid]
            symbols = self._encode(model, pairs)

            r = tree.score_sequence(symbols)
            # Varaktigheternas egen passform, oberoende av symbolordningen.
            count_ll = float(np.mean([model.logpmf(v, c) for v, c in pairs]))
            structure_ll = r["avg_log_prob_known"]
            if np.isnan(structure_ll):
                continue

            diagnostics.append({
                "cluster_id": cid,
                "rank_score": structure_ll + count_ll,   # rangordning
                "structure": structure_ll,
                "counts": count_ll,
                "z": r["z_score"],                       # accept
                "avg_depth": r["avg_depth"],
                "n_novel": r["n_novel"],
            })

        if not diagnostics:
            return None, diagnostics

        diagnostics.sort(key=lambda d: d["rank_score"], reverse=True)

        # Accepttestet körs på ALLA kandidater, inte bara den bäst rankade.
        # Annars kan en sekvens hamna i eget kluster bara för att ett annat,
        # datarikare kluster råkade ranka högre och sedan föll på tröskeln -
        # medan dess rätta hem stod strax under i listan och aldrig prövades.
        #
        # Ett kluster accepterar om NÅGOT av kriterierna håller. En sekvens
        # som är sannolik i absoluta tal hör uppenbart hemma; en som är
        # osannolik men normal för klustrets egen skala hör också hemma -
        # klustret är bara brusigt. Medvetet konservativt: ett felaktigt nytt
        # kluster är svårare att ta tillbaka än en något generös matchning.
        for d in diagnostics:
            n_seq = len(self._sequences[d["cluster_id"]])
            relax = self.cold_start_relax if n_seq < self.calibrate_after else 0.0
            d["accepted"] = bool(
                d["structure"] >= self.match_threshold - relax
                or (d["z"] is not None and d["z"] >= self.z_threshold))

        for d in diagnostics:          # redan sorterad på rank_score
            if d["accepted"]:
                return d["cluster_id"], diagnostics
        return None, diagnostics

    # ------------------------------------------------------------------
    # Direkt träning och utvärdering (utan matchning)
    # ------------------------------------------------------------------

    def observe(self, data, cluster_id=0) -> int:
        """Träna ETT bestämt kluster på en observation. Ingen matchning.

        För när man vet att alla observationer kommer från samma signal och
        vill mäta hur modellen förbättras, inte hur den grupperar.
        """
        pairs, param_vec, _ = self._preprocess(data, update=True)
        if not pairs:
            return cluster_id
        if cluster_id not in self.trees:
            return self._create_new_cluster(pairs, param_vec, cluster_id)
        self._update_cluster(cluster_id, pairs, param_vec)
        return cluster_id

    def evaluate(self, data, cluster_id=0) -> dict:
        """Mät prediktionsförmåga på en observation modellen INTE tränats på.

        Filtret körs i inferensläge och ingenting uppdateras, så samma
        observation kan mätas om och om igen medan modellen växer.

        Två träffsäkerheter rapporteras, och skillnaden mellan dem är
        poängen:
          accuracy        exakt rätt symbol, alltså både nivå OCH
                          varaktighetsklass. Har ett OUNDVIKLIGT tak: hur
                          välträna modellen än är kan den inte veta vilken
                          Poisson-komponent nästa varaktighet råkar hamna i.
          value_accuracy  rätt nivå, oavsett varaktighet. Det är HÄR den
                          lärbara strukturen sitter - mönstret i vad som
                          följer på vad.
        Planar den första ut medan den andra fortsätter stiga har modellen
        lärt sig allt som går att lära; taket är brus, inte okunskap.
        """
        pairs, _, _ = self._preprocess(data, update=False)
        if cluster_id not in self.trees or not pairs:
            return {}

        model = self.counts_params[cluster_id]
        tree = self.trees[cluster_id]
        symbols = self._encode(model, pairs)
        r = tree.score_sequence(symbols)

        n_hit = n_val = n_pred = 0
        for i in range(1, len(symbols)):
            probs = tree.predict_probabilities(symbols[:i])
            if not probs:
                continue
            n_pred += 1
            if max(probs, key=probs.get) == symbols[i]:
                n_hit += 1
            # Marginalisera bort varaktigheten: summera sannolikheten per nivå
            per_value: dict = {}
            for (v, _c), p in probs.items():
                per_value[v] = per_value.get(v, 0.0) + p
            if max(per_value, key=per_value.get) == symbols[i][0]:
                n_val += 1

        return {
            "accuracy": n_hit / n_pred if n_pred else float("nan"),
            "value_accuracy": n_val / n_pred if n_pred else float("nan"),
            "avg_log_prob": r["avg_log_prob_known"],
            "perplexity": r["perplexity"],
            "avg_depth": r["avg_depth"],
            "n_predictions": n_pred,
            "n_symbols": len(symbols),
        }

    def _create_new_cluster(self, pairs, param_vec, cid=None) -> int:
        if cid is None:
            cid = self._next_id
        self._next_id = max(self._next_id, cid) + 1

        # Antalsmodellen först: den definierar symbolernas alfabet.
        model = CountModels(per_value=self.per_value_counts,
                            min_obs_for_split=self.min_obs_for_split)
        model.observe(pairs)
        model.check_split()   # inget träd än, så inget att remappa
        self.counts_params[cid] = model
        self.raw_counts[cid] = model.raw

        tree = CustomVMM(max_depth=self.max_depth, min_count=self.min_count)
        tree.fit(self._encode(model, pairs))
        self.trees[cid] = tree

        self.value_params[cid] = np.array(param_vec, copy=True)
        self._sequences[cid] = [pairs]
        return cid

    def _update_cluster(self, cid, pairs, param_vec) -> dict:
        """Uppdaterar antalsmodell, träd och värdeparametrar.

        ORDNINGEN är inte godtycklig:
          1. antalsmodellen tar emot de nya körlängderna
          2. BIC-testet körs - kan pensionera id och skapa nya
          3. trädet remappas för varje split, INNAN ny data kodas
          4. först nu kodas sekvensen, med de id som gäller efter splitten
        Kodar man före steg 3 skrivs symboler med id som just dött.
        """
        model = self.counts_params[cid]
        model.observe(pairs)
        splits = model.check_split()
        self._sequences[cid].append(pairs)

        if splits:
            # EXAKT ombyggnad i stället för proportionell remap.
            #
            # split_count_component() måste gissa hur gamla observationer
            # skulle ha kodats med de nya idna, och fördelar dem med
            # blandningsvikterna. Det späder ut räknarna, och en kontext som
            # innehåller den delade symbolen på k positioner expanderar till
            # 2^k kontexter - mätt blev det en fyrdubbling av trädet vid
            # tredje splitten, med mätbar försämring av prediktionen.
            #
            # Men vi har den RÅA historiken sparad, och råa körlängder är
            # oberoende av antalsmodellen. Alltså kan vi koda om allt med de
            # nya komponenterna och bygga trädet på nytt - exakt, utan
            # utspädning och utan expansion. Splittar är sällsynta, så
            # kostnaden är försumbar.
            self._rebuild_tree(cid)
        else:
            self.trees[cid].fit(self._encode(model, pairs))

        # Värdeparametrarna: löpande medelvärde viktat på antal observationer.
        # Ett filter som föder komponenter online kan ha FLER komponenter nu
        # än när klustret skapades, så raderna måste fyllas på först.
        prev = self._pad_rows(self.value_params[cid], len(param_vec))
        param_vec = self._pad_rows(np.asarray(param_vec), len(prev))
        n_prev, n_new = prev[:, 4], param_vec[:, 4]
        total = np.where(n_prev + n_new > 0, n_prev + n_new, 1.0)
        blended = (prev * n_prev[:, None] + param_vec * n_new[:, None]) / total[:, None]
        blended[:, 4] = n_prev + n_new
        self.value_params[cid] = blended

        self.raw_counts[cid] = model.raw     # ogonblicksbild for insyn
        if len(self._sequences[cid]) >= self.calibrate_after:
            self._calibrate(cid)
        return {"splits": splits}

    def _rebuild_tree(self, cid) -> None:
        """Bygg om trädet från den råa historiken med aktuell antalsmodell."""
        model = self.counts_params[cid]
        tree = CustomVMM(max_depth=self.max_depth, min_count=self.min_count)
        for pairs in self._sequences[cid]:
            tree.fit(self._encode(model, pairs))
        self.trees[cid] = tree

    def _calibrate(self, cid) -> None:
        """Sätter klustrets z-skala.

        Sekvenserna kodas om mot den AKTUELLA antalsmodellen, så att gamla
        pensionerade id inte smyger in i kalibreringen.

        Caveat: vid online-clustering finns ingen hållen-ut data - vi
        kalibrerar på samma sekvenser trädet tränats på. Spridningen blir
        därför optimistiskt liten, och det är därför min_std finns.
        """
        model = self.counts_params[cid]
        seqs = [self._encode(model, p) for p in self._sequences[cid]]
        try:
            self.trees[cid].calibrate(seqs, min_std=self.min_std)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Diagnostik
    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        rows = []
        for cid, tree in self.trees.items():
            model = self.counts_params[cid]
            rows.append({
                "kluster": cid,
                "sekvenser": len(self._sequences[cid]),
                "kontexter": len(tree.context_tree),
                "symboler": len(tree.active_vocab),
                "antalskomponenter": model.summary(),
                "pensionerade": sorted(model.retired_ids),
            })
        return rows


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

def _make_series(pattern, durations, rng, sigma=0.2):
    """Bygg en signal genom att hålla varje nivå i ett antal steg."""
    out = []
    for level, d in zip(pattern, durations):
        out.extend(rng.normal(level, sigma, size=int(max(d, 1))))
    return np.asarray(out)


def _profile_a(rng, n_blocks=26):
    """Bimodala varaktigheter: korta pass blandat med långa.

    En Poisson räcker inte här - variansen är för stor för medelvärdet.
    Det är detta BIC-testet ska upptäcka och dela.
    """
    pattern, durations = [], []
    for i in range(n_blocks):
        pattern.append([0.0, 1.0, 2.0, 1.0][i % 4])
        lam = 3 if rng.random() < 0.6 else 14
        durations.append(rng.poisson(lam) + 1)
    return _make_series(pattern, durations, rng)


def _profile_b(rng, n_blocks=26):
    """Annat mönster, enhetliga varaktigheter."""
    pattern, durations = [], []
    for i in range(n_blocks):
        pattern.append([0.0, 3.0, 0.0, 2.0][i % 4])
        durations.append(rng.poisson(8) + 1)
    return _make_series(pattern, durations, rng)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    cs = Cluster(max_depth=3, min_count=1, match_threshold=-2.5,
                 z_threshold=-3.0, calibrate_after=3, min_obs_for_split=30,
                 state_filter=GaussianFilter())

    print("OBSERVATIONER")
    print("-" * 68)
    sanning = []
    for i in range(12):
        vilken = "A" if i % 2 == 0 else "B"
        data = _profile_a(rng) if vilken == "A" else _profile_b(rng)
        sanning.append(vilken)
        r = cs.process(data)
        bästa = r["candidates"][0] if r["candidates"] else None
        extra = ""
        if bästa:
            z = f"{bästa['z']:+.2f}" if bästa["z"] is not None else "  -  "
            extra = (f"  bäst={bästa['cluster_id']} rank={bästa['rank_score']:+.2f}"
                     f" struktur={bästa['structure']:+.2f} z={z}")
        print(f"{i:>2} profil {vilken}  -> kluster {r['cluster_id']} "
              f"({r['action']}, {r['n_symbols']} symboler){extra}")

    print("\nKLUSTER")
    print("-" * 68)
    for row in cs.summary():
        print(f"kluster {row['kluster']}: {row['sekvenser']} sekvenser, "
              f"{row['kontexter']} kontexter, {row['symboler']} levande symboler")
        print(f"  antalskomponenter: {row['antalskomponenter']}")
        print(f"  pensionerade id:   {row['pensionerade']}")

    print("\nFILTRETS KOMPONENTER (sanna nivåer: profil A 0/1/2, profil B 0/2/3)")
    print("-" * 68)
    print(f"skattad brusnivå: {np.sqrt(cs.state_filter.init_var):.3f}  (sant 0.20)")
    for row in cs.state_filter.summary():
        flagga = "" if row["n"] >= cs.state_filter.min_support else "   [preliminär]"
        print(f"   {row}{flagga}")
    print(f"ompermuteringar av värde-id under körningen: {cs._relabels}")

    print("\nROBUSTHET: samma försök över 10 frön")
    print("-" * 68)
    print("Renhet = summera, per kluster, den vanligaste profilen i det.")
    print("Understiger den 12 har matchningen antingen splittrat en profil på")
    print("flera kluster eller blandat två profiler i ett - tröskeln är")
    print("systemets svagaste del.")
    from collections import Counter
    for namn, gör_filter in (("DummyFilter   ", lambda: DummyFilter()),
                             ("GaussianFilter", lambda: GaussianFilter())):
        träffar, renheter, relabels = 0, [], 0
        for seed in range(10):
            r2 = np.random.default_rng(seed)
            cs2 = Cluster(max_depth=3, min_count=1, match_threshold=-2.5,
                          z_threshold=-3.0, calibrate_after=3,
                          min_obs_for_split=30, state_filter=gör_filter())
            t2, a2 = [], []
            for i in range(12):
                w = "A" if i % 2 == 0 else "B"
                t2.append(w)
                a2.append(cs2.process(_profile_a(r2) if w == "A" else _profile_b(r2))["cluster_id"])
            # Standardrenhet: summera per KLUSTER den vanligaste profilen i
            # det. Räknar man i stället per profil får man 12/12 även när
            # allt hamnat i ETT kluster, eftersom varje profil då är
            # "konsekvent" placerad - ett mått som belönar hopslagning.
            per_kluster = {}
            for t, a in zip(t2, a2):
                per_kluster.setdefault(a, Counter())[t] += 1
            renhet = sum(c.most_common(1)[0][1] for c in per_kluster.values())
            renheter.append(renhet)
            relabels += cs2._relabels
            träffar += (renhet == 12)
        print(f"  {namn}: {träffar}/10 perfekta, median renhet "
              f"{sorted(renheter)[5]}/12, {relabels} ompermuteringar")

    print("\nKONTROLL: matchar profilerna sina egna kluster?")
    print("-" * 68)
    tilldelning = {}
    for i in range(12):
        vilken = sanning[i]
        tilldelning.setdefault(vilken, []).append(i)
    for vilken in ("A", "B"):
        data = _profile_a(rng) if vilken == "A" else _profile_b(rng)
        pairs, _, _ = cs._preprocess(data)
        _, diag = cs._match(pairs)
        rad = "  ".join(f"k{d['cluster_id']}: rank={d['rank_score']:+.2f}"
                        f" (struktur {d['structure']:+.2f}, antal {d['counts']:+.2f})"
                        for d in diag)
        print(f"ny profil {vilken}: {rad}")
