"""
Antalsmodellen: en Poisson-mixtur som växer via BIC.

En körlängd från RLE:n är ett positivt heltal. Modellen börjar med EN
Poisson-komponent och kontrollerar mellan observationer om två komponenter
förklarar datan bättre - mätt med BIC. Gör de det delas komponenten, det
gamla idt pensioneras permanent och två nya tar över.

Det är precis den händelse CustomVMM.split_count_component() finns för:
när ett id dör måste trädets statistik följa med till de nya, annars
nollställs modellen vid varje split.

BIC = k·ln(n) - 2·loglik. Lägre är bättre. En Poisson har k=1 (lambda),
en tvåkomponentsmixtur har k=3 (lambda1, lambda2, blandningsvikt).
Straffet k·ln(n) är det som hindrar modellen från att dela i all oändlighet:
med lite data måste förbättringen vara stor för att bära tre parametrar.
"""

import math

import numpy as np


def _log_factorial(x) -> np.ndarray:
    """ln(x!) via lgamma. Behövs för äkta log-likelihood.

    (Termen är gemensam för båda modellerna och tar ut sig i BIC-jämförelsen,
    men vi tar med den så att loglik-värdena går att tolka på egen hand.)
    """
    arr = np.ravel(np.asarray(x, dtype=float))
    return np.array([math.lgamma(v + 1.0) for v in arr]).reshape(np.shape(x))


def poisson_logpmf(x, lam) -> np.ndarray:
    """ln P(x | Poisson(lam)), i logrummet för att undvika underflow."""
    lam = max(float(lam), 1e-12)
    x = np.asarray(x, dtype=float)
    return x * math.log(lam) - lam - _log_factorial(x)


def fit_single_poisson(x) -> tuple[float, float]:
    """ML-anpassning av en Poisson. Returnerar (lambda, loglik)."""
    x = np.asarray(x, dtype=float)
    lam = max(float(x.mean()), 1e-6)
    return lam, float(poisson_logpmf(x, lam).sum())


def fit_two_poisson(x, n_iter=300, tol=1e-9) -> tuple[float, float, float, float, np.ndarray]:
    """EM-anpassning av en tvåkomponents Poisson-mixtur.

    Returnerar (pi, lam1, lam2, loglik, responsibilities) där pi är vikten
    för komponent 1 och responsibilities[i] = P(komponent 1 | x_i).

    Startvärden tas från medianuppdelningen - det är en robust init som
    lägger de två komponenterna på var sin sida av datan istället för att
    starta dem ovanpå varandra (vilket EM inte tar sig ur).
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    lo, hi = x[x <= med], x[x > med]
    if len(lo) == 0 or len(hi) == 0:
        lam1, lam2 = max(x.mean() * 0.5, 1e-3), max(x.mean() * 1.5, 1e-3)
    else:
        lam1, lam2 = max(lo.mean(), 1e-3), max(hi.mean(), 1e-3)
    if abs(lam1 - lam2) < 1e-6:
        lam2 = lam1 + 1e-3

    pi = 0.5
    loglik = -np.inf
    resp = np.full(len(x), 0.5)

    for _ in range(n_iter):
        # E-steg (log-sum-exp för stabilitet)
        l1 = math.log(pi) + poisson_logpmf(x, lam1)
        l2 = math.log(1.0 - pi) + poisson_logpmf(x, lam2)
        m = np.maximum(l1, l2)
        denom = m + np.log(np.exp(l1 - m) + np.exp(l2 - m))
        new_loglik = float(denom.sum())
        resp = np.exp(l1 - denom)

        # M-steg
        s = resp.sum()
        pi = min(max(s / len(x), 1e-6), 1.0 - 1e-6)
        lam1 = max(float((resp * x).sum() / max(s, 1e-12)), 1e-6)
        lam2 = max(float(((1 - resp) * x).sum() / max(len(x) - s, 1e-12)), 1e-6)

        if new_loglik - loglik < tol:
            loglik = new_loglik
            break
        loglik = new_loglik

    return pi, lam1, lam2, loglik, resp


class PoissonCountModel:
    """Håller komponenterna som en RÅ körlängd översätts till antal_id via.

    params   lista av (medelvärde, varians), index = antal_id. Växer monotont;
             ett pensionerat id ligger kvar som gravsten så att indexen aldrig
             flyttar på sig. `active` säger vilka som fortfarande lever.
             (För en ren Poisson är varians = medelvärde. Vi lagrar den
             EMPIRISKA variansen för den tilldelade datan i stället - avviker
             den kraftigt uppåt är det överdispersion, alltså just det som
             BIC-testet är till för att fånga.)
    active   mängden levande antal_id
    weights  blandningsvikt per levande id
    raw      antal_id -> lista av råa körlängder som tilldelats komponenten.
             Detta är datan BIC-testet körs på.
    """

    def __init__(self, min_obs_for_split=30, bic_margin=0.0, max_components=12) -> None:
        self.params: list[tuple[float, float]] = []
        self.active: set[int] = set()
        self.weights: dict[int, float] = {}
        self.raw: dict[int, list[int]] = {}
        self.min_obs_for_split = min_obs_for_split
        self.bic_margin = bic_margin
        self.max_components = max_components

    # -- uppslagning ---------------------------------------------------

    def assign(self, raw_count) -> int:
        """Vilken komponent förklarar den här körlängden bäst?

        argmax över ln(vikt) + ln P(x | komponent) - alltså MAP-tilldelning,
        inte bara närmaste medelvärde. Vikten spelar roll: en sällsynt
        komponent ska inte vinna på marginalen.
        """
        if not self.active:
            raise RuntimeError("Modellen är otränad - kör observe() först.")
        best_cid, best_lp = None, -np.inf
        for cid in self.active:
            lam = self.params[cid][0]
            lp = math.log(max(self.weights.get(cid, 1e-9), 1e-12)) \
                + float(poisson_logpmf(raw_count, lam))
            if lp > best_lp:
                best_cid, best_lp = cid, lp
        return best_cid

    def logpmf(self, raw_count, cid=None) -> float:
        """ln P(körlängd | komponent). Utan cid används den bäst passande.

        Detta är ett mått vid sidan om VMM:ens: hur väl förklaras VARAKTIGHETERNA
        av det här klustrets antalsmodell, oberoende av symbolordningen.
        """
        if cid is None:
            cid = self.assign(raw_count)
        return float(poisson_logpmf(raw_count, self.params[cid][0]))

    # -- inlärning -----------------------------------------------------

    def observe(self, raw_counts) -> None:
        """Ta emot nya råa körlängder och uppdatera komponenterna.

        Första anropet skapar komponent 0 ur all data. Därefter hård
        tilldelning (classification-EM): varje observation går till sin
        MAP-komponent och lambda skattas om som medelvärdet av det som
        tilldelats. Enkelt, stabilt och tillräckligt - den mjuka EM:en
        används där den behövs, i splittestet.
        """
        raw_counts = [int(c) for c in raw_counts]
        if not raw_counts:
            return

        if not self.active:
            self.params.append((0.0, 0.0))
            self.active.add(0)
            self.raw[0] = list(raw_counts)
            self.weights[0] = 1.0
            self._refit(0)
            return

        for c in raw_counts:
            self.raw[self.assign(c)].append(c)
        for cid in self.active:
            self._refit(cid)
        self._renormalize()

    def _refit(self, cid) -> None:
        xs = self.raw.get(cid, [])
        if not xs:
            return
        arr = np.asarray(xs, dtype=float)
        self.params[cid] = (float(arr.mean()), float(arr.var()))

    def _renormalize(self) -> None:
        total = sum(len(self.raw.get(c, [])) for c in self.active)
        if total == 0:
            return
        for cid in self.active:
            self.weights[cid] = len(self.raw.get(cid, [])) / total

    # -- BIC-testet ----------------------------------------------------

    def check_split(self) -> list[dict]:
        """Testa varje levande komponent: passar 1 eller 2 Poisson bäst?

        Returnerar en lista av splittar som utförts. Varje post ser ut som
            {"old": 2, "new": [7, 8], "weights": [0.7, 0.3], ...}
        och ska skickas rakt in i CustomVMM.split_count_component().

        Bara komponenter som fanns vid anropets början testas, så en nyss
        skapad komponent kan inte delas igen i samma pass.
        """
        splits = []
        for cid in sorted(self.active):
            if len(self.params) + 1 > self.max_components:
                break
            xs = self.raw.get(cid, [])
            n = len(xs)
            if n < self.min_obs_for_split:
                continue
            arr = np.asarray(xs, dtype=float)
            if arr.max() == arr.min():
                continue  # ingenting att dela

            lam, ll1 = fit_single_poisson(arr)
            pi, lam1, lam2, ll2, resp = fit_two_poisson(arr)

            bic1 = 1 * math.log(n) - 2 * ll1
            bic2 = 3 * math.log(n) - 2 * ll2
            if bic2 + self.bic_margin >= bic1:
                continue  # en komponent räcker

            # Hård uppdelning av den gamla komponentens data
            mask = resp >= 0.5
            xs1 = [int(v) for v in arr[mask]]
            xs2 = [int(v) for v in arr[~mask]]
            if not xs1 or not xs2:
                continue  # degenererad split, hoppa över

            new1, new2 = len(self.params), len(self.params) + 1
            self.params.append((0.0, 0.0))
            self.params.append((0.0, 0.0))
            self.raw[new1], self.raw[new2] = xs1, xs2
            del self.raw[cid]

            old_w = self.weights.pop(cid, 1.0)
            self.active.discard(cid)          # gravsten: idt återkommer aldrig
            self.active.update((new1, new2))
            self.weights[new1] = old_w * pi
            self.weights[new2] = old_w * (1.0 - pi)
            self._refit(new1)
            self._refit(new2)

            splits.append({
                "old": cid,
                "new": [new1, new2],
                "weights": [pi, 1.0 - pi],
                "lambdas": [lam1, lam2],
                "bic_1": bic1,
                "bic_2": bic2,
                "n": n,
            })

        if splits:
            self._renormalize()
        return splits

    # -- diagnostik ----------------------------------------------------

    def summary(self) -> list[dict]:
        return [
            {"id": cid,
             "mean": round(self.params[cid][0], 2),
             "var": round(self.params[cid][1], 2),
             "weight": round(self.weights.get(cid, 0.0), 3),
             "n": len(self.raw.get(cid, []))}
            for cid in sorted(self.active)
        ]

    @property
    def retired_ids(self) -> set:
        return set(range(len(self.params))) - self.active


class CountModels:
    """Antalsmodell(er) for ett kluster - en delad, eller en per varde_id.

    Fragan det handlar om: ska varje varde-komponent ha sin EGEN
    Poissonfordelning over korlangder, eller ska alla dela pa en?

    per_value=False (delad)
        En enda mixtur over alla korlangder oavsett niva. Poolar data, sa
        varje komponent far mer att gora skattningen pa. Men fordelningen
        som modelleras ar da en BLANDNING av olika nivaers varaktigheter,
        och den ar overdispergerad av konstruktion - vilket BIC svarar pa
        genom att dela, om och om igen. Splittarna beskriver da skillnaden
        MELLAN nivaer, inte spridningen inom nagon av dem.

    per_value=True (en per niva)
        Varje varde-komponent far sin egen fordelning. For dwell & switch
        ar det den generativa sanningen: varje dwell har sin egen langd.
        Betingat pa nivan forsvinner den mellan-niva-spridning som drev
        splittarna, och varje modell behover ofta bara en komponent.
        Priset ar att datan delas upp per niva.

    En niva som aldrig setts under traningen har ingen egen modell. Da
    anvands en poolad reservmodell som alltid tar emot allt.
    """

    def __init__(self, per_value=True, min_obs_for_split=30, bic_margin=0.0,
                 max_components=12) -> None:
        self.per_value = per_value
        self._kw = dict(min_obs_for_split=min_obs_for_split,
                        bic_margin=bic_margin, max_components=max_components)
        self.models: dict = {}                       # varde_id -> PoissonCountModel
        self.pooled = PoissonCountModel(**self._kw)  # reserv for osedda nivaer

    # -- uppslag -------------------------------------------------------

    def _model_for(self, v):
        if not self.per_value:
            return self.pooled
        m = self.models.get(v)
        return m if m is not None and m.active else self.pooled

    def assign(self, v, raw_count) -> int:
        return self._model_for(v).assign(raw_count)

    def logpmf(self, v, raw_count) -> float:
        return self._model_for(v).logpmf(raw_count)

    # -- inlarning -----------------------------------------------------

    def observe(self, pairs) -> None:
        """pairs: [(varde_id, ra korlangd), ...]"""
        self.pooled.observe([c for _, c in pairs])
        if not self.per_value:
            return
        per: dict = {}
        for v, c in pairs:
            per.setdefault(v, []).append(c)
        for v, cs in per.items():
            if v not in self.models:
                self.models[v] = PoissonCountModel(**self._kw)
            self.models[v].observe(cs)

    def check_split(self) -> list[dict]:
        """Returnerar splittar, varje post markt med vilken niva den gallde."""
        ut = []
        if self.per_value:
            for v, m in self.models.items():
                for s in m.check_split():
                    s["value_id"] = v
                    ut.append(s)
            self.pooled.check_split()
        else:
            for s in self.pooled.check_split():
                s["value_id"] = None
                ut.append(s)
        return ut

    # -- diagnostik ----------------------------------------------------

    @property
    def active(self) -> set:
        if not self.per_value:
            return set(self.pooled.active)
        return {(v, c) for v, m in self.models.items() for c in m.active}

    @property
    def retired_ids(self) -> set:
        if not self.per_value:
            return set(self.pooled.retired_ids)
        return {(v, c) for v, m in self.models.items() for c in m.retired_ids}

    @property
    def raw(self) -> dict:
        if not self.per_value:
            return self.pooled.raw
        return {(v, c): xs for v, m in self.models.items()
                for c, xs in m.raw.items()}

    def summary(self) -> list[dict]:
        if not self.per_value:
            return self.pooled.summary()
        ut = []
        for v in sorted(self.models):
            for r in self.models[v].summary():
                ut.append({"value_id": v, **r})
        return ut
