"""
Adaptiv gaussisk mixtur med online-födsel av komponenter.

Ersätter DummyFilter. För varje sampel:

  1. mät standardiserat avstånd till varje befintlig komponent
  2. ligger ALLA längre bort än birth_threshold sigma -> föd en ny komponent,
     seedad på sampelvärdet
  3. annars tilldela MAP-komponenten (ln vikt + ln täthet) och uppdatera den

Komponenter föds men dör aldrig och slås aldrig ihop, enligt beslutet. Ingen
hysteres: byter signalen komponent för ett enda sampel blir det en körning av
längd 1, och den får antalsmodellen ta hand om.

TVÅ UPPSÄTTNINGAR MOMENT PER KOMPONENT
Kolumnerna i param_vec skiljer på "mean, variance" och "raw sequential mean,
raw sequential variance". Min tolkning:

  mean/variance          exponentiellt glidande skattning med glömska (alpha).
                         Det är den som svarar mot "mu at the specific cluster
                         AT THIS MOMENT" i out - den följer med om nivån
                         driver under passets gång.
  raw sequential mean/   Welfords löpande exakta empiriska moment över ALLA
  variance               råa punkter komponenten någonsin tilldelats. Ingen
                         glömska, ingen drift.

Skillnaden mellan de två är i sig en signal: divergerar de har komponenten
drivit sedan den skapades. Säg till om du menade något annat med kolumnerna,
det är den enda punkten i kontraktet jag fick gissa på.

STATE SPARAS MELLAN ANROP
Komponenter som aldrig dör vore meningslösa om filtret nollställdes vid varje
filter()-anrop - då skulle värde-idna inte betyda samma sak mellan
observationer, och hela standardiseringen faller. Filtret är därför ETT
globalt, persistent objekt, vilket också är precis vad "värdemodellen antas
vara konstant" innebär i praktiken. reset() finns för tester.
"""

import math

import numpy as np


class _Component:
    """En gaussisk komponent med både adaptiva och råa moment."""

    __slots__ = ("mu", "var", "raw_mean", "raw_m2", "n", "first_value")

    def __init__(self, x, init_var) -> None:
        self.mu = float(x)          # adaptiv (glömska)
        self.var = float(init_var)
        self.raw_mean = float(x)    # Welford (exakt, all data)
        self.raw_m2 = 0.0
        self.n = 1
        self.first_value = float(x)  # värdet som initierade komponenten

    @property
    def raw_var(self) -> float:
        return self.raw_m2 / self.n if self.n > 0 else 0.0


class GaussianFilter:
    """Parametrar, i ungefärlig ordning efter hur mycket de betyder.

    init_std        Antagen standardavvikelse för en NY komponent. Detta är
                    huvudratten. För liten -> varje brusigt sampel föder en
                    egen komponent. För stor -> närliggande nivåer smälter
                    ihop till en. Sätt den till brusnivån i din signal.
    birth_threshold Hur många sigma från ALLA befintliga komponenter ett
                    sampel måste ligga för att föda en ny. 3.0 betyder att
                    ungefär 0.3 % av rent brus föder komponenter i onödan.
    alpha           Glömskefaktor för den adaptiva skattningen. 0.05 ~ ett
                    effektivt minne på 20 sampel. Högre = följer drift
                    snabbare men brusigare.
    prior_n         Pseudo-observationer av init_var som blandas in i
                    variansskattningen. Utan detta har en nyfödd komponent
                    (n=1, var=0) oändlig täthet i sin egen punkt och drar
                    till sig allt. Skattningen krymps mot init_var tills
                    komponenten sett tillräckligt.
    min_std         Golv på standardavvikelsen. Skydd mot varianskollaps på
                    konstanta signalavsnitt, vilket annars ger division med
                    noll och en oändlig ström av nya komponenter.
    """

    def __init__(self, init_std="auto", birth_threshold=3.0, alpha=0.05,
                 alpha_var=None, prior_n=5.0, min_std=1e-3,
                 min_support=5) -> None:
        self.init_var = None if init_std == "auto" else float(init_std) ** 2
        self.birth_threshold = float(birth_threshold)
        self.alpha = float(alpha)
        # Variansen får röra sig långsammare än medelvärdet. Den styr
        # beslutsgränserna, så den ska vara den tröga storheten.
        self.alpha_var = float(alpha / 4 if alpha_var is None else alpha_var)
        self.prior_n = float(prior_n)
        self.min_var = float(min_std) ** 2
        # En komponent är PRELIMINÄR tills den samlat min_support sampel. Den
        # lever vidare och kan växa, men den publiceras inte i state_seq -
        # sampel som hamnar i den rapporteras på närmaste publicerade
        # komponent i stället.
        #
        # Detta dödar ingenting; det avstår bara från att låta ett enstaka
        # utliggarsampel bli en egen nivå i symbolalfabetet. Utan det får man
        # komponenter med n=1 som aldrig försvinner, som ändå upptar en plats
        # i rangordningen, och som därmed förskjuter alla värde-id ovanför
        # sig. Sätt min_support=1 för att stänga av.
        self.min_support = int(min_support)
        self.components: list[_Component] = []

    def reset(self) -> None:
        self.components = []

    # ------------------------------------------------------------------

    @staticmethod
    def estimate_noise_std(x, max_lag=120) -> float:
        """Skattar brusnivån ur signalen själv: MAD på differensen vid det
        FÖRDELAKTIGASTE tidsavståndet.

        init_std var den enda parameter som verkligen krävde handpåläggning
        (se känslighetstestet i __main__), men den ska ju vara brusnivån, och
        den går att mäta.

        Grundtanken: hitta två tidpunkter som ligger i SAMMA komponent - då
        är deras differens rent brus, utan nivåskillnad. MAD i stället för
        standardavvikelse eftersom de par som ändå spänner över ett nivåbyte
        blir extremvärden.

        Varför inte bara lag 1: det förutsätter att signalen stannar kvar på
        en nivå några sampel i taget. Det stämmer för långa körningar, men
        inte för ett mönster som byter nivå VARJE steg - då mäter lag 1
        mönstrets spännvidd i stället för bruset. På en verklig
        staggersekvens gav lag 1 en skattning 18000 gånger för stor, vilket
        smälter samman alla nivåer till en enda komponent.

        Men ett periodiskt mönster med period n upprepar sig: x[i] och
        x[i+n] ligger i samma komponent. Alltså: pröva alla tidsavstånd och
        ta det minsta värdet. På staggersekvensen ovan hittade den lag 14 -
        exakt sekvenslängden - och återfann det sanna bruset.

        Skalfaktorn: MAD -> sigma är 1/0.6745, och differensen av två
        oberoende brusiga sampel har variansen 2·sigma², därav /sqrt(2).

        Notera: att minimera över många tidsavstånd ger en lätt UNDERskattning
        (minimum av flera brusiga skattningar). Det är den ofarligare
        riktningen - för litet ger några extra komponenter, som min_support
        håller preliminära, medan för stort smälter samman verkliga nivåer
        och den skadan går inte att reparera i efterhand.
        """
        x = np.asarray(x, dtype=float).ravel()
        if len(x) < 4:
            return 1.0

        skala = 1.0 / (0.6745 * math.sqrt(2.0))
        bästa = np.inf
        for lag in range(1, min(max_lag, len(x) // 2) + 1):
            mad = float(np.median(np.abs(x[lag:] - x[:-lag])))
            if mad < bästa:
                bästa = mad
        return max(bästa * skala, 1e-12)

    def calibrate_noise(self, data) -> float:
        """Sätt brusnivån explicit från en representativ signal.

        VIKTIGT när observationerna är korta. Skattaren måste kunna nå ett
        tidsavstånd som motsvarar mönstrets period för att hitta två sampel i
        samma komponent. Har mönstret period 14 och signalen bara 16 sampel
        finns det avståndet inte att pröva, och skattningen faller tillbaka
        på mönstrets spännvidd - alltså tusenfalt för stor, varpå alla nivåer
        smälter samman till en enda komponent och modellen ser ingenting.

        Ge den därför ihopslagna träningsdata (aldrig testdata - det vore
        läckage) innan första filter()-anropet. Skarvarna mellan hopslagna
        inspelningar stör inte en medianbaserad skattare.
        """
        self.init_var = self.estimate_noise_std(data) ** 2
        return math.sqrt(self.init_var)

    def _effective_var(self, c: _Component) -> float:
        """Variansen som används för beslut, krympt mot prior.

        En nyfödd komponent har ingen egen information om sin spridning.
        Blandningen (n·var + prior_n·init_var)/(n + prior_n) låter den ärva
        prior tills den samlat egen data.
        """
        v = (c.n * c.var + self.prior_n * self.init_var) / (c.n + self.prior_n)
        return max(v, self.min_var)

    def _update(self, c: _Component, x: float) -> None:
        # Adaptiv skattning med glömska -> "mu at this moment"
        d = x - c.mu

        # Variansen uppdateras med VINSORISERAT avstånd, och långsammare än
        # medelvärdet. Utan detta får man varianskollaps åt andra hållet:
        # ett enda gränsfallssampel från nästa nivå absorberas, variansen
        # skjuter i höjden, den vidgade komponenten sväljer resten av nivån,
        # variansen växer igen. Två nivåer smälter ihop av ett olyckligt
        # sampel. Att klippa avståndet vid birth_threshold sigma gör att ett
        # absorberat gränsfall aldrig kan flytta variansen mer än ett
        # normalt sampel skulle.
        sigma = math.sqrt(max(c.var, self.min_var))
        d_clipped = max(-self.birth_threshold * sigma,
                        min(self.birth_threshold * sigma, d))
        c.var = max(c.var + self.alpha_var * (d_clipped * d_clipped - c.var),
                    self.min_var)
        c.mu += self.alpha * d

        # Welford: exakta löpande moment över alla råa punkter, oklippta.
        # De ska vara just råa - klipper man här ljuger kolumnen.
        c.n += 1
        d1 = x - c.raw_mean
        c.raw_mean += d1 / c.n
        c.raw_m2 += d1 * (x - c.raw_mean)

    # ------------------------------------------------------------------

    def _published_index(self, k, x) -> int:
        """Preliminära komponenter rapporteras på närmaste publicerade.

        Komponenten själv behålls och fortsätter samla data - når den
        min_support publiceras den och dyker upp i symbolalfabetet av egen
        kraft. Fram till dess syns den inte i state_seq.
        """
        if self.components[k].n >= self.min_support:
            return k
        best, best_d = None, np.inf
        for j, c in enumerate(self.components):
            if j == k or c.n < self.min_support:
                continue
            d = abs(x - c.mu)
            if d < best_d:
                best, best_d = j, d
        return k if best is None else best

    def filter(self, data, update=True):
        """Returnerar (state_seq, param_vec, out) enligt kontraktet.

        update=False är INFERENSLÄGE: inga komponenter föds, inga parametrar
        uppdateras, sampel tilldelas närmaste publicerade komponent. Krävs
        vid utvärdering - annars lär sig filtret av testdatan medan den mäts,
        och mätningen är inte längre hållen ut.

        state_seq  (N, 2) int   kolumn 0 = komponentindex, 1 = rå körlängd
        param_vec  (K, 6) float mean, variance, raw sequential mean,
                                raw sequential variance, total number of
                                points, first value that initiated it
        out        (T, 4) float mu för vald komponent i det ögonblicket,
                                varians, vald komponent, negativ log-likelihood

        out[:, 0:2] sparas FÖRE uppdateringen - det är vad filtret trodde när
        beslutet fattades, inte facit i efterhand.
        """
        x = np.asarray(data, dtype=float).ravel()
        T = len(x)
        chosen = np.empty(T, dtype=int)
        out = np.empty((T, 4), dtype=float)

        # init_std="auto": mät brusnivån på den första serien vi ser och
        # behåll den. Den ska inte skattas om per observation - då skulle
        # beslutsgränserna, och därmed komponenternas betydelse, vandra
        # mellan observationer.
        if self.init_var is None:
            self.init_var = self.estimate_noise_std(x) ** 2

        for t in range(T):
            xt = float(x[t])

            if not self.components:
                if not update:
                    raise RuntimeError(
                        "Filtret är otränat - kan inte köras i inferensläge.")
                self.components.append(_Component(xt, self.init_var))
                k = 0
                mu_used, var_used = xt, self.init_var
            elif not update:
                # Inferensläge: närmaste publicerade komponent, ingen födsel,
                # ingen uppdatering.
                k, best_d = 0, np.inf
                for j, c in enumerate(self.components):
                    if c.n < self.min_support:
                        continue
                    d = abs(xt - c.mu)
                    if d < best_d:
                        k, best_d = j, d
                mu_used = self.components[k].mu
                var_used = self._effective_var(self.components[k])
            else:
                total_n = sum(c.n for c in self.components)
                best_k, best_lp = 0, -np.inf
                min_z = np.inf
                for j, c in enumerate(self.components):
                    v = self._effective_var(c)
                    z = abs(xt - c.mu) / math.sqrt(v)
                    min_z = min(min_z, z)
                    lp = (math.log(c.n / total_n)
                          - 0.5 * math.log(2 * math.pi * v)
                          - 0.5 * (xt - c.mu) ** 2 / v)
                    if lp > best_lp:
                        best_k, best_lp = j, lp

                if min_z > self.birth_threshold:
                    # Ingen befintlig komponent kan rimligen ha genererat
                    # detta sampel -> ny komponent.
                    self.components.append(_Component(xt, self.init_var))
                    k = len(self.components) - 1
                    mu_used, var_used = xt, self.init_var
                else:
                    k = best_k
                    c = self.components[k]
                    mu_used, var_used = c.mu, self._effective_var(c)
                    self._update(c, xt)

            nll = 0.5 * math.log(2 * math.pi * var_used) \
                + 0.5 * (xt - mu_used) ** 2 / var_used
            chosen[t] = self._published_index(k, xt)
            out[t] = (mu_used, var_used, k, nll)

        # Körlängdskodning av tilldelningssekvensen
        rle = []
        start = 0
        for i in range(1, T + 1):
            if i == T or chosen[i] != chosen[start]:
                rle.append((int(chosen[start]), i - start))
                start = i
        state_seq = np.asarray(rle, dtype=int).reshape(-1, 2)

        K = len(self.components)
        param_vec = np.zeros((K, 6), dtype=float)
        for j, c in enumerate(self.components):
            param_vec[j] = (c.mu, c.var, c.raw_mean, c.raw_var,
                            c.n, c.first_value)

        return state_seq, param_vec, out

    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        return [{"id": j,
                 "mu": round(c.mu, 3),
                 "std": round(math.sqrt(c.var), 3),
                 "raw_mu": round(c.raw_mean, 3),
                 "raw_std": round(math.sqrt(c.raw_var), 3),
                 "n": c.n,
                 "seed": round(c.first_value, 3)}
                for j, c in enumerate(self.components)]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    nivåer = [0.0, 1.0, 2.0, 3.0]
    signal = np.concatenate([rng.normal(nivåer[i % 4], 0.2, size=rng.poisson(8) + 2)
                             for i in range(40)])

    f = GaussianFilter()          # init_std="auto"
    state_seq, param_vec, out = f.filter(signal)

    print(f"{len(signal)} sampel -> {len(state_seq)} körningar, "
          f"{len(f.components)} komponenter")
    print(f"sanna nivåer: {nivåer}, sant brus 0.20, "
          f"skattat brus {math.sqrt(f.init_var):.3f}")
    for row in f.summary():
        flagga = "" if row["n"] >= f.min_support else "   [preliminär]"
        print("  ", row, flagga)
    print(f"medel-NLL: {out[:, 3].mean():.3f}")

    print("\nVARFÖR init_std='auto': komponenter funna vid fast init_std")
    print("(sant antal 4, sant brus 0.20, 5 frön per rad)")
    for s in ("auto", 0.1, 0.15, 0.2, 0.3, 0.5, 0.7):
        ks = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            sig = np.concatenate([r.normal(nivåer[i % 4], 0.2, size=r.poisson(8) + 2)
                                  for i in range(40)])
            g = GaussianFilter(init_std=s)
            g.filter(sig)
            ks.append(sum(1 for c in g.components if c.n >= g.min_support))
        print(f"   init_std={str(s):<6} -> {ks}")
