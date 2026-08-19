# Koppla in ditt eget filter och din egen data

Ditt filter följer redan kontraktet, så det droppas rakt in. Ingen adapter
behövs.

```python
from pipeline import check_filter_contract, SignalModel

# 1. Kontrollera att kontraktet verkligen hålls (gör detta först)
check_filter_contract(MittFilter(), en_representativ_signal)

# 2. Bygg modellen
modell = SignalModel(state_filter=MittFilter(), max_depth=6)

# 3. Sätt brusnivån, träna, använd
modell.calibrate_noise(np.concatenate(träning[:8]))
modell.observe_many(träning)

modell.evaluate(hållen_ut_signal)   # mät
modell.predict_next(pågående)       # använd
modell.score(ny_signal)             # hur väl passar den?
```

`SignalModel` är en tunn inpackning av `Cluster` med kluster-id 0 — ingen
matchning, ingen gruppering. Behöver du gruppering av flera olika signaler
senare finns `Cluster` kvar oförändrad i `main.py`.

---

## Kör kontraktskontrollen först

Eftersom jag inte kan se ditt filter är `check_filter_contract()` din enda
automatiska kontroll av att sömmen sitter rätt. Den kontrollerar formen på
de tre returvärdena, och fyra saker som inte syns i formen men som modellen
är helt beroende av:

| Kontroll | Varför |
|---|---|
| Körlängderna summerar till antalet sampel | Annars har segmenteringen tappat eller dubblerat data, och varje varaktighet modellen lär sig blir fel |
| `update=False` ändrar ingenting | Utan inferensläge lär sig filtret av testdatan medan den mäts, och all hållen-ut-utvärdering är meningslös |
| Radordningen i `param_vec` är stabil mellan anrop | Modellen använder **radindex** som komponentens identitet. Sorterar du om raderna kan den inte skilja "ny komponent" från "allt bytte plats" |
| Antalet komponenter minskar aldrig | En komponent som försvinner tar med sig sin betydelse och lämnar trädet pekande på fel nivåer |

Den sista punkten är värd att stanna vid. Du svarade att dina id är
**rangordnade efter medelvärde**. Det betyder att en ny komponent som föds
mitt i skalan förskjuter alla id ovanför sig, och varje symbol trädet lärt
sig pekar plötsligt på fel fysisk nivå. Inget kraschar, inget id dör —
modellen slutar bara tyst att betyda något. I mätningarna förstörde det 7
körningar av 10.

Skyddet finns och är på: `Cluster._apply_relabel` upptäcker att avbildningen
radindex → rang har ändrats och flyttar hela trädet och all sparad historik
med. **Men det bygger på att radindex är stabilt.** Det är den enda
egenskapen hos ditt filter som hela konstruktionen vilar på, och den som
kontrollen inte kan verifiera fullt ut från utsidan. Kontrollera den för
hand: raden för en komponent ska ligga kvar på samma index genom hela
körningen, även när nya komponenter tillkommer.

---

## Fem fällor, med de uppmätta konsekvenserna

**1. Brusnivån är den enskilt viktigaste inställningen.** Sätts den för högt
smälter verkliga nivåer samman till en komponent, och den skadan går inte
att reparera i efterhand. För lågt ger några extra komponenter, vilket är
ofarligare. Har ditt filter en egen brusparameter: se till att den blir
satt, och sätt den hellre för lågt än för högt.

Skattaren i `gaussian_filter.py` letar upp det *tidsavstånd* som ger minsta
mediandifferens i stället för att anta att intilliggande sampel ligger i
samma komponent. Det spelar roll när mönstret byter nivå varje steg: på en
verklig staggersekvens gav antagandet om intilliggande sampel en skattning
**18 000 gånger för stor**. Skattaren behöver dock kunna nå ett tidsavstånd
som motsvarar mönstrets period — ge den därför ihopslagen träningsdata via
`calibrate_noise()`, inte en enskild kort snutt.

**2. Utvärdera aldrig utan `update=False`.** Ett filter som uppdaterar sig
på testdatan gör mätningen värdelös. `SignalModel.evaluate()` sköter det,
men bygger du egna mätningar: gå via `_preprocess(..., update=False)`.

**3. Långa observationer döljer inlärningen.** Innehåller en enda
observation tiotals varv av mönstret är uppgiften löst efter den första,
och kurvan blir platt — inte för att modellen är dålig utan för att
experimentet inte lämnar något kvar att lära. Håll träningsobservationerna
korta nog att en ensam inte räcker. Testobservationerna kan vara långa,
det ger stabilare mätvärden.

**4. Poisson tål inte överdispersion.** En Poissonmixtur kan approximera
vilken räknefördelning som helst med tillräckligt många komponenter, så på
överdispergerad data fortsätter BIC dela så länge data växer. På en
degenererad signal med varians/medelvärde 50,7 delade den sju gånger i rad
och prediktionen blev stadigt sämre. Håll ett öga på antalet komponenter i
`modell.summary()`. Växer det monotont är det den här fällan, och rätt
lösning är en negativ binomialfördelning — en extra parameter som hanterar
överdispersion direkt.

**5. Mät med rätt mått.** Träffsäkerhet tittar bara på om toppgissningen
blev rätt; perplexitet väger hela fördelningen. Skillnaden avslöjade en
överkonfident modell som gissade rätt allt oftare medan den blev sämre
kalibrerad. Rapportera båda. `evaluate()` ger dig `accuracy` (nivå **och**
varaktighetsklass), `value_accuracy` (bara nivå) och `perplexity`.

---

## Vad som är värt att ställa in

| Parameter | Standard | Kommentar |
|---|---|---|
| `max_depth` | 6 | Hur långt bakåt kontexten sträcker sig. Ska överstiga mönstrets period om du vill fånga den helt |
| `per_value_counts` | `True` | En varaktighetsfördelning per nivå i stället för en delad. Gav symbolträff 0,67 → 0,78 och perplexitet 5,20 → 3,15 på riktig data |
| `min_obs_for_split` | 30 | Hur mycket data en komponent behöver innan BIC får dela den |
| `interpolate` | `True` | Väger samman alla kontextlängder i stället för att lita blint på den längsta. Perplexitet 3,3 → 2,3 utan kostnad i träffsäkerhet |

---

## Filerna

| Fil | Roll |
|---|---|
| `pipeline.py` | **Börja här.** Fasad + kontraktskontroll |
| `vlmc.py` | Trädet: kontexter, interpolerad back-off, scoring, omnumrering |
| `count_model.py` | Poisson + BIC. `CountModels` väljer delad eller per nivå |
| `main.py` | `Cluster` (flera signaler, med matchning) och `DummyFilter` |
| `gaussian_filter.py` | Referensfilter — ersätts av ditt, men brusskattaren är värd att låna |
| `learning_curve.py` | Inlärningskurva på syntetisk data |
| `emitter_curve.py` | Samma på emitter_data |
| `drop_impact.py` | Vad pulsbortfall kostar |
| `count_ab.py` | Delad kontra per-nivå antalsmodell |

`pipeline.learning_curve()` kör kurvan på din egen data: ge den en
filterfabrik och två listor av signaler.

---

## Det som inte är löst

**Matchningen mellan kluster** är fortfarande systemets svagaste del. Den
frågan — nytt kluster eller inte — är samma modellvalsfråga som BIC redan
löser en nivå ner, och borde besvaras likadant i stället för med en
handsatt tröskel. Blir aktuellt först när du vill gruppera flera signaler.

**Bortfallsdetektering** finns inte. Enligt mätningen i `drop_impact.py`
kostar 10 % bortfall dig 0,955 → 0,732 i nivåträffsäkerhet, och hela den
förlusten går att få tillbaka. Signaturen är exakt: en tappad puls slår ihop
två intervall till ett, så värdet blir ungefär summan av de intilliggande.
En värdekomponent vars medelvärde ligger nära en heltalsmultipel av en annan
komponents är därför en bortfallsartefakt, inte en verklig nivå — det testet
kan köras direkt på `param_vec`.
