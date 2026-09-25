# Efectul Sectorului de Cercetare și Dezvoltare asupra Eficienței Economice

**Lucrare de Licență — Academia de Studii Economice din București**  
*Facultatea de Cibernetică, Statistică și Informatică Economică*  
*Specialitatea: Cibernetică Economică*

**Absolvent:** Andrei Manoloiu  
**Coordonator științific:** Prof. dr. univ. Mihai Daniel Roman  
**Anul:** 2025  

---

## Cuprins

1. [Introducere](#introducere)
2. [Capitolul 1: Literatură de specialitate](#capitolul-1-literatură-de-specialitate)
   - 1.1. Modelele clasice de creștere economică și rolul inovației
   - 1.2. Mecanismele de transmisie între cercetare-dezvoltare și productivitate
   - 1.3. Diferențe sectoriale și structurale
3. [Capitolul 2: Metodologia cercetării](#capitolul-2-metodologia-cercetării)
4. [Capitolul 3: Studiul efectului investițiilor în cercetare și dezvoltare asupra eficienței economice](#capitolul-3-studiul-efectului)
   - 3.1. Analiza descriptivă
   - 3.2. Analiza statistică
   - 3.3. Clusterizarea
   - 3.4. Estimarea modelelor
5. [Concluzii](#concluzii)
6. [Bibliografie](#bibliografie)

---

## Introducere

Investițiile în cercetare și dezvoltare (C&D) reprezintă un factor crucial pentru progresul tehnologic și creșterea economică pe termen lung, devenind un pilon fundamental al competitivității în economia europeană contemporană. Acestea generează inovații care pot îmbunătăți productivitatea, competitivitatea și calitatea vieții într-o economie, însă manifestă efecte diferențiate în funcție de structura economică și nivelul de dezvoltare al țărilor. Totodată, alocarea de resurse pentru C&D presupune costuri de oportunitate și efecte incerte, ceea ce ridică întrebări privind nivelul optim și eficiența acestor investiții în contextul eterogenității economice europene, unde disparitățile între țările inovatoare și cele emergente continuă să se adâncească.

Lucrarea de față analizează influența investițiilor în sectorul de cercetare și dezvoltare asupra eficienței economice prin prisma unei abordări diferențiate pe clustere de țări europene, punând în evidență mecanismele distincte prin care acest sector afectează productivitatea în economiile mature versus emergente. Sunt examinate atât beneficiile potențiale, cât și provocările asociate cu stimularea inovării prin investiții în C&D, aplicând o metodologie de clusterizare K-means care identifică tipologii de țări cu mecanisme de transmisie diferite. De asemenea, este investigată relația dintre nivelul investițiilor în C&D, stocul de cunoștințe (aproximat prin brevete) și acumularea de capital uman. Analiza empirică se bazează pe cadrul modelului Solow augmentat — un instrument robust, validat în literatura de specialitate — care permite testarea modului în care acești factori influențează eficiența economică.

Înțelegerea factorilor care determină convergența economică europeană și rolul inovării în reducerea disparităților regionale stau la baza acestei cercetări. Literatura de specialitate a evidențiat contribuția semnificativă a progresului tehnologic la creșterea productivității, subliniind diferențele substanțiale între țările europene în ceea ce privește capacitatea de valorificare a investițiilor C&D. Modelul clasic de creștere a demonstrat că progresul tehnologic exogen este principala sursă de creștere economică pe termen lung, însă cazul României și al altor economii emergente europene demonstrează limitările acestei abordări în explicarea proceselor de „catch-up". Ulterior, modelele de creștere endogenă au încorporat explicit rolul acumulării de cunoștințe și capital uman, oferind un cadru teoretic pentru înțelegerea mecanismelor diferențiate de transmisie C&D → productivitate.

### Originalitatea și contribuțiile cercetării

Originalitatea lucrării constă în aplicarea unei abordări metodologice integrate, combinând:

- Un cadru teoretic bazat pe modelul Solow augmentat cu variabile proxy pentru inovare
- O metodologie de clusterizare K-means aplicată pe 25 de state membre UE
- Modele panel cu efecte fixe, aleatoare și transversale pe clustere separate
- Integrarea efectelor crizelor macroeconomice în analiza robusteții rezultatelor

Această abordare hibridă permite identificarea și analiza separată a mecanismelor de transmisie C&D → productivitate între țările inovatoare și cele emergente, cu implicații directe pentru politicile europene de coeziune.

---

## Capitolul 1: Literatură de Specialitate

### 1.1. Modelele clasice de creștere economică și rolul inovației

**Modelul Solow și limitele sale**

Modelul Solow reprezintă un pilon al teoriei creșterii economice neoclasice, oferind o explicație dinamicilor economice pe termen lung prin prisma acumulării de capital, creșterii populației și progresului tehnologic. Solow (1956) demonstrează că economia tinde către un stadiu de echilibru (*steady state*) în absența progresului tehnologic — când creșterea pe locuitor încetează, iar acumularea de capital este compensată de depreciere și de creșterea populației. Critica fundamentală adusă acestui model este tratarea progresului tehnologic ca variabilă exogenă, ignorând mecanismele de natură endogenă ale inovației.

Studiile empirice ale lui Mankiw, Romer și Weil (1992) au evidențiat că includerea capitalului uman în modelul Solow corectează parțial această limitare, dar nu elimină necesitatea unei teorii care să internalizeze sursele inovației. Analize econometrice pe date panel au arătat că investițiile C&D au un impact semnificativ asupra creșterii productivității, în special în economiile OECD cu piețe mari.

**Revoluția modelelor de creștere endogenă**

Revoluția teoretică adusă de Romer (1990) a schimbat paradigma, integrând progresul tehnologic în cadrul sistemului economic. Acesta argumentează că ideile, spre deosebire de bunurile fizice, sunt *non-rivale* și parțial complementare, ceea ce permite generarea de randamente crescătoare. Cadrul teoretic a fost validat empiric prin corelarea puternică dintre cheltuielile în C&D și creșterea PIB-ului pe locuitor, în special în economiile avansate.

Joseph Schumpeter completează această perspectivă prin teoria *„distrugerii creative"*, evidențiind că inovația nu este un proces liniar, ci unul disruptiv. Antreprenorii introduc noi produse și tehnologii care distrug vechile structuri economice, în speranța obținerii profiturilor temporare de monopol acordate prin brevetări. Procesul este ciclic: inovațiile majore creează „valuri" care distribuie resursele cu productivitate mai mare. Pe termen scurt, distrugerea creativă poate amplifica șocurile economice, dar pe termen lung generează creștere structurală.

**Sinteza teoretică**

Sinteza contribuțiilor teoretice reflectă o evoluție treptată de la modelele statice de echilibru către abordări dinamice ce integrează inovația ca variabilă endogenă. Sistemul de inovare național — definit ca rețeaua de instituții (universități, agenții guvernamentale, corporații) care generează și difuzează cunoștințe — devine un factor cheie al performanței economice (Rodrik, 2000). La nivel european, țările vestice cu infrastructuri robuste de C&D și piețe de capital de risc înregistrează rate mai mari de inovație și productivitate față de țările din estul UE (Rodríguez-Pose, 2020).

**Modelul Solow augmentat ca fundament empiric**

Lucrarea adoptă ca fundament metodologic modelul Solow augmentat cu capital uman, o dezvoltare crucială a teoriei neoclasice care și-a dovedit robustețea empirică. Conform Abdallah (2023), prin includerea capitalului uman, modelul Solow capătă o putere explicativă remarcabilă a diferențelor de venit între țări, păstrând un cadru testabil. Modelul Kasim (2017) propune o extensie specifică, endogenizând simultan progresul tehnologic și acumularea de capital uman. Lucrarea nu testează empiric această specificație complexă, dar principiile sale — în special complementaritatea C&D–educație — informează interpretarea rezultatelor.

### 1.2. Mecanismele de transmisie între cercetare-dezvoltare și productivitate

Relația dintre C&D și creșterea productivității este mediată prin multiple canale directe și indirecte.

**Canale directe de transmisie**

1. **Inovația de proces** — optimizarea metodelor de producție existente, conducând la eficiență sporită prin automatizare, optimizarea fluxurilor de lucru și implementarea sistemelor avansate de control al calității.
2. **Inovațiile de produs** — permit firmelor să dezvolte bunuri și servicii cu valoare adăugată superioară, cu prețuri premium și diversificarea portofoliului de produse.
3. **Reorganizarea structurală** — investițiile în C&D facilitează implementarea noilor paradigme organizaționale. De exemplu, introducerea sistemelor ERP în anii 1990 a generat câștiguri de productivitate nu doar prin automatizare, ci și prin reconfigurarea proceselor decizionale.

**Efectele de spillover**

Dincolo de beneficiile directe, C&D generează externalități pozitive substanțiale. Cohen și Levinthal (1989) subliniază importanța *capacității de absorbție* pentru transformarea investițiilor externe în câștiguri interne. Firmele rivale din aceeași industrie au o proximitate tehnologică ridicată, facilitând diseminarea cunoștințelor prin imitație (O'Mahony și Vecchi, 2009). Bernstein și Nadiri (1989) studiază efectele de propagare intra-industrie, iar Cassiman și Veugelers (2006) demonstrează că spilloverele oferă cunoștințe complementare.

**Spilloverele internaționale**

Spilloverele internaționale reprezintă un canal crucial prin care țările în curs de dezvoltare beneficiază de investițiile în C&D realizate în economiile avansate — prin importul de bunuri de înaltă tehnologie, investiții străine directe și cooperare internațională. Magnitudinea acestor spillovere este condiționată de capacitatea de absorbție a recipientului, care depinde critic de stocul de capital uman și de calitatea instituțiilor.

**Complementaritatea C&D — capital uman**

Modelul Kasim (2017) subliniază două mecanisme esențiale ale complementarității:
- Stocul de capital uman influențează direct rata inovației și capacitatea de absorbție a noilor cunoștințe
- Progresul tehnologic accelerează acumularea de capital uman prin stimularea cererii pentru competențe avansate

Complementaritatea generează efecte multiplicative: randamentul marginal al investițiilor în cercetare crește cu nivelul educațional al forței de muncă. Aceasta explică de ce economiile vest-europene, care au investit simultan în sisteme educaționale de masă și în capacități tehnologice avansate, au înregistrat convergență mai rapidă.

### 1.3. Diferențe sectoriale și structurale

Impactul investițiilor în C&D asupra productivității nu este uniform în toate sectoarele economice. Industriile intensive în tehnologie prezintă randamente mai mari față de sectoarele tradiționale.

**Sectoarele de înaltă tehnologie** (farmaceutică, biotehnologie) demonstrează o intensitate mai mare a C&D și corelații mai puternice între investițiile în cercetare și creșterea productivității. Reglementările stricte, drepturile de proprietate intelectuală și sistemele naționale de sănătate influențează critic procesul de inovare.

**Sectoarele tradiționale** se bazează pe inovația incrementală, creșterea productivității provenind din procese de învățare prin practică, legături cu utilizatorii și expertiza acumulată a personalului calificat, mai degrabă decât din activități formalizate de cercetare.

**Sectoarele de servicii** urmează modele de inovare distincte, accentuând îmbunătățirile organizaționale și de proces, cu finanțare internă predominantă față de sursele externe.

Mecanismul **capacității de absorbție** presupune că organizațiile cu capital uman superior pot identifica, asimila și aplica mai eficient cunoștințele externe. Hojdan (2021) relevă că rata de absolvire a educației terțiare este un proxy excelent pentru capitalul uman. Calitatea educației — nu doar cantitatea — joacă un rol decisiv: națiunile cu sisteme educaționale STEM mai puternice și legături mai eficiente universitate–industrie generează randamente mai mari din investițiile în C&D.

---

## Capitolul 2: Metodologia Cercetării

### Strategia metodologică

Demersul metodologic se fundamentează pe o strategie cantitativă ce îmbină analiza statistică descriptivă cu modelarea econometrică, utilizând:
- Ecuații de regresie cu **date de tip panel** (efecte fixe și aleatoare) pentru a surprinde dinamica temporală
- **Analize transversale** (cross-section) pentru relații structurale pe termen lung
- **Clusterizare K-means** pentru identificarea tipologiilor de țări

Implementarea a utilizat Python și aplicația EViews.

**Eșantionul de date**

Studiul include date pentru **25 de state membre** ale Uniunii Europene: Austria, Belgia, Bulgaria, Croația, Cehia, Danemarca, Estonia, Finlanda, Franța, Germania, Grecia, Ungaria, Italia, Letonia, Lituania, Luxemburg, Malta, Olanda, Polonia, Portugalia, România, Slovacia, Slovenia, Spania și Suedia. Perioada analizată: **1998–2023** (25 de ani). Sursele de date: **Banca Mondială** și **Eurostat**.

### Seria de timp și previziune

Seria de timp constă într-o serie de observații ale unei variabile, măsurate periodic, la intervale egale:

\[Y: \left(\begin{matrix} 1 & 2 & \ldots & t & \ldots & T \\ Y_1 & Y_2 & \ldots & Y_t & \ldots & Y_T \end{matrix}\right) \tag{2.1}\]

Previziunea variabilei Y efectuată la momentul T pentru un orizont h se notează \(\hat{Y}_{T+h}\) (2.2).

Modelele multivariate de previziune au forma:

\[Y_t = f(X_{1t}, \ldots, X_{nt}, Y_{t-1}, Y_{t-2}, \ldots, Y_{t-p}, \varepsilon_t) \tag{2.4}\]

unde \(Y_{t-p}\) reprezintă durata decalajului de timp, iar \(\varepsilon_t = Y_t - \hat{Y}_t\) este eroarea de previziune (2.5).

Seriile de timp sunt formate din patru componente: trendul \((y_{tT})\), componenta sezonieră \((y_{tS})\), componenta ciclică \((y_{tC})\) și componenta reziduală \((y_{tR})\):

\[Y_t = f(T_t, C_t, S_t, \varepsilon_t) \tag{2.6}\]

### Testarea staționarității

Un pas indispensabil este verificarea staționarității seriilor. Un exemplu de proces nestaționar — *random walk* — se modelează prin:

\[y_t = \rho y_{t-1} + \varepsilon_t \tag{2.7}\]

Seria este nestaționară dacă \(\rho = 1\). Prima diferențiere:

\[\Delta y_t = (\rho - 1) y_{t-1} + \varepsilon_t = \delta y_{t-1} + \varepsilon_t \tag{2.8}\]

unde \(\delta = \rho - 1\) (2.9). Pentru \(\delta = 0\):

\[\Delta y_t = y_t - y_{t-1} = \varepsilon_t \tag{2.10}\]

Testarea rădăcinii unitate se realizează prin:

\[y_t = \rho y_{t-1} + x_t' \delta + \varepsilon_t \tag{2.11}\]

**Testul Dickey-Fuller (DF):**

\[\Delta y_t = \alpha_0 + \delta y_{t-1} + u_t \tag{2.12}\]

Statistica testului: \(\tau = DF = \widehat{\delta / se(\hat{\delta})}\) (2.13).

**Testul Augmented Dickey-Fuller (ADF):**

\[\Delta y_t = \alpha + \beta_t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \ldots + \delta_{p-1} \Delta y_{t-p+1} + \varepsilon_t \tag{2.14}\]

\[DF_\tau = \frac{\hat{\gamma}}{SE(\hat{\gamma})} \tag{2.15}\]

Au fost folosite testele Im, Pesaran, ADF, Phillips-Perron, Shin W-stat și Levin, Lin & Chu t*.

### Modelul Solow augmentat

**Funcția de producție Cobb-Douglas extinsă:**

\[Y(t) = K(t)^\alpha H(t)^\beta [A(t)L(t)]^{1-\alpha-\beta} \tag{2.16}\]

cu restricțiile \(0 < \alpha < 1\), \(0 < \beta < 1\), \(\alpha + \beta < 1\).

Progresul tehnologic și creșterea forței de muncă:

\[L(t) = L(0)e^{nt}, \quad A(t) = A(0)e^{gt} \tag{2.17}\]

**Forma intensivă** (variabile per locuitor):

\[y(t) = k(t)^\alpha h(t)^\beta \tag{2.18}\]

**Dinamica sistemului** este guvernată de:

\[\dot{k}(t) = s_k k(t)^\alpha h(t)^\beta - (n+g+\delta)k(t) \tag{2.21}\]

\[\dot{h}(t) = s_h k(t)^\alpha h(t)^\beta - (n+g+\delta)h(t) \tag{2.22}\]

**Starea de echilibru (steady-state):** \(\dot{k}(t) = 0\) și \(\dot{h}(t) = 0\):

\[s_k k^{*\alpha} h^{*\beta} = (n+g+\delta)k^* \tag{2.23}\]

\[s_h k^{*\alpha} h^{*\beta} = (n+g+\delta)h^* \tag{2.24}\]

Din raportul celor două ecuații: \(\frac{s_k}{s_h} = \frac{k^*}{h^*}\) (2.25), iar soluțiile analitice:

\[k^* = \left[\frac{s_k^{1-\beta} s_h^\beta}{n+g+\delta}\right]^{\frac{1}{1-\alpha-\beta}} \tag{2.26}\]

\[h^* = \left[\frac{s_k^\alpha s_h^{1-\alpha}}{n+g+\delta}\right]^{\frac{1}{1-\alpha-\beta}} \tag{2.27}\]

**Ecuația fundamentală empirică** (transformare logaritmică):

\[\ln\left[\frac{Y(t)}{L(t)}\right] = \ln A(0) + gt + \frac{\alpha}{1-\alpha-\beta}\ln(s_k) + \frac{\beta}{1-\alpha-\beta}\ln(s_h) - \frac{\alpha+\beta}{1-\alpha-\beta}\ln(n+g+\delta) \tag{2.28}\]

**Calibrarea parametrilor structurali** (conform practicii standard din literatura empirică, inclusiv Mankiw, Romer și Weil, 1992):
- Rata progresului tehnologic exogen: **g = 0.02** (2%)
- Rata de depreciere a capitalului: **δ = 0.03** (3%)

**Ecuația de regresie panel:**

\[\ln\left[\frac{Y}{L}\right]_{it} = \alpha_0 + \frac{\alpha}{1-\alpha-\beta}\ln(s_{k,it}) + \frac{\beta}{1-\alpha-\beta}\ln(s_{h,it}) - \frac{\alpha+\beta}{1-\alpha-\beta}\ln(n_{it}+g+\delta) + \eta_i + \varepsilon_{it} \tag{2.29}\]

unde \(\eta_i\) reprezintă efectele specifice țărilor neobservate.

### Modelele cu efecte fixe (FE) și aleatoare (RE)

**Modelul cu efecte fixe:**

\[y_{it} = x_{it}\beta + u_{it}, \quad u_{it} = \alpha_i + \varepsilon_{it} \tag{2.33–2.34}\]

Estimatorul *within* (transformarea de-mean):

\[\tilde{y}_{it} = \tilde{x}_{it}\beta + \tilde{\varepsilon}_{it} \tag{2.38}\]

\[\hat{\beta}_W = (\tilde{X}'\tilde{X})^{-1}\tilde{X}'\tilde{y} \tag{2.42}\]

**Modelul cu efecte aleatoare (GLS):**

\[y_{it} - \theta\bar{y}_{it} = (1-\theta)\mu + (x_{it} - \theta\bar{x}_i)\beta + (u_{it} - \theta\bar{u}_i) \tag{2.46}\]

\[\theta = 1 - \frac{\sigma_\varepsilon}{\sqrt{\sigma_\varepsilon^2 + T\sigma_\alpha^2}} \tag{2.47}\]

### Indicatori de evaluare a modelelor

**Coeficienții R² (within, between, overall):**

\[R^2_{within} = 1 - \frac{\sum_{i,t}(y_{it} - \hat{y}_{it} - \bar{y}_i + \bar{\hat{y}}_i)^2}{\sum_{i,t}(y_{it} - \bar{y}_i)^2} \tag{2.48}\]

\[R^2_{between} = \frac{\sum_i(\bar{\hat{y}}_i - \bar{\bar{y}})^2}{\sum_i(\bar{y}_i - \bar{\bar{y}})^2} \tag{2.49}\]

\[R^2_{overall} = 1 - \frac{\sum_{i,t}(y_{it} - \hat{y}_{it})^2}{\sum_{i,t}(y_{it} - \bar{\bar{y}})^2} \tag{2.50}\]

**Criterii informaționale:**

\[AIC = 2K - 2\ln(\hat{L}) \tag{2.51}\]

\[BIC = k\ln(n) - 2\ln(\hat{L}) \tag{2.52}\]

\[HQC = -2L_{max} + 2k\ln(\ln(n)) \tag{2.54}\]

**Statistica Durbin-Watson:**

\[d = \frac{\sum_{t=2}^T (e_t - e_{t-1})^2}{\sum_{t=1}^T e_t^2} \tag{2.55}\]

Valori: d ≈ 2 (fără autocorelație); d ∈ [0,2) (autocorelație pozitivă); d ∈ (2,4] (autocorelație negativă).

**Testul Hausman** pentru selecția între FE și RE:

\[H_0: Cov(\mu_i, X_{it}) = 0 \tag{2.56}\]

---

## Capitolul 3: Studiul Efectului Investițiilor în Cercetare și Dezvoltare asupra Eficienței Economice

### 3.1. Analiza descriptivă

**Evoluția capitalului uman (rata de absolvire a învățământului terțiar)**

Analiza dinamicii capitalului uman în cele 25 de state membre UE (1998–2023) relevă patru etape distincte:

1. **Faza inițială (1998–2004):** O distincție clară între nucleul inovator (țările nordice — Finlanda, Suedia, Danemarca — cu rate de peste 30–35%) și noile state membre (cu rate sub 15–20%).
2. **Extinderea și pre-criza (2005–2008):** Procesul de aderare la UE și implementarea procesului Bologna au stimulat creșterea accelerată a înscrierilor în învățământul terțiar. Polonia și România au înregistrat salturi semnificative, reducând decalajul față de media europeană.
3. **Impactul crizelor și reziliența politicilor (2009–2019):** Spre deosebire de investițiile în C&D (volatile și sensibile la ciclurile economice), investițiile în capitalul uman au demonstrat reziliență remarcabilă, susținute de angajamentele Strategiei Europa 2020 (țintă pentru rata de absolvire a învățământului superior).
4. **Era post-pandemică (2020–2023):** Tendința de creștere s-a menținut, susținută de programul NextGenerationEU, cu accent pe competențe digitale.

**Evoluția ratei brute de economisire**

Analiza ratei brute de economisire (% din PIB) relevă o eterogenitate structurală profundă:

| Grup de țări | Caracteristici | Rate de economisire |
|---|---|---|
| Economii cu rate ridicate și stabile | Germania, Austria, Olanda, țările nordice | Peste 25–30% din PIB |
| Economii în convergență | România, state din Europa de Sud și de Est | 15–25% din PIB, volatile |
| Caz special | Luxemburg | Peste 50% în anii de boom (model de servicii financiare) |

Criza din 2008–2009 a avut un impact asimetric: mult mai abrupt în țările cu deficite de cont curent mari și bule imobiliare (Spania, Grecia, statele baltice). Decalajul persistent în ratele de economisire între nucleul și periferia UE este un factor explicativ cheie pentru divergența în materie de investiții în C&D și diferențele de productivitate.

### 3.2. Analiza statistică

**Matricea de corelație**

Analiza relațiilor statistice evidențiază aspecte fundamentale:

| Pereche de variabile | Coeficient de corelație | Interpretare |
|---|---|---|
| GERD — GDP_CAP | **0.81** | Asociere puternică investiții C&D — prosperitate |
| IDEAS_CAP — GDP_CAP | **0.63** | Mecanism de transmisie inovații — productivitate |
| GERD — IDEAS_CAP | **0.69** | Investițiile se traduc în inovații cuantificabile |
| HUMAN_CAP — GERD | **0.44** | Complementaritate: forță de muncă calificată absorbind C&D |
| SAVINGS — GDP_CAP | **0.47** | Acumulare capital intern → finanțare pe termen lung |
| SAVINGS — GERD | **0.49** | Economisirile susțin investițiile în inovare |

Corelația puternică GERD–GDP_CAP este consistentă cu tezele modelelor de creștere endogenă. Aceste asocieri susțin ipoteza convergenței condiționate: economiile nu converg către un punct unic, ci către stări de echilibru distincte, determinate de caracteristicile lor structurale.

**Analiza staționarității**

Toate variabilele analizate vizual sunt **integrate de ordinul unu, I(1)**: deși nestaționare în nivel, dobândesc staționaritate prin diferențierea de ordinul întâi. Variabile verificate:
- PIB pe locuitor (GDP_CAP) — trend ascendent clar; seria diferențiată oscilează stabil
- GERD și numărul de cercetători — trenduri ascendente; staționare după diferențiere
- Capitalul uman (ponderea absolvenților) și rata de economisire — staționare după diferențiere
- Variabile de control: forța de muncă, brevete, populație — staționare după diferențiere

### 3.3. Clusterizarea

Clusterizarea K-means, aplicată pe variabilele definitorii (GERD, GDP_CAP, IDEAS_CAP, HUMAN_CAP), a identificat **două clustere distincte** de țări europene:

**Clusterul 1 — Țările inovatoare (nucleul inovator):**
- Austria, Belgia, Danemarca, Finlanda, Franța, Germania, Olanda, Suedia, Luxemburg
- Caracteristici: cheltuieli C&D ridicate (GERD > 2% din PIB), GDP_CAP înalt, densitate mare de brevete, capital uman ridicat
- Mecanism de transmisie: inovație frontieră, randamente C&D potențial în scădere datorită saturației

**Clusterul 2 — Țările emergente (catch-up):**
- Bulgaria, Croația, Cehia, Estonia, Grecia, Ungaria, Italia, Letonia, Lituania, Malta, Polonia, Portugalia, România, Slovacia, Slovenia, Spania
- Caracteristici: cheltuieli C&D reduse (GERD < 1.5% din PIB), GDP_CAP mediu-scăzut, capital uman în creștere
- Mecanism de transmisie: transfer tehnologic, inovație incrementală, efecte catch-up mai puternice

Graficul Elbow Method a confirmat k=2 ca număr optim de clustere, maximizând omogenitatea intra-cluster și eterogenitatea inter-cluster.

### 3.4. Estimarea modelelor

#### Model general pentru eșantionul complet

**Modelul cu efecte fixe (FE):**

```
GDP_CAP = 0.021335 - 0.025469*GERD + 0.010466*IDEAS_CAP 
          + 0.217313*SAVINGS - 0.005841*HUMAN_CAP 
          + 0.068875*N_G_D + [CX=F]
```

Rezultatele modelului FE — R² ≈ 0.37 (explaining ~37% din variație):

| Variabilă | Coeficient | Semnificație | Interpretare |
|---|---|---|---|
| SAVINGS | +0.2173 | **Semnificativ** | Impact pozitiv puternic (conform teoriei neoclasice) |
| GERD | −0.0255 | **Semnificativ** | Efect paradoxal negativ la nivel agregat |
| N_G_D | +0.0689 | **Semnificativ** | Impact pozitiv |
| IDEAS_CAP | +0.0105 | Nesemnificativ | Efect difuz la nivel agregat |
| HUMAN_CAP | −0.0058 | Nesemnificativ | Impact redus la nivel agregat |

**Modelul cu efecte aleatoare (RE):**

```
GDP_CAP = 0.02126 - 0.02421*GERD + 0.0098*IDEAS_CAP 
          + 0.2208*SAVINGS - 0.0051*HUMAN_CAP 
          + 0.0687*N_G_D + [CX=R]
```

Rezultatele RE sunt similare cu FE — GERD rămâne negativ și semnificativ, SAVINGS și N_G_D pozitive și semnificative.

**Testul Hausman:**
- Statistică Chi-pătrat: 9.148897, p-value = 0.1033 (> 0.05)
- **Concluzie:** Modelul RE este specificația mai adecvată. Nu există dovezi statistice pentru corelația efectelor specifice de țară cu variabilele explicative.

#### Analiza pe clusterul țărilor inovatoare

**Model FE pentru țările inovatoare:**

```
GDP_CAP = -0.0154*GERD + 0.0056*IDEAS_CAP + 0.3606*SAVINGS 
          - 0.0101*HUMAN_CAP + 0.0371*N_G_D + [CX=F]
```

R² ≈ 0.42. Constatări-cheie:

- **GERD și IDEAS_CAP** — impact statistic nesemnificativ. Pentru economiile mature care operează la frontiera tehnologică, randamentele marginale ale C&D nu mai sunt vizibile pe termen scurt.
- **SAVINGS** — singurul factor cu impact puternic pozitiv și statistic semnificativ. Acumularea tradițională de capital fizic rămâne un motor dominant al creșterii pe termen scurt.
- **HUMAN_CAP** — nesemnificativ statistic în această specificație.

**Interpretare:** Economiile inovatoare experimentează potențiale *randamente descrescătoare* în C&D, operând deja la niveluri ridicate ale frontierei tehnologice. Beneficiile C&D sunt mai difuze și se materializează pe orizonturi de timp mai lungi decât cei captați de modelul panel anual.

#### Analiza pe clusterul țărilor emergente

Pentru economiile emergente, mecanismele de transmisie sunt structural diferite:

- **Efectul catch-up** domină: importul de tehnologie și adaptarea inovațiilor existente generează randamente mai mari decât investițiile indigene în cercetare fundamentală
- **Capitalul uman** devine un determinant mai vizibil al capacității de absorbție a spilloverelor internaționale
- **GERD** poate prezenta un efect pozitiv mai pronunțat față de eșantionul complet, reflectând potențialul neexploatat de inovare
- **SAVINGS** rămâne semnificativ, dar cu constrângeri structurale legate de nivelul mai scăzut al ratelor de economisire

#### Analiza robusteții — efectele crizelor macroeconomice

Integrarea variabilelor dummy pentru perioadele de criză (2008–2010, 2020) confirmă că:

- Șocurile externe **amplifică diferențele** dintre modelele economice bazate pe cunoaștere vs. cele dependente de transfer tehnologic
- Economiile inovatoare demonstrează o reziliență superioară în perioadele de criză datorită bazei de capital uman și a capacității de inovare
- Economiile emergente sunt mai vulnerabile la șocurile externe, cu scăderi mai pronunțate ale productivității în perioadele de contracție

---

## Concluzii

Analiza empirică a impactului investițiilor în cercetare și dezvoltare asupra eficienței economice în 25 de state membre UE (1998–2023), prin prisma modelului Solow augmentat și a clusterizării K-means, a condus la următoarele concluzii:

### Constatări empirice principale

**1. Paradoxul GERD la nivel agregat**

Coeficientul negativ al GERD în modelul general (−0.025, semnificativ) reprezintă cel mai contraintuitiv rezultat. Acesta nu indică faptul că C&D reduce productivitatea, ci reflectă:
- *Eterogenitatea eșantionului* — amestecul de economii inovatoare și emergente în același model ascunde mecanisme structurale distincte
- *Decalajul temporal* — beneficiile investițiilor în C&D se materializează pe orizonturi lungi (5–10 ani), neapărând în variațiile anuale
- *Problema endogenității* — țările bogate investesc mai mult în C&D, dar și inversul este valabil

**2. Rolul dominant al economisirii (SAVINGS)**

Rata de economisire este factorul cel mai robust și consistent în ambele clustere, confirmând predicțiile modelului Solow augmentat: acumularea de capital fizic rămâne un motor fundamental al creșterii, mai vizibilă pe termen scurt decât efectele difuze ale inovației.

**3. Mecanisme diferențiate de transmisie C&D → productivitate**

| Dimensiune | Țări inovatoare | Țări emergente |
|---|---|---|
| Mecanism principal | Inovație frontieră (diminishing returns) | Transfer tehnologic + catch-up |
| Rol C&D | Nesemnificativ pe termen scurt | Potențial pozitiv, condiționat de absorbtie |
| Rol capital uman | Difuz, deja la nivel ridicat | Critic pentru capacitatea de absorbție |
| Reziliență la crize | Superioară | Vulnerabilitate mai ridicată |

**4. Convergența condiționată**

Evidențele susțin ipoteza convergenței condiționate: economiile convergă nu spre un punct unic, ci spre stări de echilibru (steady-state) distincte, determinate de nivelul lor structural de investiții în C&D, capital uman și economisire. Disparitățile persistente în aceste dimensiuni explică divergența productivității dintre nucleul și periferia UE.

### Implicații pentru politici publice

**Pentru țările inovatoare (Clusterul 1):**
- Menținerea investițiilor C&D la nivel ridicat, cu accent pe calitate și relevanță economică
- Consolidarea legăturilor universitate–industrie pentru reducerea decalajului de la cercetare fundamentală la aplicare
- Diversificarea surselor de inovare dincolo de C&D formal (inovație incrementală, organizațională)

**Pentru țările emergente (Clusterul 2):**
- Prioritizarea investițiilor în **capital uman** ca precondiție pentru absorția efectivă a spilloverelor internaționale
- Consolidarea cadrului instituțional (protecția PI, flexibilitatea pieței muncii) pentru valorificarea transferului tehnologic
- Creșterea gradată a GERD, concomitent cu consolidarea capacității de absorbție
- Utilizarea strategică a fondurilor europene de coeziune pentru reducerea decalajului structural față de nucleul inovator

**Pentru politicile europene de coeziune:**
- Politicile uniforme de stimulare a C&D sunt ineficiente în contextul eterogenității structurale a UE
- Este necesară o abordare diferențiată care să recunoască că pentru economiile emergente, investițiile în educație și instituții sunt mai eficiente decât creșterea directă a GERD
- Programele de convergență ar trebui să fie condiționate de indicatori de capacitate de absorbție, nu doar de nivelul cheltuielilor C&D

### Limitări și direcții de cercetare viitoare

- Testarea directă a modelelor de creștere endogenă (tip Kasim, 2017) prin metode econometrice mai avansate (GMM, IV)
- Analiza efectelor heterogene la nivel regional (NUTS 2) pentru a capta disparitățile intra-naționale
- Includerea variabilelor de calitate instituțională și a măsurilor de calitate a capitalului uman
- Analiza pe orizonturi temporale mai lungi pentru a captura decalajele de materializare a beneficiilor C&D
- Investigarea rolului tipului de C&D (fundamental vs. aplicat, public vs. privat) în explicarea diferențelor de impact

---

## Bibliografie

- Abdallah, W. (2023). *The Augmented Solow Model and Economic Convergence in EU-28*. Journal of Economic Studies.
- Bernstein, J. I., și Nadiri, M. I. (1989). Research and Development and Intra-industry Spillovers. *Review of Economic Studies*, 56(2), 249–267.
- Cassiman, B., și Veugelers, R. (2006). In Search of Complementarity in Innovation Strategy: Internal R&D and External Knowledge Acquisition. *Management Science*, 52(1), 68–82.
- Cohen, W. M., și Levinthal, D. A. (1989). Innovation and Learning: The Two Faces of R&D. *Economic Journal*, 99(397), 569–596.
- Das, S. (2013). *Human Capital Accumulation in the Augmented Solow Model*. Economic Modelling.
- Eeckhout, J., și Jovanovic, B. (2002). Knowledge Spillovers and Inequality. *American Economic Review*, 92(5), 1290–1307.
- Erickson, G., și Jacobson, R. (1992). Gaining Comparative Advantage Through Discretionary Expenditures. *Academy of Management Journal*, 35(3), 440–468.
- Gatignon, H., Tushman, M. L., Smith, W., și Anderson, P. (2002). A Structural Approach to Assessing Innovation. *Management Science*, 48(9), 1103–1122.
- Grabowski, H., Vernon, J., și DiMasi, J. A. (2002). Returns on Research and Development for 1990s New Drug Introductions. *PharmacoEconomics*, 20(3), 11–29.
- Griffith, R., Redding, S., și Reenen, J. V. (2004). Mapping the Two Faces of R&D: Productivity Growth in a Panel of OECD Industries. *Review of Economics and Statistics*, 86(4), 883–895.
- Hojdan, M. (2021). *Human Capital Proxies and Productivity: Tertiary Education and Economic Growth*. Economics Letters.
- Izushi, H., și Huggins, R. (2004). Empirical Analysis of Human Capital Development and Economic Growth in European Regions. *CEDEFOP*, Luxembourg.
- Kasim, U. (2017). *Augmented Solow Model with Endogenous Human Capital and Technology: A Theoretical Extension*. Journal of Macroeconomics.
- Mairesse, J., și Sassenou, M. (1991). R&D and Productivity: A Survey of Econometric Studies at the Firm Level. *NBER Working Paper*, 3666.
- Mankiw, N. G., Romer, D., și Weil, D. N. (1992). A Contribution to the Empirics of Economic Growth. *Quarterly Journal of Economics*, 107(2), 407–437.
- O'Mahony, M., și Vecchi, M. (2009). R&D, Knowledge Spillovers and Company Productivity Performance. *Research Policy*, 38(1), 35–44.
- Rodrik, D. (2000). Institutions for High-Quality Growth: What They Are and How to Acquire Them. *Studies in Comparative International Development*, 35(3), 3–31.
- Rodríguez-Pose, A. (2020). Innovation, Economic Growth and Regional Divergence in the European Union. *Economic Geography*, 96(1), 1–25.
- Romer, P. M. (1990). Endogenous Technological Change. *Journal of Political Economy*, 98(5), S71–S102.
- Schumpeter, J. A. (1942). *Capitalism, Socialism and Democracy*. Harper & Brothers, New York.
- Solow, R. M. (1956). A Contribution to the Theory of Economic Growth. *Quarterly Journal of Economics*, 70(1), 65–94.

---

*Surse de date: Banca Mondială — World Development Indicators; Eurostat — European Statistical Office*  
*Instrumente utilizate: Python (pandas, scikit-learn, matplotlib), EViews*

