# Úloha Z6: Poissonova rovnice

## Zadání

Pro oblast $\Omega = (0, 1)^2$ je dána okrajová úloha

$$
\begin{aligned}
-\Delta u &= f && \text{v } \Omega, \\
u &= 0 && \text{na } \Gamma_D, \\
-\frac{\partial u}{\partial n} &= \frac{1}{8}(u + 7) && \text{na } \Gamma_N,
\end{aligned}
$$

kde

$$
f(x) = 1 - \sin(\pi x_1)\sin(\pi x_2),
$$

$$
\Gamma_D = \{x : x_2 \in [0, 0.5],\ x_1 = 0\}
$$

a $\Gamma_N$ je zbytek hranice.

### Pokyny

1. Odvoďte slabou formulaci daného problému.
2. Ověřte, že splňuje podmínky existence a jednoznačnosti řešení.
3. Popište diskretizaci pomocí MKP, sestavte matici tuhosti a vektor zatížení.
4. Vytvořte vlastní kód (např. MATLAB/Octave) a numericky řešte úlohu alespoň na dvou různých sítích.
5. Srovnejte výsledek s referenčním řešením.
6. Zobrazte průběh řešení ve vhodných řezech.

## Podklady k řešení

### 1. Definice úlohy (Poissonova rovnice)

Hledáme funkci $u(x,y)$ na oblasti $\Omega = (0,1)^2$ splňující:

$$
-\Delta u = f(x,y) \quad \text{v } \Omega.
$$

Hranice je rozdělena na dvě části $\partial\Omega = \Gamma_D \cup \Gamma_N$ s následujícími okrajovými podmínkami:

- **Dirichletova OP (homogenní):** $u = 0$ na $\Gamma_D$,
- **Robinova OP:** $-\frac{\partial u}{\partial n} = \frac{1}{8}u + \frac{7}{8}$ na $\Gamma_N$.

### 2. Slabá formulace

Hledáme $u \in V = \{v \in H^1(\Omega) : v = 0 \text{ na } \Gamma_D\}$ takové, že:

$$
a(u,v) = L(v) \quad \forall v \in V.
$$

Bilineární forma $a(u,v)$ a lineární forma $L(v)$ jsou definovány jako:

$$
a(u,v) = \int_{\Omega} \nabla u \cdot \nabla v \,\mathrm{d}x\mathrm{d}y
+ \frac{1}{8}\int_{\Gamma_N} u v \,\mathrm{d}s,
$$

$$
L(v) = \int_{\Omega} f v \,\mathrm{d}x\mathrm{d}y
- \frac{7}{8}\int_{\Gamma_N} v \,\mathrm{d}s.
$$

### 3. Lokální matice tuhosti (2D elementy $K$)

Díky transformaci na referenční element se velikost kroku $h$ kompletně vykrátí. Lokální matice tuhosti jsou konstantní pro libovolné dělení sítě.

**Dolní trojúhelník (typ A)** má lokální uzly vlevo dole, vpravo dole a vpravo nahoře:

$$
\mathbf{A}^{K}_{\mathrm{dolni}} = \frac{1}{2}
\begin{pmatrix}
1 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 1
\end{pmatrix}.
$$

**Horní trojúhelník (typ B)** má lokální uzly vlevo dole, vpravo nahoře a vlevo nahoře:

$$
\mathbf{A}^{K}_{\mathrm{horni}} = \frac{1}{2}
\begin{pmatrix}
1 & 0 & -1 \\
0 & 1 & -1 \\
-1 & -1 & 2
\end{pmatrix}.
$$

### 4. Numerická kvadratura pravé strany (2D elementy $K$)

Pro přibližnou integraci zdroje $f(x,y)$ na trojúhelníku $K$ o ploše $|K| = \frac{h^2}{2}$ používáme **třívrcholové integrační pravidlo**:

$$
\int_K \Phi(x,y) \,\mathrm{d}x\mathrm{d}y
\approx \frac{|K|}{3}\left(\Phi(A) + \Phi(B) + \Phi(C)\right).
$$

Po transformaci na referenční element a dosazení bázových funkcí $\phi_i$ získáváme pro každý uzel $i \in \{A, B, C\}$ elementu $K$ lokální příspěvek pravé strany:

$$
F^K_i \approx f(x_i, y_i) \cdot \frac{h^2}{6}.
$$

### 5. Příspěvky od okrajových podmínek (OP)

#### Robinova OP (1D hraniční hrany $e$ na $\Gamma_N$)

Pro každou hraniční hranu $e$ o délce $h_e = h$ spojující uzly $p$ a $q$:

**Lokální Robinova matice** (přičítá se do $\mathbf{A}$):

$$
\mathbf{A}^e_{\mathrm{Robin}} = \frac{h}{48}
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}.
$$

**Lokální Robinův vektor** (přičítá se do $\mathbf{F}$):

$$
\mathbf{F}^e_{\mathrm{Robin}} = -\frac{7h}{16}
\begin{pmatrix}
1 \\
1
\end{pmatrix}.
$$

#### Dirichletova OP (homogenní na $\Gamma_D$)

Aplikuje se **metodou redukce** algebraické soustavy:

1. Definuje se indexové pole `free_nodes` obsahující všechny uzly mimo hranici $\Gamma_D$.
2. Z globální matice $\mathbf{A}$ a vektoru $\mathbf{F}$ se **odstraní (vyříznou) řádky a sloupce** odpovídající uzlům na $\Gamma_D$:

$$
\mathbf{A}_{\mathrm{free}}
= \mathbf{A}[\mathrm{free\_nodes}, \mathrm{free\_nodes}],
$$

$$
\mathbf{F}_{\mathrm{free}} = \mathbf{F}[\mathrm{free\_nodes}].
$$

3. Řeší se zredukovaná soustava rovnic:

$$
\mathbf{A}_{\mathrm{free}}\mathbf{U}_{\mathrm{free}}
= \mathbf{F}_{\mathrm{free}}.
$$
