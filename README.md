# Úloha Z6

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

## Pokyny

Práci zpracujte písemně (PDF), stačí stručně. V práci:

1. Odvoďte slabou formulaci daného problému.
2. Ověřte, že splňuje podmínky existence a jednoznačnosti řešení.
3. Popište diskretizaci pomocí MKP, sestavte matici tuhosti a vektor zatížení.
4. Vytvořte vlastní kód (např. MATLAB/Octave) a numericky řešte úlohu alespoň na dvou různých sítích.
5. Srovnejte výsledek s referenčním řešením.
6. Zobrazte průběh řešení ve vhodných řezech.
