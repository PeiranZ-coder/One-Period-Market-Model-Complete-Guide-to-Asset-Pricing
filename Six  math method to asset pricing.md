

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [Linear Algebra Foundation](#2-linear-algebra-foundation)
3. [The Six Pricing Methods](#3-the-six-pricing-methods)
   - [Method 1: Replicating Portfolio](#method-1-replicating-portfolio)
   - [Method 2: Risk-Neutral Pricing](#method-2-risk-neutral-pricing)
   - [Method 3: Arrow-Debreu State Prices](#method-3-arrow-debreu-state-prices)
   - [Method 4: Stochastic Discount Factor](#method-4-stochastic-discount-factor-sdf)
   - [Method 5: Linear Programming Duality](#method-5-linear-programming-duality)
   - [Method 6: Utility Indifference Pricing](#method-6-utility-indifference-pricing)
4. [Why All Methods Agree: The Deep Equivalence](#4-why-all-methods-agree-the-deep-equivalence)
5. [The Fundamental Theorem of Asset Pricing (FTAP)](#5-the-fundamental-theorem-of-asset-pricing-ftap)
6. [Arrow-Debreu Prices: Deep Dive](#6-arrow-debreu-prices-deep-dive)
7. [Connection to Continuous Time](#7-connection-to-continuous-time)
8. [Summary Tables](#8-summary-tables)
9. [Recommended Reading](#9-recommended-reading)


## 1. Problem Setup

### Exercise 15 (Capiński & Zastawniak)

A stock with spot price $S(0) = \$50$ will be worth either $S_u = \$60$ or $S_d = \$45$ in three months. The continuously compounded risk-free rate is $r = 6\%$, $T = 0.25$ years.

**Securities:**
- Cash (bond): invest $B(0) = 1$ today, receive $B(T) = e^{rT}$ at maturity
- Underlying asset: price $S(0) = 50$ today

**Questions:**
1. Find the payoff matrix
2. Is the market complete?
3. Replicate a 3-month ATM put ($K = 50$)

### General Example (Six Methods)

For the comprehensive pricing exercise we use:

| Parameter | Value |
|---|---|
| $S(0)$ | 100 |
| $S_u$ | 120 |
| $S_d$ | 90 |
| $r$ | 5% |
| $T$ | 1 year |
| $K$ (call strike) | 105 |
| $V_u = \max(S_u - K, 0)$ | 15 |
| $V_d = \max(S_d - K, 0)$ | 0 |
| $B(T) = e^{rT}$ | 1.05127 |
| True probability $p_u$ | 0.6 |
| Risk aversion $\gamma$ | 2 |

---

## 2. Linear Algebra Foundation

### State Space and Vector Space

With $m$ states and $n$ securities, define:

- **State space**: $\Omega = \{\omega_1, \ldots, \omega_m\}$
- **Payoff vector of security $j$**: $d_j \in \mathbb{R}^m$, where $(d_j)_i$ = payoff in state $i$
- **Portfolio vector**: $\phi = (\phi_1, \ldots, \phi_n)^T \in \mathbb{R}^n$, units held of each security
- **Price vector today**: $p_0 = (p_1, \ldots, p_n)^T \in \mathbb{R}^n$

### The Payoff Matrix

Stacking payoff vectors as columns:

$$D = [d_1 \mid d_2 \mid \cdots \mid d_n] \in \mathbb{R}^{m \times n}$$

Entry $D_{ij}$ = payoff of security $j$ in state $i$.

For our two-security, two-state model:

$$D = \begin{pmatrix} e^{rT} & S_u \\ e^{rT} & S_d \end{pmatrix}$$

### Column Space = Set of Replicable Claims

$$\text{col}(D) = \{D\phi : \phi \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$$

Any contingent claim $V \in \text{col}(D)$ can be replicated. The market is complete when $\text{col}(D) = \mathbb{R}^m$.

### Market Completeness

$$\text{Market complete} \iff \text{rank}(D) = m \iff \text{col}(D) = \mathbb{R}^m$$

When $m = n$ (square payoff matrix):

$$\text{Complete} \iff \det(D) \neq 0 \iff D \text{ invertible}$$

**Exercise 15 verification:**

$$\det(D) = e^{rT}(S_d - S_u) = 1.0151 \times (-15) = -15.227 \neq 0 \quad \checkmark$$

**Geometric intuition**: The two payoff vectors $d_1 = (e^{rT}, e^{rT})^T$ and $d_2 = (S_u, S_d)^T$ are linearly independent — they span all of $\mathbb{R}^2$, so any claim can be written as a linear combination.

### The Primal-Dual Structure

The entire theory rests on two dual linear systems:

| | System | Unknown | Meaning |
|---|---|---|---|
| **Primal** | $D\phi = V$ | $\phi$ (portfolio weights) | Replication |
| **Dual** | $D^T\psi = p_0$ | $\psi$ (state prices) | Pricing |

Both have unique solutions when $D$ is invertible. The optimal values coincide:

$$\phi^T p_0 = \psi^T V$$

This is the Strong Duality Theorem applied to financial markets.

---

## 3. The Six Pricing Methods

All methods price the same European call option with $V_u = 15$, $V_d = 0$.

### Method 1: Replicating Portfolio

**Principle**: Construct a portfolio $\phi = (\phi_1, \phi_2)^T$ (cash, stock) that exactly matches the claim's payoff in every state. By no-arbitrage (Law of One Price), the claim's price equals the portfolio's cost today.

**System to solve** — $D\phi = V$:

$$\begin{pmatrix} 1.05127 & 120 \\ 1.05127 & 90 \end{pmatrix} \begin{pmatrix} \phi_1 \\ \phi_2 \end{pmatrix} = \begin{pmatrix} 15 \\ 0 \end{pmatrix}$$

**Step 1** — Subtract the two equations (eliminating $\phi_1$):

$$\phi_2 = \frac{V_u - V_d}{S_u - S_d} = \frac{15 - 0}{120 - 90} = \frac{1}{2} = 0.5$$

This is the **option delta** $\Delta$: the stock position needed to hedge.

**Step 2** — Substitute into the down-state equation:

$$\phi_1 \times 1.05127 + 0.5 \times 90 = 0 \implies \phi_1 = \frac{-45}{1.05127} = -42.804$$

Negative $\phi_1$ means **borrowing** at the risk-free rate.

**Price today:**

$$C(0) = \phi_1 \cdot B(0) + \phi_2 \cdot S(0) = -42.804 \times 1 + 0.5 \times 100 = \mathbf{7.196}$$

**Cash flow interpretation:**

| Position | Today | Up state | Down state |
|---|---|---|---|
| Long 0.5 shares | $-50$ | $+60$ | $+45$ |
| Borrow $42.804 | $+42.804$ | $-45$ | $-45$ |
| **Net** | $-7.196$ | $+15$ | $0$ |

**Why $B(0) = 1$?** This is a normalization convention — we choose the bond as the numeraire. Investing \$1 today grows to $e^{rT}$. Changing this convention changes $\phi_1$ numerically but not the final price.

---

### Method 2: Risk-Neutral Pricing

**Principle**: There exists a probability measure $Q$ (the risk-neutral measure) under which all discounted asset prices are martingales. Pricing any claim reduces to computing a discounted expectation under $Q$.

**Step 1** — Find $q$ from the stock's martingale condition:

$$S(0) = e^{-rT} \mathbb{E}^Q[S(T)] = e^{-rT}[q \cdot S_u + (1-q) \cdot S_d]$$

$$100 = e^{-0.05}[120q + 90(1-q)] \implies q = \frac{S(0)e^{rT} - S_d}{S_u - S_d} = \frac{105.127 - 90}{30} = 0.5043$$

**Step 2** — Price the call:

$$C(0) = e^{-rT}\mathbb{E}^Q[V] = e^{-0.05}[0.5043 \times 15 + 0.4957 \times 0] = 0.9512 \times 7.565 = \mathbf{7.196}$$

**Key insight**: $q$ is determined entirely by $S(0), S_u, S_d, r$. The true probability $p_u = 0.6$ plays no role. The risk-neutral measure absorbs investors' risk preferences, so pricing requires no knowledge of beliefs about the future.

**Why $q \neq p_u$?** Because investors are risk-averse. They require a premium for holding risky assets, which "tilts" the true probabilities. The risk-neutral measure adjusts for this tilt, making all assets earn the risk-free rate on average.

---

### Method 3: Arrow-Debreu State Prices

**Principle**: An Arrow-Debreu (AD) security pays \$1 in exactly one state and \$0 in all others. $\psi_i$ is today's price of the AD security for state $i$. Any claim is a portfolio of AD securities, so its price is the inner product $\psi^T V$.

**AD securities in our model:**

$$e_u = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \quad \text{pays \$1 if up}, \qquad e_d = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \quad \text{pays \$1 if down}$$

**Step 1** — Solve $D^T \psi = p_0$:

$$\begin{pmatrix} e^{rT} & e^{rT} \\ S_u & S_d \end{pmatrix} \begin{pmatrix} \psi_u \\ \psi_d \end{pmatrix} = \begin{pmatrix} 1 \\ 100 \end{pmatrix}$$

From the first row: $\psi_u + \psi_d = e^{-rT} = 0.9512$

Substituting into the second row: $120\psi_u + 90\psi_d = 100$

Solving:

$$\psi_u = \frac{100 - 90 \times 0.9512}{30} = \frac{14.39}{30} = 0.4797$$

$$\psi_d = 0.9512 - 0.4797 = 0.4715$$

**Step 2** — Price the call:

$$C(0) = \psi_u V_u + \psi_d V_d = 0.4797 \times 15 + 0.4715 \times 0 = \mathbf{7.196}$$

**Relationship to risk-neutral probabilities:**

$$\psi_i = e^{-rT} q_i \implies q_u = \frac{\psi_u}{e^{-rT}} = \frac{0.4797}{0.9512} = 0.5043 \quad \checkmark$$

AD prices are the "all-in-one" objects: they encode both time discounting ($e^{-rT}$) and state probability ($q_i$) in a single number.

**No-arbitrage condition:** $\psi_i > 0$ for all $i$. If any $\psi_i \leq 0$, arbitrage exists (you can buy a claim that pays something for free or less than nothing).

---

### Method 4: Stochastic Discount Factor (SDF)

**Principle**: Introduce a random variable $m$ (the pricing kernel or stochastic discount factor) such that pricing works under the **true** probability $P$:

$$C(0) = \mathbb{E}^P[m \cdot V]$$

**Construction**: $m_i = \dfrac{\psi_i}{p_i}$ — AD price divided by true probability.

$$m_u = \frac{\psi_u}{p_u} = \frac{0.4797}{0.6} = 0.7995, \qquad m_d = \frac{\psi_d}{p_d} = \frac{0.4715}{0.4} = 1.1788$$

**Pricing:**

$$C(0) = p_u m_u V_u + p_d m_d V_d = 0.6 \times 0.7995 \times 15 + 0.4 \times 1.1788 \times 0 = \mathbf{7.196}$$

**Interpretation of $m_d > m_u$:**

$$m_d = 1.1788 > 1 > 0.7995 = m_u$$

The SDF is **higher in bad states** (down) than in good states (up). This encodes risk aversion: a dollar in a bad state is worth more than a dollar in a good state. The SDF is the marginal utility of wealth normalized by the price of consumption.

**Relationship to risk-neutral measure:**

$$m_i = e^{-rT} \frac{q_i}{p_i} = e^{-rT} \frac{dQ}{dP}(\omega_i)$$

The SDF is the Radon-Nikodym derivative $dQ/dP$ scaled by the discount factor. In continuous time, this becomes the density process that drives the Girsanov change of measure.

**True expected value vs. risk-adjusted value:**

$$e^{-rT}\mathbb{E}^P[V] = e^{-0.05}(0.6 \times 15 + 0.4 \times 0) = 8.56 > 7.196$$

The call is worth less than the discounted true expectation because the call pays off in the *good* state (where $m$ is small). Risk-averse investors "discount" good-state payoffs.

---

### Method 5: Linear Programming Duality

**Principle**: No-arbitrage pricing is equivalent to a linear program. The price of any replicable claim is the unique solution where the LP's primal and dual bounds coincide (Strong Duality).

**Primal** (super-replication from below, buyer's perspective):

$$\underline{C} = \min_\phi \; \phi^T p_0 \quad \text{subject to} \quad D\phi \geq V$$

This finds the cheapest portfolio that dominates $V$ in every state.

**Dual** (state price approach, seller's perspective):

$$\underline{C} = \min_\psi \; \psi^T V \quad \text{subject to} \quad D^T\psi = p_0, \quad \psi \geq 0$$

**Strong Duality Theorem**: Primal optimal = Dual optimal, and the dual optimum is achieved by the AD prices $\psi$ from Method 3.

In a **complete market**: the primal constraint $D\phi \geq V$ is binding at the optimum ($D\phi = V$), recovering exact replication from Method 1. The price interval collapses to a single point:

$$\underline{C} = \overline{C} = 7.196$$

In an **incomplete market** (more states than securities, $m > n$): the constraint cannot be satisfied with equality for every $V$. The primal-dual gap becomes a genuine price interval $[\underline{C}, \overline{C}]$, representing the range of prices consistent with no-arbitrage.

**Why this matters**: LP duality gives a rigorous foundation for no-arbitrage pricing. The dual variables $\psi \geq 0$ are exactly the state prices, and the dual feasibility constraint $D^T\psi = p_0$ is exactly the no-arbitrage condition. Duality says: arbitrage-free pricing from the buyer's side and state-price representation from the seller's side give the same answer.

---

### Method 6: Utility Indifference Pricing

**Principle**: Find the price $\pi^*$ at which a utility-maximizing investor is exactly indifferent between:
- (a) Not trading the call, investing optimally in available assets
- (b) Selling the call at $\pi^*$ and investing optimally to hedge

Uses the **CARA (Constant Absolute Risk Aversion)** utility function:

$$U(x) = -e^{-\gamma x}, \quad \gamma > 0$$

Properties: $U' > 0$ (monotone), $U'' < 0$ (concave, risk-averse), $-U''/U' = \gamma$ (constant risk aversion).

**Step 1 — Baseline (no call)**: Optimize stock position $\alpha$ under true probability $P$:

$$\max_\alpha \mathbb{E}^P[U(\alpha \cdot \text{excess return})]$$

where excess return $= S(T) - S(0)e^{rT}$ (gain over risk-free).

Analytic FOC for CARA:

$$\alpha^* = \frac{\ln\!\left(\dfrac{-p_d \cdot \text{excess}_d}{p_u \cdot \text{excess}_u}\right)}{\gamma(\text{excess}_u - \text{excess}_d)}$$

With our parameters: $\text{excess}_u = 14.873$, $\text{excess}_d = -15.127$:

$$\alpha^* = \frac{\ln\!\left(\frac{0.4 \times 15.127}{0.6 \times 14.873}\right)}{2 \times 30} \approx -0.0065 \approx 0$$

The baseline position is nearly zero because $p_u = 0.6 \approx q = 0.504$ — the stock is nearly fairly priced under $P$.

**Step 2 — Sell call + delta hedge**: With delta $\Delta = (V_u - V_d)/(S_u - S_d) = 0.5$, wealth becomes deterministic:

$$W = \pi - V(T) + \Delta \cdot \text{excess}(T) = \pi - 7.5636 \quad \text{(constant!)}$$

**Step 3 — Indifference equation**:

$$U(\pi^* - 7.5636) = \mathbb{E}^P[U(\alpha^* \cdot \text{excess})]$$

$$-e^{-\gamma(\pi^* - 7.5636)} = \text{EU}_{\text{baseline}}$$

$$\pi^* = 7.5636 - \frac{\ln(-\text{EU}_{\text{baseline}})}{\gamma} = 7.536$$

**Key finding — why $\pi^* \neq C_{\text{na}} = 7.196$:**

The gap $\pi^* - C_{\text{na}} = 0.34$ exists because:

1. Baseline: investor holds $\alpha^* \approx 0$ (nearly no stock, based on true $P$)
2. Selling call forces delta hedge: $\alpha = 0.5$
3. This disrupts the investor's preferred portfolio → demands premium

**Sensitivity to $\gamma$:**

| $\gamma$ | $\pi^*$ | Gap vs $C_{\text{na}}$ |
|---|---|---|
| 0.01 | 7.194 | $\approx 0$ |
| 0.1 | 7.016 | $-0.18$ |
| 0.5 | 7.454 | $+0.26$ |
| 2.0 | 7.536 | $+0.34$ |
| 10.0 | 7.558 | $+0.36$ |
| $\infty$ | 7.564 | $+0.37$ |

**Critical insight**: $\pi^* = C_{\text{na}}$ if and only if $p_u = q$ (true probability equals risk-neutral probability). When true and risk-neutral probabilities differ, utility indifference pricing gives a different result that depends on $\gamma$ and $p$.

In **complete markets**, as $\gamma \to 0$ (risk-neutral limit), $\pi^* \to C_{\text{na}}$. In **incomplete markets**, $\pi^*$ is the unique internally consistent price for a given investor type, and different investors assign different prices.

---

## 4. Why All Methods Agree: The Deep Equivalence

### The Core Identity

All six methods compute the same quantity from different angles:

$$C(0) = \psi^T V$$

This is a linear functional — price is a weighted sum of payoffs across states, with AD prices as weights.

### Algebraic Proof

**Method 1** gives $\phi = D^{-1}V$, so:

$$C(0) = \phi^T p_0 = (D^{-1}V)^T p_0 = V^T (D^{-1})^T p_0$$

**Method 3** gives $\psi = (D^T)^{-1} p_0$, so:

$$C(0) = \psi^T V = [(D^T)^{-1}p_0]^T V = p_0^T D^{-1} V$$

Since $(D^{-1})^T = (D^T)^{-1}$:

$$V^T (D^{-1})^T p_0 = p_0^T D^{-1} V \quad \checkmark$$

The equality is a matrix transpose identity — it holds exactly, not approximately.

### The Chain of Equivalences

Starting from Method 1 ($D\phi = V$, Method 3 $D^T\psi = p_0$):

$$\underbrace{\phi^T p_0}_{\text{Method 1}} = \phi^T (D^T \psi) = (D\phi)^T \psi = V^T \psi = \underbrace{\psi^T V}_{\text{Method 3}}$$

The middle step uses $D\phi = V$ (replication condition) and $D^T\psi = p_0$ (no-arbitrage condition). These two equations are **dual** to each other — one says we can replicate, the other says prices are consistent.

### The Primal-Dual Correspondence

$$\underbrace{D\phi = V}_{\text{Primal: replication}} \quad \xleftrightarrow{\text{duality}} \quad \underbrace{D^T\psi = p_0}_{\text{Dual: no-arbitrage}}$$

| Primal (replication) | Dual (pricing) |
|---|---|
| Unknown: $\phi$ (portfolio) | Unknown: $\psi$ (state prices) |
| Constraint: $D\phi = V$ | Constraint: $D^T\psi = p_0$ |
| Objective: minimize $\phi^T p_0$ | Objective: minimize $\psi^T V$ |
| Solution: $\phi = D^{-1}V$ | Solution: $\psi = (D^T)^{-1}p_0$ |
| **Optimal value: $C(0)$** | **Optimal value: $C(0)$** |

---

## 5. The Fundamental Theorem of Asset Pricing (FTAP)

### Statement (Discrete Time)

The following four conditions are **equivalent**:

$$\underbrace{\text{No Arbitrage}}_{\nexists\, \phi:\, D\phi \geq 0,\, \phi^T p_0 < 0}
\iff
\underbrace{\exists\, \psi \gg 0 \text{ s.t. } D^T\psi = p_0}_{\text{Positive state prices}}
\iff
\underbrace{\exists\, Q \sim P \text{ s.t. } S \text{ is a } Q\text{-martingale}}_{\text{Equivalent martingale measure}}
\iff
\underbrace{\exists\, m > 0 \text{ s.t. } p_0 = \mathbb{E}^P[mD]}_{\text{Positive SDF}}$$

### What Each Condition Means

**No Arbitrage**: You cannot construct a portfolio that costs nothing (or negative) today and pays non-negative (and sometimes positive) in every state.

**Positive state prices** ($\psi \gg 0$): Every state has a positive price — the market assigns positive value to insurance against any scenario. If any $\psi_i = 0$, you could buy a claim paying in state $i$ for free, creating arbitrage.

**Equivalent martingale measure** ($Q \sim P$): A probability measure that (i) agrees with $P$ on which events are possible ($Q \sim P$, they share the same null sets), and (ii) makes all discounted asset prices martingales ($\mathbb{E}^Q[e^{-rT}S(T)] = S(0)$).

**Positive SDF** ($m > 0$): A strictly positive random variable that prices all assets correctly under the true measure: $p_j = \mathbb{E}^P[m \cdot d_j]$ for all securities $j$.

### Market Completeness (Second Fundamental Theorem)

$$\text{Market Complete} \iff \text{The EMM } Q \text{ is unique} \iff \psi \text{ is unique} \iff \text{rank}(D) = m$$

In our 2×2 model: one unique $Q$ → one unique $\psi$ → one unique price for every claim.

In incomplete markets: infinitely many $Q$'s → infinitely many $\psi$'s → price interval $[\underline{C}, \overline{C}]$.

### Verification in Our Model

From Exercise 15:

$$\psi_u = 0.3781 > 0, \quad \psi_d = 0.6070 > 0 \quad \checkmark \text{ (No arbitrage)}$$

$$q_u = 0.3837 \in (0,1), \quad q_d = 0.6163 \in (0,1) \quad \checkmark \text{ (Valid EMM)}$$

$$m_u = 0.7561 > 0, \quad m_d = 1.2143 > 0 \quad \checkmark \text{ (Positive SDF)}$$

---

## 6. Arrow-Debreu Prices: Deep Dive

### What They Represent

$\psi_i$ is the price today of a security that pays exactly \$1 if state $i$ occurs and \$0 otherwise. They are the "atoms" of the pricing system — once you know all AD prices, you can price anything.

### Three-Way Decomposition

$$\psi_i = \underbrace{e^{-rT}}_{\text{time value}} \times \underbrace{q_i}_{\text{risk-neutral prob}} = \underbrace{p_i}_{\text{true prob}} \times \underbrace{m_i}_{\text{SDF}}$$

Each decomposition is a different way to separate the two things AD prices capture: **time value of money** and **state-contingent value**.

### Properties

1. **Positivity**: $\psi_i > 0 \iff$ no arbitrage
2. **Summing to discount factor**: $\sum_i \psi_i = e^{-rT}$ (price of risk-free bond)
3. **Linearity**: $\text{Price}(V) = \psi^T V$ (linear pricing functional)
4. **Uniqueness**: $\psi$ unique $\iff$ market complete

### Numerical Values (Exercise 15 with ATM Put)

| | $\psi_u$ | $\psi_d$ |
|---|---|---|
| AD price | 0.3781 | 0.6070 |
| Risk-neutral prob $q_i$ | 0.3837 | 0.6163 |
| True prob $p_i$ | — | — |
| SDF $m_i$ | 0.7561 | 1.2143 |

$$\psi_u + \psi_d = 0.9851 = e^{-0.06 \times 0.25} \quad \checkmark$$

### The Law of One Price

AD prices enforce the Law of One Price (LOP):

$$D\phi^{(1)} = D\phi^{(2)} \implies (\phi^{(1)})^T p_0 = (\phi^{(2)})^T p_0$$

Two portfolios with identical payoffs must have identical prices. LOP is weaker than no-arbitrage (no-arb implies LOP, but not vice versa). AD prices exist under LOP; positive AD prices require no-arbitrage.

---

## 7. Connection to Continuous Time

The one-period model is the foundation for continuous-time finance. Every concept generalizes:

| Discrete (One-Period) | Continuous (Black-Scholes) |
|---|---|
| States $\omega_1, \ldots, \omega_m$ | Paths $\omega \in C[0,T]$ |
| Payoff matrix $D \in \mathbb{R}^{m \times n}$ | Payoff functional on path space |
| State prices $\psi_i$ | State price density $\psi(\omega)$ |
| EMM $Q$ with weights $q_i$ | EMM $Q$ (equivalent martingale measure) |
| $\psi_i = e^{-rT} q_i$ | $\psi(\omega) = e^{-rT} \frac{dQ}{dP}(\omega)$ |
| SDF $m_i = \psi_i/p_i$ | SDF $m_t = e^{-rt} \frac{dQ}{dP}\big\vert_{\mathcal{F}_t}$ |
| Change $p \to q$: divide by $m_i$ | Girsanov theorem: $dW^P = dW^Q + \lambda\, dt$ |
| $C(0) = e^{-rT}\mathbb{E}^Q[V]$ | $C(0) = e^{-rT}\mathbb{E}^Q[V]$ (same formula!) |

### Girsanov Theorem as Change of Measure

In the Black-Scholes model, the stock follows under $P$:

$$dS = \mu S\, dt + \sigma S\, dW^P$$

Under $Q$ (the EMM), the drift changes to the risk-free rate:

$$dS = rS\, dt + \sigma S\, dW^Q$$

The Brownian motion changes: $dW^Q = dW^P + \lambda\, dt$ where $\lambda = (\mu - r)/\sigma$ is the **market price of risk**. This is the continuous-time analog of replacing $p_i$ with $q_i$.

The Radon-Nikodym derivative (continuous-time SDF):

$$\frac{dQ}{dP} = \exp\!\left(-\lambda W^P_T - \frac{\lambda^2}{2}T\right)$$

This is exactly $m_T / m_0$ — the SDF evaluated at maturity.

---

## 8. Summary Tables

### The Six Methods at a Glance

| Method | Core Object | Pricing Formula | Needs True $P$? | Needs $\gamma$? | Market |
|---|---|---|---|---|---|
| 1. Replicating portfolio | $\phi \in \mathbb{R}^n$ | $\phi^T p_0$ | No | No | Complete |
| 2. Risk-neutral pricing | $Q$ (EMM) | $e^{-rT}\mathbb{E}^Q[V]$ | No | No | Complete |
| 3. Arrow-Debreu | $\psi \in \mathbb{R}^m$ | $\psi^T V$ | No | No | Complete |
| 4. SDF / Pricing kernel | $m > 0$ | $\mathbb{E}^P[mV]$ | Yes (but $m$ absorbs it) | No | Complete/Incomplete |
| 5. LP duality | $[\underline{C}, \overline{C}]$ | Price bounds | No | No | Both |
| 6. Utility indifference | $U, \gamma$ | $\pi^*$ from indiff. eq. | Yes | Yes | Both |

### Key Results: Exercise 15 (ATM Put, $K=50$)

| Quantity | Value |
|---|---|
| $B(T) = e^{rT}$ | 1.01511 |
| $\det(D)$ | $-15.227$ |
| Risk-neutral prob $q_u$ | 0.3837 |
| Risk-neutral prob $q_d$ | 0.6163 |
| AD price $\psi_u$ | 0.3781 |
| AD price $\psi_d$ | 0.6070 |
| Delta $\Delta = \phi_2$ | $-1/3$ |
| Bond position $\phi_1$ | $19.702$ |
| Put fair value $P(0)$ | **3.036** |

### Key Results: General Example (ATM Call, $K=105$)

| Quantity | Value |
|---|---|
| $B(T) = e^{rT}$ | 1.05127 |
| Risk-neutral prob $q$ | 0.5043 |
| AD price $\psi_u$ | 0.4797 |
| AD price $\psi_d$ | 0.4715 |
| Delta $\Delta = \phi_2$ | 0.5 |
| Bond position $\phi_1$ | $-42.804$ |
| SDF $m_u$ | 0.7995 |
| SDF $m_d$ | 1.1788 |
| Call fair value $C(0)$ | **7.196** |
| Utility indifference $\pi^*$ ($\gamma=2$) | **7.536** |

---

