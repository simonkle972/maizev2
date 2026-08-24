# Conceptual-question probe — combined corpus vs books-only

_2026-08-17T04:03:43Z — 18 questions x 2 corpora_

**combined** = professor's material + books (`EgZ14pvqEYzfQRTM`) · **books-only** = books alone (`tLMAxWBRrycrJsWl`), the control

The question this answers: on conceptual ground that BOTH corpora cover, does the combined TA answer from the professor's material or from the textbooks — and does the combined answer read like his treatment or like the book's?

**Book sources appeared in 17/18 combined-corpus answers.**

| # | question | combined sources | book? |
|---|---|---|---|
| 1 | What does OLS actually do — what is it minimising? | Interactive Lecture 9 slides-1, URfIE_web, 625807794-Nick-Huntington-K | YES |
| 2 | What are the assumptions behind OLS and which ones matter mo | URfIE_web, Interactive Lecture 7-1, 625807794-Nick-Huntington-Klein-Th | YES |
| 3 | What is omitted variable bias and how do I know if I have it | Interactive Lecture 9 slides-1, Pre-recorded lecture 08-1 | no |
| 4 | What is heteroskedasticity and why is it a problem? | 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web | YES |
| 5 | When should I use robust standard errors instead of regular  | 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web | YES |
| 6 | What does endogeneity mean and why does it break OLS? | Interactive Lecture 9 slides-1, 625807794-Nick-Huntington-Klein-The-Ef | YES |
| 7 | How does an instrumental variable fix endogeneity? What make | URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021,  | YES |
| 8 | What does R² actually tell me, and why shouldn't I just maxi | 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web | YES |
| 9 | How do I interpret a p-value on a regression coefficient? | 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web | YES |
| 10 | What is multicollinearity and does it always need fixing? | 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web | YES |
| 11 | How do I interpret a dummy variable coefficient? | Pre-recorded lecture 07-1, URfIE_web | YES |
| 12 | What does an interaction term mean in a regression? | URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021 | YES |
| 13 | How do I interpret coefficients when the dependent variable  | how_to_intepret_beta_coefficients-1, Pre-recorded lecture 07-1, URfIE_ | YES |
| 14 | What is the difference between correlation and causation in  | Pre-recorded lecture 05 -2025-1, Pre-recorded lecture 08-1, 625807794- | YES |
| 15 | What are fixed effects and when do I need them? | URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021 | YES |
| 16 | How does difference-in-differences identify a causal effect? | Pre-recorded lecture 09-1, Interactive Lecture 9 slides-1, URfIE_web | YES |
| 17 | What is the difference between a standard error and a standa | Interactive Lecture 5-1, 625807794-Nick-Huntington-Klein-The-Effect-Ro | YES |
| 18 | How do I read a confidence interval on a coefficient? | Pre-recorded lecture 06-1, Pre-recorded lecture 07-1, URfIE_web | YES |

---

## 1. What does OLS actually do — what is it minimising?

### combined

*sources: Interactive Lecture 9 slides-1, URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 21599ms + 5817ms_

OLS chooses the coefficients that MINIMIZE the sum of squared residuals (prediction errors).

In a regression with $K$ regressors, it solves:

$$
(\hat\beta_0,\hat\beta_1,\ldots,\hat\beta_K)
=
\arg\min_{b_0,\ldots,b_K}\sum_{i=1}^n\left(Y_i-\left(b_0+b_1X_{1i}+b_2X_{2i}+\cdots+b_KX_{Ki}\right)\right)^2
$$

The residual for observation $i$ is

$$
e_i = Y_i-\hat Y_i
$$

so the objective is $\sum_{i=1}^n e_i^2$. Squaring means big mistakes get penalized a lot more than small ones, which is why OLS can be sensitive to outliers (compared to LAD, which minimizes $\sum_i |e_i|$ and targets the conditional median instead of the conditional mean).

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 22115ms + 4840ms_

OLS chooses the coefficients in your line (or regression function) to MINIMIZE the sum of squared residuals.

In the simple linear regression

$$
Y_i = \beta_0 + \beta_1 X_i + u_i,
$$

OLS picks $(\beta_0,\beta_1)$ to minimize the objective function

$$
\min_{\beta_0,\beta_1}\ \sum_{i=1}^n \left(Y_i - (\beta_0+\beta_1 X_i)\right)^2.
$$

The term in parentheses is the residual for observation $i$:

$$
\hat u_i = Y_i - \hat Y_i,\ \ \hat Y_i=\beta_0+\beta_1 X_i.
$$

So “what OLS does” is: among all possible lines, it picks the one that makes the squared vertical gaps between the observed points $Y_i$ and the fitted values $\hat Y_i$ as small as possible in total.

---

## 2. What are the assumptions behind OLS and which ones matter most in practice?

### combined

*sources: URfIE_web, Interactive Lecture 7-1, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 20318ms + 13525ms_

Start from the population model you’re fitting:

$$y_i=\beta_0+\beta_1 x_i+u_i$$

(or in multiple regression $y_i=\beta_0+\beta_1 x_{1i}+\cdots+\beta_k x_{ki}+u_i$).

Core OLS assumptions (Wooldridge’s SLR.1 to SLR.5):
1. SLR.1 LINEAR POPULATION REGRESSION FUNCTION: $y=\beta_0+\beta_1 x+u$
2. SLR.2 RANDOM SAMPLING of $(x_i,y_i)$
3. SLR.3 SAMPLE VARIATION IN $x$: not all $x_i$ are the same
4. SLR.4 ZERO CONDITIONAL MEAN: $E(u\mid x)=0$ (in multiple regression: $E(u\mid X)=0$)
5. SLR.5 HOMOSCEDASTICITY: $\mathrm{Var}(u\mid x)=\sigma^2$

Extra “classical” assumption for exact small-sample inference (Wooldridge MLR.6 / CLM):
6. NORMALITY: roughly $u\mid X\sim N(0,\sigma^2)$, which gives exact $t$ and $F$ distributions in finite samples.

Which matter most in practice:
1. SLR.4 $E(u\mid X)=0$ is usually the MOST IMPORTANT, because violations (omitted variables, simultaneity, bad controls) bias $\hat\beta$.
2. SLR.5 homoscedasticity mostly matters for STANDARD ERRORS and tests, not for unbiasedness; if it fails you typically switch to heteroskedasticity-robust standard errors (as your notes emphasize in the heteroskedasticity section).
3. Normality is mainly about small-sample exact inference; with large $n$, asymptotics often makes inference approximately valid even without normal errors (your Chapter 5 notes).

What’s the setting you have in mind (cross-section vs time series, and what’s $y$ and the key $x$)? That determines whether random sampling and $E(u\mid X)=0$ are plausible.

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 17718ms + 10486ms_

In Wooldridge’s setup, OLS is built on a population model like

$$
y=\beta_0+\beta_1 x+u
$$

(or with multiple regressors, $y=\beta_0+\beta_1 x_1+\dots+\beta_k x_k+u$), plus assumptions about how $(x,y,u)$ behave.

1. Linearity in parameters (SLR.1 / MLR.1): the conditional mean is correctly written as a linear function of regressors plus an error.
2. Random sampling (SLR.2): observations are an i.i.d. sample from the population.
3. Variation in regressors (SLR.3): $x$ is not constant in the sample (multiple regression analog is “no perfect collinearity”).
4. Zero conditional mean (SLR.4): 

$$
E(u\mid x)=0 \quad \text{(or }E(u\mid x_1,\dots,x_k)=0\text{)}
$$

This is the key “exogeneity” condition.
5. Homoscedasticity (SLR.5 / MLR.5):

$$
\operatorname{Var}(u\mid x_1,\dots,x_k)=\sigma^2
$$

6. Normality (MLR.6): $u$ is normal conditional on regressors, giving exact finite-sample normal/t distributions for test statistics (the CLM result).

Which matter most in practice: SLR.4 is the big one for causal interpretation and unbiasedness; SLR.2 and regressor variation are basic necessities to even make the problem well-posed. Homoscedasticity matters mainly for getting the usual (non-robust) standard errors and efficiency; if it fails, OLS coefficients can still be unbiased/consistent but inference based on the usual SEs becomes unreliable (so you switch to heteroskedasticity-robust SEs). Normality is usually the least important because large-sample (asymptotic) arguments often justify approximate inference without it.

---

## 3. What is omitted variable bias and how do I know if I have it?

### combined

*sources: Interactive Lecture 9 slides-1, Pre-recorded lecture 08-1*  ·  _8 chunks, v2_hybrid_rrf, 20843ms + 7927ms_

Omitted variable bias (OVB) is when the coefficient you estimate on an included regressor is pulled away from the true causal effect because you left out a relevant variable that both affects $Y$ and is correlated with your included regressor.

Write the true model as

$$Y_i=\beta_0+\beta_1 X_{1i}+\beta_2 X_{2i}+e_i$$

If you omit $X_2$ and instead estimate

$$Y_i=\beta_0+\beta_1 X_{1i}+u_i$$

then the error term becomes

$$u_i=\beta_2 X_{2i}+e_i$$

The key result is

$$\text{plim }\hat{\beta}_1=\beta_1+\beta_2\frac{\text{Cov}(X_{2i},X_{1i})}{\text{Var}(X_{1i})}$$

So you have NO OVB if either (1) $\beta_2=0$ (the omitted variable doesn’t affect $Y$) or (2) $\text{Cov}(X_2,X_1)=0$ (the omitted variable is unrelated to $X_1$). You “know” you might have it by asking: is there a plausible omitted determinant of $Y$ that’s also related to your $X_1$?

What’s your specific regression (what is $Y$ and what is $X_1$)? Name one plausible $X_2$ you’re worried you omitted, and tell me whether you think it’s positively or negatively correlated with $X_1$.

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 15715ms + 11434ms_

Omitted variable bias is the failure of the ceteris paribus interpretation in multiple regression: you think $x_1$’s coefficient is “holding other causes fixed,” but an important cause is missing and ends up inside the error term, making $x_1$ correlated with the error.

Write the “true” model as

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + u
$$

If you omit $x_2$ and run a regression of $y$ on $x_1$ only, the omitted-variable-bias formula is

$$
E[\tilde\beta_1] = \beta_1 + \beta_2 \frac{\operatorname{Cov}(x_1,x_2)}{\operatorname{Var}(x_1)}
$$

So bias shows up when BOTH are true: (1) the omitted variable matters for $y$ (so $\beta_2 \neq 0$) and (2) it is correlated with the included regressor (so $\operatorname{Cov}(x_1,x_2)\neq 0$), which is exactly the exogeneity failure $x_1$ correlated with the error.

You generally cannot “test” for omitted variable bias directly, because you don’t observe the omitted variable; you diagnose it by arguing about the data-generating process (or a causal diagram) and asking what plausible $x_2$ you left out that affects $y$ and is related to your regressor of interest.

What is your $x_1$ (the effect you care about) and what is one plausible omitted $x_2$ that might both affect $y$ and be correlated with $x_1$?

---

## 4. What is heteroskedasticity and why is it a problem?

### combined

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 19918ms + 4566ms_

Heteroskedasticity means the conditional variance of the regression error changes with the regressors. In the homoskedastic case, the assumption is

$$\mathrm{Var}(u \mid x_1,\ldots,x_k)=\sigma^2$$

Heteroskedasticity is when instead

$$\mathrm{Var}(u \mid x_1,\ldots,x_k) \neq \sigma^2 \text{ and depends on } x$$

Why it’s a problem: OLS coefficient estimates can still be unbiased and consistent, but the USUAL OLS standard errors are wrong under heteroskedasticity, so $t$ tests, $F$ tests, confidence intervals, and p-values become unreliable. It also breaks the usual efficiency result for OLS (you might be able to do better than OLS with something like WLS). The common fix is to keep OLS coefficients but use heteroskedasticity-robust (Huber-White “sandwich”) standard errors for valid inference.

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 17506ms + 7414ms_

Heteroskedasticity means the error variance changes with the regressors. The homoskedasticity assumption in SLR/MLR is

$$\operatorname{Var}(u \mid x_1,\ldots,x_k)=\sigma^2$$

Heteroskedasticity is when instead

$$\operatorname{Var}(u \mid x_1,\ldots,x_k)=\sigma^2(x_1,\ldots,x_k)$$

It’s a problem mainly because the usual OLS standard error formulas assume a single common $\sigma^2$, so when variance is higher in some regions of $X$, $\hat\beta$ varies more across samples than the conventional SE formula accounts for. Result: t-tests, F-tests, and confidence intervals based on the usual SEs become unreliable, and OLS is no longer (asymptotically) efficient. Importantly, unbiasedness and consistency of OLS coefficients do NOT depend on homoskedasticity; it’s the inference that breaks unless you use heteroskedasticity-robust (“sandwich,” e.g., Huber-White) standard errors.

---

## 5. When should I use robust standard errors instead of regular ones?

### combined

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 18749ms + 4923ms_

Use “regular” OLS standard errors when you’re willing to assume homoskedasticity, i.e.

$$\mathrm{Var}(u\mid X)=\sigma^2$$

Use heteroskedasticity-robust (Huber-White) standard errors when you think the error variance changes with $X$ (or with fitted values), i.e.

$$\mathrm{Var}(u\mid X)=\sigma^2(X)$$

OLS coefficients $\hat\beta$ don’t change, but the usual SE formula relies on the constant-variance assumption, so your $t$ tests and confidence intervals can be off if that assumption fails; robust SE replace it with a “sandwich” variance estimator.

First diagnostic question: in your data, is there a reason groups with different $X$ levels (or different subpopulations) would naturally have different outcome variability, even after controlling for $X$?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 19467ms + 10432ms_

“Regular” OLS standard errors are valid only under the homoskedasticity assumption

$$\mathrm{Var}(u\mid X)=\sigma^2 I,$$

which implies

$$\mathrm{Var}(\hat\beta\mid X)=\sigma^2 (X'X)^{-1}.$$

If you think $\mathrm{Var}(u\mid X)$ is NOT constant across observations (heteroskedasticity), switch to heteroskedasticity-robust (Huber-White) standard errors, which use the sandwich form

$$\widehat{\mathrm{Var}}(\hat\beta\mid X)=(X'X)^{-1}X'\hat\Omega X(X'X)^{-1}.$$

In panel data or grouped data, if you think errors are correlated within unit (same person/firm/county over time), use CLUSTERED standard errors (a version of the sandwich that allows arbitrary within-cluster correlation and heteroskedasticity).

What’s your data structure: pure cross-section, or panel/grouped (so you’d worry about within-unit correlation), and what makes you suspect heteroskedasticity?

---

## 6. What does endogeneity mean and why does it break OLS?

### combined

*sources: Interactive Lecture 9 slides-1, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, final-fall-2018-solutions-1*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021**  ·  _8 chunks, v2_hybrid_rrf, 18998ms + 6527ms_

Endogeneity means your regressor $X$ is correlated with the regression error term $\varepsilon$.

In the baseline OLS model

$$
Y=\beta_0+\beta_1X+\varepsilon,
$$

the key “exogeneity” condition is

$$
E[\varepsilon\mid X]=0 \quad \text{(equivalently } \operatorname{Cov}(X,\varepsilon)=0\text{)}.
$$

If instead $X$ is endogenous, then $E[\varepsilon\mid X]\neq 0$, so variation in $X$ is systematically tied to “other stuff” in $\varepsilon$ that also moves $Y$. OLS then can’t separate the causal effect of $X$ on $Y$ from the influence of those omitted/unobserved factors, so the OLS estimate of $\beta_1$ is biased (and typically inconsistent).

Two common ways this happens (from the lecture slide recap) are:
1. Omitted variable bias: some unobserved variable affects both $X$ and $Y$, and because it’s omitted it sits inside $\varepsilon$, creating $\operatorname{Cov}(X,\varepsilon)\neq 0$.
2. Simultaneity: $X$ affects $Y$ but $Y$ also affects $X$, so $X$ is partly “determined within the system,” again making it correlated with $\varepsilon$.

Pick one concrete example from your notes (like education and earnings, or price and quantity) and tell me what you think is inside $\varepsilon$ that might be correlated with $X$—that’s the fastest way to diagnose endogeneity.

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 13568ms + 7061ms_

Endogeneity in regression means your regressor of interest $X$ is correlated with the error term $\varepsilon$ (equivalently, $X$ is “determined from within the system” in a way that shares unobserved causes with $Y$). The OLS identifying assumption is exogeneity, typically written as

$$
E[\varepsilon \mid X]=0
\quad\text{(or)}\quad
\operatorname{Cov}(X,\varepsilon)=0.
$$

Why it breaks OLS: if $X$ and $\varepsilon$ are correlated, OLS attributes to $X$ some of the variation in $Y$ that is actually coming from whatever is inside $\varepsilon$ (an omitted confounder, simultaneity/reverse causality, measurement error, selection, etc.). In the simple regression $Y=\beta_0+\beta_1 X+\varepsilon$, you can see the problem in the probability limit:

$$
\operatorname{plim}\,\hat\beta_1
=
\beta_1+\frac{\operatorname{Cov}(X,\varepsilon)}{\operatorname{Var}(X)}.
$$

So when $\operatorname{Cov}(X,\varepsilon)\neq 0$, the estimator converges to the wrong value even with huge samples: that is bias/inconsistency. In causal-diagram language (as in the book), endogeneity is what happens when a backdoor path from $X$ to $Y$ remains open because a variable that affects both $X$ and $Y$ is sitting in $\varepsilon$ instead of being controlled for.

---

## 7. How does an instrumental variable fix endogeneity? What makes a good instrument?

### combined

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, econ117_s2019_final_v7_suggested_solutions-1*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 20233ms + 9408ms_

IV “fixes” endogeneity by replacing the problematic variation in $X$ (the part correlated with the error) with variation in $X$ that comes from an outside source $Z$.

The key idea is the moment condition:

$$E\left[Z\left(Y-\beta_0-\beta_1 X\right)\right]=0$$

This works if $Z$ is a GOOD instrument, meaning:
1. Relevance: $Cov(Z,X)\neq 0$ (the instrument actually moves the treatment)
2. Exclusion: $Cov(Z,\varepsilon)=0$ (no path from $Z$ to $Y$ except through $X$, i.e., $Z$ is as-good-as-random with respect to unobservables in $\varepsilon$)

First concrete check: in an example you care about, what candidate $Z$ could you use, and what’s your argument that it affects $X$ but cannot affect $Y$ directly (other than through $X$)?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 15416ms + 6809ms_

Endogeneity means the regressor you care about, $X$, violates the exogeneity condition

$$\operatorname{Cov}(X,\varepsilon)=0$$

so OLS can’t isolate the causal effect because variation in $X$ is tangled up with $\varepsilon$ (omitted variables, reverse causality, etc.).

What two conditions would you want an instrument $Z$ to satisfy in terms of (1) its relationship with $X$ and (2) its relationship with $\varepsilon$?

---

## 8. What does R² actually tell me, and why shouldn't I just maximise it?

### combined

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 19416ms + 7455ms_

$R^2$ tells you how much of the variation in your dependent variable $y$ is “explained” by the fitted values $\hat y$ from your OLS regression.

Formally, with

$$SST=\sum_{i=1}^n (y_i-\bar y)^2,\quad SSR=\sum_{i=1}^n \hat u_i^2,\quad SSE=\sum_{i=1}^n (\hat y_i-\bar y)^2,$$

you can write

$$R^2=\frac{SSE}{SST}=1-\frac{SSR}{SST}=\frac{\mathrm{Var}(\hat y)}{\mathrm{Var}(y)}=1-\frac{\mathrm{Var}(\hat u)}{\mathrm{Var}(y)}.$$

Why not maximize it?

1. $R^2$ rewards adding variables even if they’re irrelevant: in OLS, adding any regressor cannot increase $SSR$, so $R^2$ mechanically (weakly) rises even if the new variable has nothing to do with the true data-generating process.

2. High $R^2$ is about prediction fit, not causal identification. A model can have a high $R^2$ and still give a biased estimate of the causal effect you care about (bad controls, omitted variables, reverse causality, etc.).

3. Maximizing $R^2$ can push you toward overfitting and “control for everything,” which can actually make your key coefficient less interpretable (or more biased) even while fit improves.

If you want a “fit measure that penalizes extra variables,” that’s the role of adjusted $R^2$, but even adjusted $R^2$ is still mainly about predictive fit, not about whether your coefficient answers the causal question.

One check for your own work: are you trying to PREDICT $y$ well, or IDENTIFY a particular causal coefficient? Which is the assignment asking for?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 17397ms + 9667ms_

$R^2$ is the fraction of the sample variation in $y$ that your OLS fitted values explain:

$$
R^2=\frac{\text{Var}(\hat y)}{\text{Var}(y)}=1-\frac{\text{Var}(\hat u)}{\text{Var}(y)}=1-\frac{SSR}{SST}
$$

So a low $R^2$ means “there’s a lot going on in $y$ beyond what’s in the model,” not automatically that the model is “bad” for your purpose. The reason not to maximize $R^2$ is that adding regressors mechanically pushes $R^2$ up (even irrelevant ones), but that can hurt what you usually care about in econometrics: identifying and precisely estimating a particular causal coefficient. A high $R^2$ can coexist with biased causal estimates if you control for the wrong things, and a low $R^2$ can coexist with a well-identified causal effect.

What is your goal in the assignment you’re thinking about: prediction of $y$, or estimating a causal effect of one regressor on $y$?

---

## 9. How do I interpret a p-value on a regression coefficient?

### combined

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 21063ms + 6472ms_

A p-value on a regression coefficient is the probability, ASSUMING the null hypothesis is true (usually $H_0:\beta_j=0$), of getting an estimate at least as far from 0 as the one you got.

In the usual two-sided test, it’s computed from the test statistic

$$
t = \frac{\hat\beta_j - 0}{se(\hat\beta_j)}
$$

and the p-value is

$$
p = \Pr\left(\lvert T\rvert \ge \lvert t\rvert \mid H_0 \text{ true}\right).
$$

If $p \le \alpha$ (like $\alpha=0.05$), you “reject” $H_0$ and call the coefficient statistically significant at that level.

When you see significance stars, they’re just p-value cutoffs (the exact mapping depends on the table note).

What’s the null hypothesis you’re using for the coefficient in your regression output, and what $\alpha$ level are you comparing the p-value to?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 18256ms + 6081ms_

A p-value for a regression coefficient is:

$$p = \Pr\left(\left|\hat\beta_j - \beta_{j,0}\right| \ge \left|\hat\beta_j^{obs} - \beta_{j,0}\right| \;\middle|\; H_0:\beta_j=\beta_{j,0}\right)$$

In the standard regression-table case, the null is $\beta_{j,0}=0$ and (approximately) you’re using a $t$ (often close to standard normal) sampling distribution for

$$t = \frac{\hat\beta_j - 0}{se(\hat\beta_j)}.$$

Interpretation: it’s the probability of getting an estimate at least as far from the null value (usually 0) as the one you got, IF the null were true.

Decision rule with a two-tailed test: reject $H_0$ at significance level $\alpha$ if $p \le \alpha$ (common choices: $\alpha=0.05$ or $0.01$). In regression tables, “significance stars” are just shorthand for ranges of p-values (the exact cutoffs depend on the table note).

One check to make sure you’re reading it right: are you thinking of the p-value as a probability about the DATA given the null (correct), or a probability the null is true (not what it is)?

---

## 10. What is multicollinearity and does it always need fixing?

### combined

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 19687ms + 7084ms_

Multicollinearity is when a regressor $x_j$ is strongly predicted by a linear combination of the other regressors (perfect multicollinearity means it’s predicted EXACTLY, and then OLS can’t estimate the model).

The key consequence is inflated standard errors for the collinear variables. One way to see it is:

$$
Var(\hat b_j)=\frac{s^2}{SST_j(1-R_j^2)}=\frac{1}{n}\cdot \frac{s^2}{Var(x_j)}\cdot \frac{1}{1-R_j^2}
$$

where $R_j^2$ is from regressing $x_j$ on the other predictors. This links directly to the variance inflation factor:

$$
VIF_j=\frac{1}{1-R_j^2}
$$

So it doesn’t “always” need fixing mechanically; it depends on what you’re trying to do. Are you mainly trying to interpret/identify the separate effect of one regressor (a “holding others fixed” causal story), or are you mainly trying to predict $Y$ well?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web*  ·  _8 chunks, v2_hybrid_rrf, 13787ms + 7673ms_

Multicollinearity is when one regressor is strongly linearly predicted by other regressors in the same model (high correlation among predictors). Perfect multicollinearity is the extreme case where a linear combination of regressors predicts another regressor EXACTLY, and then OLS can’t estimate the model because the coefficients aren’t uniquely identified.

High (but not perfect) multicollinearity mainly shows up as inflated standard errors, not bias. One way to see it is in the variance formula

$$
\mathrm{Var}(\hat b_j)=\frac{s^2}{\mathrm{SST}_j(1-R_j^2)}
=\frac{1}{n}\cdot\frac{s^2}{\mathrm{Var}(x_j)}\cdot\frac{1}{1-R_j^2},
$$

where $R_j^2$ is from regressing $x_j$ on the other regressors. Equivalently, the variance inflation factor is

$$
\mathrm{VIF}_j=\frac{1}{1-R_j^2}.
$$

So it does NOT always “need fixing”: it’s a problem if it makes inference too imprecise for your purpose (big SEs, unstable estimates), or if you’re “controlling away” the variation you need; it’s not automatically a model violation. What’s your main goal in the regression you’re thinking about: estimating one coefficient precisely (inference/causal effect) or maximizing predictive fit?

---

## 11. How do I interpret a dummy variable coefficient?

### combined

*sources: Pre-recorded lecture 07-1, URfIE_web*  ·  **book sources: URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 17759ms + 5860ms_

In a linear regression like

$$
Y_i = \beta_0 + \beta_1 D_i + \beta_2 X_i + u_i,
$$

where $D_i$ is a dummy (0 or 1), the coefficient $\beta_1$ is the difference in the predicted value of $Y$ when you “switch on” the dummy, holding the other regressors $X_i$ fixed.

Specifically:

$$
E[Y \mid D=1, X] - E[Y \mid D=0, X] = \beta_1.
$$

So $\beta_1$ shifts the intercept for the $D=1$ group relative to the $D=0$ (reference) group.

If you have multiple categories, you include $g-1$ dummies and leave one category out; each dummy’s coefficient is interpreted relative to that omitted reference category.

What’s the dummy in your problem (e.g., married, female, formal firm), and what outcome variable is on the left-hand side?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 16256ms + 10558ms_

In a linear regression, a dummy variable’s coefficient is an INTERCEPT SHIFT between the group coded 1 and the group coded 0.

Start with the simplest model:

$$
Y = \beta_0 + \beta_1 D + u
$$

where $D \in \{0,1\}$. Then the conditional means are

$$
E[Y \mid D=0] = \beta_0
\quad\text{and}\quad
E[Y \mid D=1] = \beta_0 + \beta_1
$$

so

$$
\beta_1 = E[Y \mid D=1] - E[Y \mid D=0].
$$

With controls,

$$
Y = \beta_0 + \beta_1 D + X'\gamma + u,
$$

the interpretation becomes “difference between $D=1$ and $D=0$ HOLDING $X$ FIXED”:

$$
\beta_1 = E[Y \mid D=1, X] - E[Y \mid D=0, X].
$$

If you have a categorical variable with $g>2$ categories, each dummy coefficient is interpreted RELATIVE TO THE OMITTED REFERENCE CATEGORY.

Which category is your reference (the one you left out)?

---

## 12. What does an interaction term mean in a regression?

### combined

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 16324ms + 9036ms_

An interaction term means you include the PRODUCT of two regressors to allow the effect of one to vary with the level of the other:

$$
y=\beta_0+\beta_1 x_1+\beta_2 x_2+\beta_3(x_1x_2)+u
$$

Here, $\beta_3$ is the coefficient on the interaction $x_1x_2$.

First step to interpret it: hold $x_2$ fixed and figure out the marginal effect of $x_1$ on $y$ (either by taking $\frac{\partial y}{\partial x_1}$ or by collecting the $x_1$ terms). What expression do you get for “the slope on $x_1$” as a function of $x_2$?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 16116ms + 6532ms_

An interaction term lets the effect of one regressor depend on the level of another regressor.

Write the model as

$$
Y=\beta_0+\beta_1X+\beta_2Z+\beta_3(XZ)+\varepsilon.
$$

Then the marginal effect of $X$ is not just $\beta_1$; it varies with $Z$:

$$
\frac{\partial Y}{\partial X}=\beta_1+\beta_3Z.
$$

So $\beta_3$ tells you how the effect of $X$ changes when $Z$ rises by 1 unit. If $Z$ is binary (0 or 1), then $\beta_1$ is the effect of $X$ when $Z=0$, and $\beta_1+\beta_3$ is the effect of $X$ when $Z=1$.

What are $X$ and $Z$ in your application, and which one do you think should change the effect of the other?

---

## 13. How do I interpret coefficients when the dependent variable is logged?

### combined

*sources: how_to_intepret_beta_coefficients-1, Pre-recorded lecture 07-1, URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 20924ms + 7263ms_

When the dependent variable is logged, your model looks like

$$\log(Y)=\beta_0+\beta_1 X+u.$$

Here $\beta_1$ is a SEMI-ELASTICITY: a one-unit increase in $X$ is associated with about a $100\beta_1$ percent change in $Y$ (this is the “small change” approximation from lecture: $\%\Delta Y \approx 100\Delta\log(Y)$).

If the regressor is also logged,

$$\log(Y)=\beta_0+\beta_1\log(X)+u,$$

then $\beta_1$ is an ELASTICITY: a 1 percent increase in $X$ is associated with a $\beta_1$ percent change in $Y$.

If $X$ is a dummy (0/1) in a log($Y$) model, the exact percent effect is

$$100\left(e^{\beta_1}-1\right)\%$$

rather than $100\beta_1\%$.

Which of these are you working with: $X$ in levels, $\log(X)$, or a dummy?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 13967ms + 5502ms_

When the dependent variable is logged, the coefficient tells you about a PERCENT change in $y$ (not a level change), holding other regressors fixed.

Write your model first:

$$\log(y)=\beta_0+\beta_1 x+u$$

In this semi-log model, $\beta_1$ is a SEMI-ELASTICITY: a 1-unit increase in $x$ is associated with an approximate $100\cdot \beta_1$ percent change in $y$. For a finite change $\Delta x$, the approximation is $100\cdot \beta_1\Delta x$ percent. (The exact percent change uses $\exp(\beta_1\Delta x)-1$.)

If instead you have

$$\log(y)=\beta_0+\beta_1 \log(x)+u,$$

then $\beta_1$ is an ELASTICITY: a 1 percent increase in $x$ is associated with about a $\beta_1$ percent change in $y$.

Which form are you using in your regression output: $\log(y)$ on $x$, or $\log(y)$ on $\log(x)$ (or both)?

---

## 14. What is the difference between correlation and causation in a regression context?

### combined

*sources: Pre-recorded lecture 05 -2025-1, Pre-recorded lecture 08-1, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, Interactive Lecture 9 slides-1*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021**  ·  _8 chunks, v2_hybrid_rrf, 21155ms + 7817ms_

In regression, “correlation” means $X$ and $Y$ move together in your data, e.g. $\text{Corr}(X_i,Y_i)\neq 0$ or you estimate a relationship in

$$
Y_i=\beta_0+\beta_1 X_i+e_i.
$$

That by itself is NOT “causation”: causation is directional and intervention-based: if you changed $X$ (held other determinants fixed), would $Y$ change? In the regression model, a key condition for reading $\beta_1$ causally is exogeneity:

$$
E[e_i\mid X_i]=0,
$$

which rules out omitted variables $Z$ (endogeneity), reverse causation, and simultaneity as explanations for the correlation.

Pick a concrete example you have in mind: what is one plausible third factor $Z$ that could affect BOTH $X$ and $Y$ (creating correlation without $X\rightarrow Y$)?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 16216ms + 6887ms_

In regression, “correlation” is about ASSOCIATION in the data, while “causation” is about what would happen to $Y$ if you INTERVENED to change $X$.

Correlation (association) in a simple regression:

$$
Y = \beta_0 + \beta_1 X + u
$$

means $\beta_1$ describes how the conditional mean changes with $X$:

$$
\beta_1 = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}
$$

So you can say “a one-unit increase in $X$ is associated with a $\beta_1$ change in $Y$,” but that statement alone does NOT mean $X$ causes $Y$.

Causation means an intervention on $X$ changes the distribution of $Y$ (the “do” idea): if we were to set $X$ to a different value, $Y$ would change as a result. In regression terms, you only get a causal interpretation of $\beta_1$ when $X$ is effectively as-good-as-random after conditioning, i.e., no unblocked “backdoor paths” (no confounding) so that $X$ is uncorrelated with what’s left in the error term.

Key regression warning sign: if $X$ is correlated with $u$ (endogeneity), then OLS is picking up both the effect of $X$ and the influence of omitted/confounding factors, so the regression captures correlation, not causation.

What’s one concrete example in your class notes where $X$ and $Y$ move together, but you would NOT believe changing $X$ would change $Y$?

---

## 15. What are fixed effects and when do I need them?

### combined

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 18667ms + 7365ms_

Fixed effects means you’re controlling for an unobserved, time-invariant factor for each unit $i$ (person, firm, state), usually written as $a_i$ in the panel model

$$
y_{it} = \beta_0 + \beta_1 x_{it1} + \cdots + \beta_k x_{itk} + a_i + u_{it}.
$$

The key point is: FE allows $a_i$ to be correlated with the regressors $x_{itj}$, so you’re not attributing those permanent differences across units to $x$. The standard FE implementation uses the “within” (demeaning) transformation:

$$
y_{it} - \bar y_i = \beta_1 (x_{it1}-\bar x_{i1}) + \cdots + \beta_k (x_{itk}-\bar x_{ik}) + (u_{it}-\bar u_i).
$$

You need fixed effects when you think there are unit-specific, constant traits (ability, baseline productivity, geography, culture) that both affect $y_{it}$ and are correlated with your $x_{it}$.

What is your observational unit $i$ and what’s one plausible time-invariant factor $a_i$ that might be correlated with your key regressor?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 14177ms + 6929ms_

Fixed effects are a way to handle UNOBSERVED, TIME-INVARIANT differences across units in panel data (like ability, baseline risk, firm culture) that would otherwise sit in the error term and bias OLS if they correlate with your regressors.

Start from the unobserved effects model:

$$
y_{it}=\beta_0+\beta_1 x_{it1}+\cdots+\beta_k x_{itk}+a_i+u_{it}
$$

Fixed effects removes $a_i$ by the within (demeaning) transformation:

$$
\ddot y_{it}=y_{it}-\bar y_i=\beta_1(x_{it1}-\bar x_{i1})+\cdots+\beta_k(x_{itk}-\bar x_{ik})+(u_{it}-\bar u_i)
$$

You NEED fixed effects when you think $a_i$ is correlated with one or more $x_{itj}$ (so pooled OLS on $y_{it}$ and $x_{it}$ would confound the effect of $x$ with stable unit traits). You typically DO NOT need FE (and might prefer RE or pooled OLS) if you’re willing to assume $a_i$ is uncorrelated with all regressors for all $t$.

One key diagnostic question: what is a plausible time-invariant factor that differs across your units $i$ and might be correlated with your main regressor $x_{it}$?

---

## 16. How does difference-in-differences identify a causal effect?

### combined

*sources: Pre-recorded lecture 09-1, Interactive Lecture 9 slides-1, URfIE_web, Final 2024 - suggested solutions-2-1*  ·  **book sources: URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 18366ms + 6518ms_

Difference-in-differences (DiD) is trying to isolate a treatment effect by subtracting off (i) permanent level differences between groups and (ii) common time shocks that hit both groups.

The DiD estimand is the “double difference”:

$$
\text{DiD}=
\Big(E[Y \mid Post=1,Tr=1]-E[Y \mid Post=0,Tr=1]\Big)
-
\Big(E[Y \mid Post=1,Tr=0]-E[Y \mid Post=0,Tr=0]\Big)
$$

In the regression version, you estimate:

$$
Y=\beta_0+\beta_1 D_{post}+\beta_2 D_{Tr}+\beta_3(D_{post}\times D_{Tr})+\beta_4 X+u
$$

and $\beta_3$ corresponds to that DiD term.

What assumption has to be true so that the treatment group’s UNOBSERVED counterfactual change over time equals the control group’s observed change over time?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 19966ms + 9121ms_

Difference-in-differences identifies a causal effect by comparing changes over time in the treated group to changes over time in an untreated control group:

$$
\text{DiD}=\Big(E[Y\mid Treat=1,Post=1]-E[Y\mid Treat=1,Post=0]\Big)-\Big(E[Y\mid Treat=0,Post=1]-E[Y\mid Treat=0,Post=0]\Big)
$$

In the regression version,

$$
Y=\beta_0+\beta_1 Treat+\beta_2 Post+\beta_3(Treat\times Post)+u,
$$

the DiD effect is captured by $\beta_3$.

What assumption do you need about the treated group’s “no-treatment” change over time relative to the control group’s change over time, so that subtracting the control change removes the confounding time effects?

---

## 17. What is the difference between a standard error and a standard deviation?

### combined

*sources: Interactive Lecture 5-1, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, Pre-recorded lecture 06-1*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021**  ·  _8 chunks, v2_hybrid_rrf, 19443ms + 6056ms_

Standard deviation (SD) is about DISPERSION IN THE DATA; standard error (SE) is about UNCERTAINTY IN AN ESTIMATE.

1) Standard deviation: spread of a random variable or sample around its mean  
Population definition:

$$\mathrm{Var}(X)=\mathbb{E}\left[(X-\mathbb{E}[X])^2\right], \qquad \mathrm{sd}(X)=\sqrt{\mathrm{Var}(X)}$$

In a sample, you estimate the population variance $\sigma^2$ with:

$$s^2=\frac{1}{N-1}\sum_{i=1}^N (X_i-\bar X)^2, \qquad s=\sqrt{s^2}$$

So SD answers: “How spread out are individual observations $X_i$?”

2) Standard error: spread of an ESTIMATOR across hypothetical repeated samples  
SE is the standard deviation of a sampling distribution (e.g., of $\bar X$). The course slide says:

$$\mathrm{Var}(\bar X)=\frac{\sigma^2}{N}\quad\Rightarrow\quad \mathrm{SD}(\bar X)=\frac{\sigma}{\sqrt{N}}$$

Since $\sigma$ is unknown, we plug in $s$:

$$\mathrm{SE}(\bar X)=\frac{s}{\sqrt{N}}$$

So SE answers: “If I re-sampled and recomputed the statistic (like $\bar X$), how much would that statistic vary?”

One quick check: if $N$ increases, $s$ (SD of the data) doesn’t mechanically shrink, but $\mathrm{SE}(\bar X)=s/\sqrt{N}$ DOES shrink because the mean gets more precisely estimated.

To make it concrete: are you thinking about SE for a sample mean $\bar X$, or SE for a regression slope $\hat\beta_1$?

### books-only

*sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 18124ms + 5848ms_

Standard deviation (SD) describes how spread out the DATA are around their sample mean:

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^n (Y_i-\bar Y)^2}
$$

Standard error (SE) describes how spread out an ESTIMATOR is across repeated samples, i.e., the standard deviation of the sampling distribution of something like $\bar Y$ or $\hat\beta$:

$$
SE(\bar Y)=\frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}}
$$

So SD is variability in outcomes within one sample; SE is uncertainty/variability in your estimate due to sampling (it typically shrinks as $n$ grows, via the $\sqrt{n}$ in the denominator). In regression, the “residual standard error” (RMSE) is an estimate of $\sigma$ (the SD of the error term), and the reported coefficient standard errors are estimates of $SD(\hat\beta)$ across hypothetical repeated samples.

When you see “standard error” in a regression table, which estimator’s sampling variability is it referring to: $\hat\beta_j$ or the residuals?

---

## 18. How do I read a confidence interval on a coefficient?

### combined

*sources: Pre-recorded lecture 06-1, Pre-recorded lecture 07-1, URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  **book sources: 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021, URfIE_web**  ·  _8 chunks, v2_hybrid_rrf, 22374ms + 6221ms_

A confidence interval (CI) for a regression coefficient $\beta_j$ is the range of coefficient values that are CONSISTENT with your data, given sampling uncertainty.

Mechanically, for a two-sided $(100-\alpha)\%$ CI you compute:

$$
\hat\beta_j \pm z_{1-\alpha/2}\cdot SE(\hat\beta_j)
$$

So a 95% CI is:

$$
[\hat\beta_j - 1.96\cdot SE(\hat\beta_j),\ \hat\beta_j + 1.96\cdot SE(\hat\beta_j)]
$$

Interpretation (the correct “frequentist” one): if you repeatedly drew samples the same way and built a 95% CI each time, then about 95% of those intervals would contain the true $\beta_j$. It is NOT “there’s a 95% probability that $\beta_j$ is in this particular interval.”

What CI are you looking at (the lower and upper numbers), and what is the regressor $x_j$?

### books-only

*sources: URfIE_web, 625807794-Nick-Huntington-Klein-The-Effect-Routledge-2021*  ·  _8 chunks, v2_hybrid_rrf, 15709ms + 9754ms_

A confidence interval (CI) for a regression coefficient $b_j$ is the set of coefficient values that are “consistent with the data” at a given confidence level.

For a 95% CI, you usually construct it as

$$
\hat b_j \pm c \cdot se(\hat b_j)
$$

where $\hat b_j$ is your estimated coefficient, $se(\hat b_j)$ is its standard error, and $c$ is the two-sided critical value from a $t$ distribution (typically $c=t_{0.975,df}$).

How to read it: “If we repeatedly drew samples and rebuilt the interval this way, about 95% of those intervals would contain the true $b_j$.” In practice you treat the interval as a plausible range for $b_j$.

One key reading check: if the CI includes $0$, what does that say about rejecting $H_0: b_j=0$ at the 5% (two-sided) level?

---

