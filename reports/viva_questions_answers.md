# University Viva Voce Defense Guide: 50+ Questions and Comprehensive Answers

**Project Title**: Human Activity Recognition Using the Forward and Backward Algorithms  
**Academic Focus**: Probabilistic Graphical Models, Time-Series Pattern Recognition, Signal Processing  
**Core Benchmark**: PAMAP2 Physical Activity Monitoring Benchmark (UCI ML Repository #231)

---

## Category 1: HMM Theory & Problem Formulation (Q1–Q10)

### Q1: What is a Hidden Markov Model, and what are its two fundamental conditional independence assumptions?
**Answer:** An HMM is a bivariate stochastic process $(\{q_t\}, \{O_t\})$ where $\{q_t\}_{t=1}^T$ is an unobservable (hidden) first-order Markov chain taking discrete values in $\mathcal{S} = \{S_1, \dots, S_N\}$, and $\{O_t\}_{t=1}^T$ is a sequence of continuous or discrete observations generated conditionally by the active state. Its two independence assumptions are:
1. **First-order Markov property on hidden states**: The probability of transition to state $q_{t+1}$ depends strictly on the current state $q_t$, conditionally independent of all preceding history:
   $$P(q_{t+1} \mid q_1, \dots, q_t) = P(q_{t+1} \mid q_t)$$
2. **Conditional independence of emissions**: Given the hidden state $q_t$ at time $t$, the observation $O_t$ is conditionally independent of all other observations and states across the sequence:
   $$P(O_t \mid O_1, \dots, O_T, q_1, \dots, q_T) = P(O_t \mid q_t)$$

### Q2: Why are human activities modeled as hidden states rather than observed variables in this project?
**Answer:** In practical wearable computing, sensors measure physical kinematics (e.g., proper acceleration in $\text{m/s}^2$ and angular rate), not semantic labels. A subject's intention or bodily posture (Walking, Sitting, Running) is an underlying latent phenomenon that manifests indirectly through noisy, periodic, or static dynamical patterns. The hidden state framework provides a principled generative model that explicitly accounts for state persistence (a person walking at step $t$ is overwhelmingly likely to walk at step $t+1$) while handling sensory variance.

### Q3: What is the formal definition of the three parameters $\lambda = (\pi, A, B)$?
**Answer:** 
1. **Initial distribution $\pi \in \mathbb{R}^N$**: $\pi_i = P(q_1 = S_i)$ such that $\sum_{i=1}^N \pi_i = 1$, representing prior belief before observing data.
2. **Transition probability matrix $A \in \mathbb{R}^{N \times N}$**: $A_{ij} = P(q_{t+1} = S_j \mid q_t = S_i)$, satisfying $\sum_{j=1}^N A_{ij} = 1$ (row-stochastic).
3. **Emission probability density $B$**: $b_j(O_t) = P(O_t \mid q_t = S_j)$, here modeled as a continuous multivariate Gaussian distribution $\mathcal{N}(O_t \mid \mu_j, \Sigma_j)$.

### Q4: What are the three canonical problems of HMMs according to Rabiner's classic formulation, and which does this project solve?
**Answer:** 
1. **Problem 1 (Evaluation/Likelihood)**: Given model $\lambda$ and sequence $O$, compute $P(O \mid \lambda)$. Solved by the **Forward** and **Backward** algorithms.
2. **Problem 2 (Decoding)**: Given $\lambda$ and $O$, discover the optimal hidden state sequence. Solved by **Posterior Smoothing ($\gamma_t$)** for pointwise state estimation and the **Viterbi Algorithm** for the globally optimal path.
3. **Problem 3 (Learning/Parameter Estimation)**: Given observations $O$ (and optionally state labels $q$), estimate parameters $\lambda = (\pi, A, B)$ that maximize likelihood. In this project, parameters are estimated via supervised Maximum Likelihood Estimation (MLE) with Laplace smoothing.

### Q5: Do the Forward and Backward algorithms learn model parameters?
**Answer:** **No.** This is a frequent oral defense trap. The Forward and Backward algorithms are purely **inference algorithms**. They assume parameters $\lambda = (\pi, A, B)$ are fixed and already known, and evaluate sequence likelihoods or marginal state posteriors. Parameter learning when labels are hidden requires the **Baum-Welch algorithm** (an Expectation-Maximization procedure) which uses Forward-Backward variables in its E-step to re-estimate parameters in its M-step.

### Q6: Why did you choose supervised MLE parameter estimation instead of unsupervised Baum-Welch for your primary model?
**Answer:** In human activity recognition on benchmark datasets like PAMAP2, we possess calibrated ground-truth annotations from laboratory video recordings. Supervised MLE directly maximizes joint log-likelihood without suffering from local optima or label permutation ambiguity (identifiability problem). Supervised estimation guarantees that state 0 maps strictly to Walking, state 1 to Sitting, and state 2 to Running.

### Q7: Why is Laplace (add-one) smoothing applied to the transition matrix $A$?
**Answer:** If the training data contains zero observed transitions between two states (e.g., direct transition from Sitting to Running), Maximum Likelihood assigns $A_{ij} = 0$. During inference, if a real user suddenly transitions from sitting to running, a zero transition probability causes $A_{ij} = 0 \implies \log A_{ij} = -\infty$, zeroing out the entire sequence probability and causing catastrophic numerical collapse. Laplace smoothing regularizes discrete counts:
$$A_{ij} = \frac{\text{count}(S_i \rightarrow S_j) + \alpha_{\text{smooth}}}{\sum_k \text{count}(S_i \rightarrow S_k) + N \alpha_{\text{smooth}}}$$
ensuring $A_{ij} > 0$ for all state pairs.

### Q8: What does the diagonal dominance of matrix $A$ signify physically in HAR?
**Answer:** The estimated matrix $A$ exhibits high diagonal values ($A_{ii} > 0.90$). Physically, human activities exhibit high temporal persistence (inertia): human gait does not toggle randomly every $40\text{ ms}$. If an individual is walking at time step $t$, the physical probability that they remain walking at $t+1$ ($40\text{ ms}$ later) is over $98\%$. The expected continuous duration in state $S_i$ is given by the geometric distribution mean:
$$\mathbb{E}[d_i] = \frac{1}{1 - A_{ii}} \text{ time steps} = \frac{1}{(1 - A_{ii}) \cdot f_s} \text{ seconds}$$

### Q9: What is the ergodic vs left-to-right (Bakis) topology, and which one does this model use?
**Answer:** A left-to-right model (Bakis) restricts transitions such that $A_{ij} = 0$ for $j < i$ (used in speech recognition or degradation models). Our HAR project uses an **ergodic (fully connected) topology**, where every activity state can transition to any other activity state, reflecting unconstrained natural human movement.

### Q10: How does an HMM differ from a simple Gaussian Mixture Model (GMM) for activity classification?
**Answer:** A GMM treats each sensor reading $O_t$ as an independent and identically distributed (i.i.d.) draw from a mixture of Gaussians, ignoring temporal order. If you randomly shuffle the sequence in time, GMM classification outputs remain identical. An HMM models dynamic temporal dependencies via the Markov transition matrix $A$, exploiting historical context to resolve ambiguous sensor readings that fall within overlapping Gaussian regions.

---

## Category 2: Forward & Backward Algorithms & Numerical Stability (Q11–Q20)

### Q11: State the exact mathematical definition and inductive recursion of the Forward variable $\alpha_t(i)$.
**Answer:** The Forward variable is defined as the joint probability of observing the initial sequence up to step $t$ and being in state $S_i$ at time $t$:
$$\alpha_t(i) \triangleq P(O_1, O_2, \dots, O_t, q_t = S_i \mid \lambda)$$
1. **Initialization ($t = 1$):**
   $$\alpha_1(i) = \pi_i \cdot b_i(O_1), \quad 1 \le i \le N$$
2. **Induction ($t = 1, \dots, T-1$):**
   $$\alpha_{t+1}(j) = \left[ \sum_{i=1}^N \alpha_t(i) A_{ij} \right] b_j(O_{t+1}), \quad 1 \le j \le N$$
3. **Termination:**
   $$P(O \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$$

### Q12: State the exact mathematical definition and recursive induction of the Backward variable $\beta_t(i)$.
**Answer:** The Backward variable is defined as the conditional probability of future observations from step $t+1$ to $T$, given state $S_i$ at time $t$:
$$\beta_t(i) \triangleq P(O_{t+1}, O_{t+2}, \dots, O_T \mid q_t = S_i, \lambda)$$
1. **Initialization ($t = T$):**
   $$\beta_T(i) \triangleq 1.0, \quad 1 \le i \le N$$
2. **Induction ($t = T-1, \dots, 1$):**
   $$\beta_t(i) = \sum_{j=1}^N A_{ij} \cdot b_j(O_{t+1}) \cdot \beta_{t+1}(j), \quad 1 \le i \le N$$
3. **Termination ($t = 1$):**
   $$P(O \mid \lambda) = \sum_{i=1}^N \pi_i \cdot b_i(O_1) \cdot \beta_1(i)$$

### Q13: Prove why the Forward sequence likelihood mathematically equals the Backward sequence likelihood.
**Answer:** 
- At step $T$, the Forward termination sums joint probabilities over all final states:
  $$\sum_{i=1}^N \alpha_T(i) = \sum_{i=1}^N P(O_1, \dots, O_T, q_T = S_i \mid \lambda) = P(O \mid \lambda)$$
- At step $1$, the Backward termination combines initial state probability, first emission, and future conditional likelihood:
  $$\sum_{i=1}^N \pi_i b_i(O_1) \beta_1(i) = \sum_{i=1}^N P(q_1 = S_i) P(O_1 \mid q_1 = S_i) P(O_2, \dots, O_T \mid q_1 = S_i, \lambda)$$
  Applying conditional independence:
  $$= \sum_{i=1}^N P(O_1, \dots, O_T, q_1 = S_i \mid \lambda) = P(O \mid \lambda)$$
Both expressions sum joint probabilities $P(O, q_t = S_i \mid \lambda)$ marginalized at opposite sequence boundaries ($t=T$ vs $t=1$). In our test suite, $| \log P_{\text{fwd}} - \log P_{\text{bwd}} | = 0.00$ ($< 10^{-8}$).

### Q14: Why do linear implementations of the Forward algorithm underflow, and at what sequence length does it happen?
**Answer:** In linear space, $\alpha_t(i) = \sum_j \dots \prod_{\tau=1}^t b(O_\tau)$. Since observation likelihoods are typically smaller than $1$ (e.g. $b(O) \approx 0.1$), the joint probability scales as $\mathcal{O}((0.1)^T)$. For a 64-bit IEEE 754 float, the smallest representable positive subnormal number is $\approx 5 \times 10^{-324}$ (normalized $\approx 2.2 \times 10^{-308}$). In a 25 Hz accelerometer sequence, $T = 100$ samples represents only $4.0$ seconds. Long before $T = 300$, linear probabilities hit machine zero (`0.0`), rendering downstream inference completely inoperable.

### Q15: How does the log-space formulation with `logsumexp` prevent underflow?
**Answer:** In log space, numbers are represented by their natural logarithms $\ell = \ln(p) \in (-\infty, 0]$. Multiplication becomes addition: $\ln(ab) = \ln(a) + \ln(b)$. Addition of probabilities $\ln(\sum_i p_i) = \ln(\sum_i \exp(\ell_i))$ is computed using the **log-sum-exp trick**:
$$\text{logsumexp}(\ell_1, \dots, \ell_N) = m + \ln\left( \sum_{i=1}^N \exp(\ell_i - m) \right), \quad m = \max_{i} \ell_i$$
Subtracting $m$ guarantees that the largest exponent argument is $\exp(0) = 1$, strictly preventing floating-point overflow and underflow.

### Q16: How does Rabiner's scaled Forward algorithm work, and how does it retrieve $\log P(O \mid \lambda)$?
**Answer:** Rabiner's scaled algorithm normalizes the forward variable at every time step $t$ by scaling coefficient $c_t$:
$$c_t = \frac{1}{\sum_{i=1}^N \tilde{\alpha}_t(i)}, \quad \hat{\alpha}_t(i) = c_t \tilde{\alpha}_t(i)$$
Because $\sum_i \hat{\alpha}_t(i) = 1.0$ at every step, intermediate numbers never underflow or exceed $1.0$. The true unscaled $\alpha_t(i)$ relates to $\hat{\alpha}_t(i)$ by $\alpha_t(i) = \frac{\hat{\alpha}_t(i)}{\prod_{\tau=1}^t c_\tau}$. At termination:
$$\sum_{i=1}^N \alpha_T(i) = \frac{\sum_i \hat{\alpha}_T(i)}{\prod_{t=1}^T c_t} = \frac{1}{\prod_{t=1}^T c_t} \implies \log P(O \mid \lambda) = -\sum_{t=1}^T \log c_t$$

### Q17: What is the stability trade-off between the log-space formulation and the scaled formulation?
**Answer:** 
- **Log-space**: Completely immune to underflow; can handle arbitrary sequence lengths ($T > 10^6$). However, calling transcendental functions ($\ln$ and $\exp$) at every step incurs slightly higher computational overhead.
- **Scaled formulation**: Uses fast native multiplications and additions; ideal for DSPs and embedded wearables. However, if an emission density drops to exact zero across all states simultaneously, $\sum \tilde{\alpha}_t = 0 \implies c_t = \infty$, leading to division by zero unless guarded.

### Q18: What is the computational time and space complexity of the Forward and Backward algorithms?
**Answer:** 
- **Time Complexity:** $\mathcal{O}(T \cdot N^2)$, where $T$ is the sequence length and $N$ is the number of hidden states. At each time step $t \in \{1, \dots, T-1\}$, for each of the $N$ destination states, an inner product over $N$ source states is evaluated ($N \times N = N^2$ operations).
- **Space Complexity:** $\mathcal{O}(T \cdot N)$ to store the complete forward and backward trellis matrices ($\alpha \in \mathbb{R}^{T \times N}$, $\beta \in \mathbb{R}^{T \times N}$).

### Q19: Why is the naive brute-force calculation of $P(O \mid \lambda)$ computationally impossible?
**Answer:** The brute-force approach sums joint probabilities over all possible hidden state trajectories $q = (q_1, \dots, q_T)$:
$$P(O \mid \lambda) = \sum_{\text{all } q} P(O, q \mid \lambda) = \sum_{q_1} \dots \sum_{q_T} \pi_{q_1} b_{q_1}(O_1) \prod_{t=2}^T A_{q_{t-1}, q_t} b_{q_t}(O_t)$$
There are $N^T$ possible trajectories, each requiring $2T$ operations, leading to $\mathcal{O}(2T \cdot N^T)$ complexity. For our modest sequence of $T = 50$ steps and $N = 3$ states, $3^{50} \approx 7.18 \times 10^{23}$ operations, requiring thousands of years on a supercomputer. The Forward algorithm solves this in $\approx 50 \times 3^2 = 450$ operations via dynamic programming.

### Q20: Explain the significance of the hand-calculated 2-state 3-step benchmark in your test suite.
**Answer:** To guarantee that the code does not harbor subtle indexing, boundary, or scaling bugs that could match between forward and backward while both being wrong, we constructed an analytical pencil-and-paper benchmark:
$$N = 2, T = 3, \quad \pi = [0.6, 0.4], \quad A = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}, \quad B = \begin{bmatrix} 0.5 & 0.1 \\ 0.4 & 0.7 \\ 0.8 & 0.2 \end{bmatrix}$$
The exact analytical derivation yields $\alpha_3 = [0.07616, 0.01500] \implies P(O \mid \lambda) = 0.09116 \implies \log P = -2.395128795738$. Our Python unit test asserts agreement to $10^{-12}$ precision, verifying mathematical correctness beyond question.

---

## Category 3: Posterior Smoothing vs Viterbi Decoding (Q21–Q30)

### Q21: What is the formal definition of posterior state probability $\gamma_t(i)$, and how is it calculated from $\alpha$ and $\beta$?
**Answer:** $\gamma_t(i)$ is the conditional probability that the system was in state $S_i$ at time $t$, given the entire observation sequence $O$:
$$\gamma_t(i) \triangleq P(q_t = S_i \mid O, \lambda)$$
Using Bayes' rule and the conditional independence structure of HMMs:
$$\gamma_t(i) = \frac{P(q_t = S_i, O \mid \lambda)}{P(O \mid \lambda)} = \frac{P(O_1, \dots, O_t, q_t = S_i \mid \lambda) \cdot P(O_{t+1}, \dots, O_T \mid q_t = S_i, \lambda)}{P(O \mid \lambda)} = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(O \mid \lambda)}$$
In log space:
$$\log \gamma_t(i) = \log \alpha_t(i) + \log \beta_t(i) - \log P(O \mid \lambda)$$

### Q22: Why is the pointwise argmax of $\gamma_t$ NOT guaranteed to produce a valid state sequence?
**Answer:** Individual state decoding sets $\hat{q}_t = \arg\max_i \gamma_t(i)$ independently at each time step. While this maximizes the expected number of correct individual time frames, it evaluates marginal probabilities without enforcing path continuity. If two consecutive argmax states have an impossible transition ($A_{\hat{q}_t, \hat{q}_{t+1}} = 0$), pointwise decoding still outputs that sequence, resulting in a physically illegal path.

### Q23: How does the Viterbi algorithm solve the problem that pointwise posterior decoding cannot?
**Answer:** The Viterbi algorithm finds the single globally optimal trajectory $q^* = (q_1^*, \dots, q_T^*)$ that maximizes the joint probability:
$$q^* = \arg\max_{q_1, \dots, q_T} P(q_1, \dots, q_T, O \mid \lambda)$$
Using the Bellman principle of dynamic programming, it tracks the maximum path probability $V_t(j) = \max_i [V_{t-1}(i) + \log A_{ij}] + \log b_j(O_t)$ and maintains backpointers $B_t(j)$. By maximizing over valid transitions, Viterbi is mathematically guaranteed to output a kinematically permissible sequence ($A_{q_t^*, q_{t+1}^*} > 0$).

### Q24: What is the difference between Filtering and Smoothing in sequential estimation?
**Answer:** 
- **Filtering (Online / Real-Time):** Estimates the current state given only past and present observations up to $t$:
  $$P(q_t = S_i \mid O_{1:t}, \lambda) = \frac{\alpha_t(i)}{\sum_{j=1}^N \alpha_t(j)}$$
  Requires zero lookahead delay, making it suitable for live embedded devices.
- **Smoothing (Offline / Retrospective):** Estimates the state at $t$ using the full sequence $O_{1:T}$ including future context:
  $$P(q_t = S_i \mid O_{1:T}, \lambda) = \gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)}$$
  Future observations provide backward evidence that resolves ambiguous movements and smooths out momentary sensor glitches.

### Q25: Why does $\sum_{i=1}^N \gamma_t(i) = 1.0$ at every single time step?
**Answer:** At any given time step $t$, the system must reside in exactly one of the $N$ mutually exclusive and collectively exhaustive discrete hidden states $\mathcal{S} = \{S_1, \dots, S_N\}$. Therefore:
$$\sum_{i=1}^N \gamma_t(i) = \sum_{i=1}^N P(q_t = S_i \mid O, \lambda) = P\left( \bigcup_{i=1}^N (q_t = S_i) \;\middle|\; O, \lambda \right) = 1.0$$
In our implementation, this invariant is asserted at $10^{-12}$ precision.

### Q26: When would a practitioner prefer pointwise posterior decoding over Viterbi decoding?
**Answer:** When minimizing overall classification error rate on frame-by-frame activities under a 0-1 loss function (where each frame is evaluated independently). Pointwise argmax is also used in the Expectation-Maximization (Baum-Welch) re-estimation equations where fractional state assignments $\gamma_t(i)$ are required rather than hard binary paths.

### Q27: What is the two-slice posterior variable $\xi_t(i, j)$, and how does it relate to $\gamma_t(i)$?
**Answer:** $\xi_t(i, j) \triangleq P(q_t = S_i, q_{t+1} = S_j \mid O, \lambda)$ is the posterior probability of transitioning from state $S_i$ at time $t$ to state $S_j$ at $t+1$:
$$\xi_t(i, j) = \frac{\alpha_t(i) A_{ij} b_j(O_{t+1}) \beta_{t+1}(j)}{P(O \mid \lambda)}$$
Marginalizing over all destination states $j$ yields the single-state posterior:
$$\gamma_t(i) = \sum_{j=1}^N \xi_t(i, j)$$

### Q28: How does an HMM differ from a Kalman Filter?
**Answer:** Both are dynamic Bayesian networks. An HMM has **discrete hidden states** with arbitrary non-linear transitions ($A$) and continuous or discrete emissions. A standard Kalman Filter has **continuous hidden states** governed by linear Gaussian state transitions ($x_{t+1} = F x_t + w$) and linear Gaussian emissions ($y_t = H x_t + v$).

### Q29: What is the difference between an HMM and a Conditional Random Field (CRF)?
**Answer:** An HMM is a **generative model** that specifies the joint distribution $P(O, q \mid \lambda) = P(q) P(O \mid q)$ and requires modeling the observation distribution $P(O \mid q)$. A CRF is a **discriminative model** that models the conditional distribution $P(q \mid O)$ directly, permitting rich overlapping, non-independent feature functions across the observation sequence without requiring generative independence assumptions.

### Q30: What is the Baum-Welch algorithm and how does it relate to Expectation-Maximization?
**Answer:** Baum-Welch is a special case of the EM algorithm for HMM parameter estimation from unlabeled sequences:
- **E-Step:** Computes state posteriors $\gamma_t(i)$ and transition posteriors $\xi_t(i, j)$ using the Forward and Backward variables evaluated on the current parameter estimates $\lambda^{(k)}$.
- **M-Step:** Re-estimates parameters $\pi^{(k+1)}, A^{(k+1)}, \mu^{(k+1)}, \Sigma^{(k+1)}$ as expected counts to maximize the expected complete-data log-likelihood $Q(\lambda, \lambda^{(k)})$.

---

## Category 4: Gaussian Emissions & Cholesky Decomposition (Q31–Q38)

### Q31: Write the multivariate Gaussian probability density function for state $S_j$.
**Answer:** For observation vector $O_t \in \mathbb{R}^D$:
$$b_j(O_t) = \mathcal{N}(O_t \mid \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2}(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j) \right)$$
In logarithmic form:
$$\ln b_j(O_t) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \ln |\Sigma_j| - \frac{1}{2} (O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j)$$

### Q32: Why is the covariance matrix regularized with $\Sigma_j + \epsilon I$, and what happens if $\epsilon = 0$?
**Answer:** During stationary activities like Sitting, physical acceleration along one or more axes is near-constant (near-zero variance). If $\epsilon = 0$, the empirical covariance matrix becomes ill-conditioned or singular ($|\Sigma_j| \approx 0$). Inverting a singular matrix causes numerical overflow ($\Sigma_j^{-1} \rightarrow \infty$) and $\ln |\Sigma_j| \rightarrow -\infty$. Adding a small positive regularization ridge $\epsilon I$ ($\epsilon = 10^{-4}$) guarantees all eigenvalues are strictly positive ($\lambda_k \ge \epsilon > 0$), ensuring positive definiteness and stable inversion.

### Q33: How does Cholesky factorization $L L^T = \Sigma$ simplify log-density computation?
**Answer:** For a symmetric positive-definite covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$, Cholesky yields lower-triangular factor $L$:
1. **Log-Determinant**: Since $L$ is triangular, its determinant is the product of diagonal elements:
   $$|\Sigma| = |L L^T| = |L|^2 \implies \ln |\Sigma| = 2 \ln |L| = 2 \sum_{k=1}^D \ln L_{kk}$$
   This converts an $\mathcal{O}(D^3)$ determinant into an $\mathcal{O}(D)$ sum over diagonal elements.
2. **Mahalanobis Distance**: Solve the linear triangular system $L v = (O_t - \mu_j)$ via forward substitution. Then:
   $$(O_t - \mu_j)^T \Sigma^{-1} (O_t - \mu_j) = (O_t - \mu_j)^T (L L^T)^{-1} (O_t - \mu_j) = v^T v = \|v\|^2$$
   Forward substitution avoids explicit matrix inversion and is numerically stable.

### Q34: What is the Mahalanobis distance, and how does it differ from Euclidean distance?
**Answer:** The Euclidean distance $d_E(x, \mu) = \sqrt{(x - \mu)^T (x - \mu)}$ assumes all axes have equal, uncorrelated spherical variance. The Mahalanobis distance:
$$d_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
measures distance in units of standard deviation along the principal axes of covariance, accounting for axis scale differences and cross-correlations (e.g., correlation between lateral and vertical acceleration during gait).

### Q35: What feature dimensions $D$ are used in this project?
**Answer:** $D = 4$ continuous features per time step:
$$O_t = \begin{bmatrix} a_x(t) \\ a_y(t) \\ a_z(t) \\ a_{\text{mag}}(t) \end{bmatrix}, \quad a_{\text{mag}}(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}$$

### Q36: Why was Euclidean magnitude $a_{\text{mag}}$ included as a feature alongside $a_x, a_y, a_z$?
**Answer:** Triaxial acceleration $(a_x, a_y, a_z)$ depends on sensor orientation relative to gravity. If the chest harness slips slightly or a subject's posture tilts, acceleration shifts between axes (e.g. from $a_y$ into $a_z$). Euclidean magnitude $a_{\text{mag}} = \sqrt{a_x^2 + a_y^2 + a_z^2}$ is **rotationally invariant**—it captures total dynamic kinetic energy regardless of orientation changes.

### Q37: Why not use a discrete HMM with vector quantization (k-means) instead of continuous Gaussian emissions?
**Answer:** Discrete HMMs quantize continuous sensor readings into discrete codebook symbols (e.g. 64 centroids via k-means). Vector quantization introduces irreversible discretization error (quantization noise) and discards continuous kinematic nuances. Continuous multivariate Gaussian HMMs retain the full continuous dynamic range of sensor signals.

### Q38: How do you verify that your custom Cholesky Gaussian emission is correct?
**Answer:** In `tests/test_gaussian_emission.py`, we evaluate our pure NumPy Cholesky log-pdf against `scipy.stats.multivariate_normal.logpdf` across random observation matrices. The maximum observed difference across all test runs is $0.00$ ($< 10^{-10}$), proving exact mathematical fidelity to machine precision.

---

## Category 5: Signal Preprocessing, Sensors, & Anti-Aliasing (Q39–Q45)

### Q39: What is the Nyquist-Shannon sampling theorem, and why does it mandate anti-aliasing filtering before decimation?
**Answer:** The Nyquist theorem states that to reconstruct a continuous signal without distortion, the sampling frequency must exceed twice the highest frequency component: $f_s > 2 f_{\max}$ (Nyquist frequency $f_N = f_s / 2$). When downsampling from $100\text{ Hz}$ to $25\text{ Hz}$, the new Nyquist limit drops to $12.5\text{ Hz}$. Any frequency content above $12.5\text{ Hz}$ in the original signal reflects back across the $12.5\text{ Hz}$ boundary into the lower band as false low-frequency distortion (aliasing). An anti-aliasing low-pass filter (10 Hz cutoff) attenuates high-frequency noise before sub-sampling.

### Q40: Why is a zero-phase filter (`filtfilt`) critical for accelerometer time series?
**Answer:** Standard causal IIR digital filters introduce frequency-dependent phase shift, delaying peaks by several time steps. This delay distorts the alignment between acceleration impact spikes and the true activity state. Zero-phase filtering applies the filter in the forward direction, reverses the filtered sequence, and filters backwards. The two phase shifts cancel exactly:
$$\phi_{\text{total}}(\omega) = \phi(\omega) + (-\phi(\omega)) = 0$$
ensuring that stride peaks align accurately in time.

### Q41: Why did you choose the Chest IMU over the Wrist (Hand) or Ankle IMU?
**Answer:** 
1. **Biomechanical Center of Mass:** The sternum/chest position is closest to the whole-body center of mass, isolating translational locomotion from peripheral limb motion.
2. **Sitting Posture Invariance:** During sitting, the chest experiences steady $1g$ gravitational acceleration with negligible dynamic variance ($\sigma^2 \approx 0.04$).
3. **Freedom from Gestural Noise:** Wrist/hand sensors suffer from non-ambulatory gestures (typing, gesturing, hair adjustments, scratching) that trigger false walking predictions.
4. **Impact Ringing Suppression:** Ankle sensors experience extreme foot-strike shocks ($>16g$) that can ring or saturate the sensor.

### Q42: What is the structure of the raw PAMAP2 dataset files?
**Answer:** 9 files (`subject101.dat` to `subject109.dat`), each containing 54 whitespace-separated numeric columns recorded at $100\text{ Hz}$:
- Col 0: Timestamp (s)
- Col 1: Activity ID (0 = other/transient, 1 = lying, 2 = sitting, 3 = standing, 4 = walking, 5 = running, etc.)
- Col 2: Heart rate (bpm) - contains ~91% NaNs due to ~9 Hz sampling.
- Cols 3–19: Hand IMU (wrist)
- Cols 20–36: Chest IMU (Col 20: temp, **Cols 21–23: $\pm 16g$ accelerometer $a_x, a_y, a_z$**)
- Cols 37–53: Ankle IMU

### Q43: How do you handle missing values caused by Bluetooth packet drops?
**Answer:** Wireless Bluetooth transmission occasionally drops 1–3 consecutive samples. We apply **time-indexed linear interpolation strictly within uninterrupted activity segments for each subject**. Crucially, interpolation is **forbidden across subject boundaries or activity transitions**. Long gaps ($> 0.5\text{ s}$) are not interpolated; instead, the continuous stream is segmented into separate non-overlapping sequences.

### Q44: What physical outlier threshold is enforced and why?
**Answer:** The Colibri accelerometer hardware has an operating physical range of $\pm 16g$, corresponding to $\pm 156.96\text{ m/s}^2$. Any reading exceeding this threshold represents electrical noise or communication corruption and is flagged as invalid.

### Q45: How long are the temporal observation sequences $T$, and how was this duration selected?
**Answer:** Sequences are segmented into windows of $T = 50$ samples at $25\text{ Hz}$, corresponding to $2.0\text{ seconds}$ of physical recording. In human biomechanics, a full gait cycle (stride) takes $\sim 1.0$–$1.2\text{ seconds}$ in walking and $\sim 0.7$–$0.8\text{ seconds}$ in running. A $2.0$-second window captures at least two complete stride cycles, providing sufficient statistical support for emission modeling without introducing non-stationarity.

---

## Category 6: Evaluation, Calibration, Generalization & Viva Trap Questions (Q46–Q55)

### Q46: What is subject-wise data splitting, and why is random train/test splitting catastrophic for HAR?
**Answer:** In random splitting, consecutive time steps $(t, t+1, t+2)$ are shuffled randomly into train and test sets. Because consecutive samples are highly correlated (temporal persistence and identical subject biometrics), the model memorizes individual participant quirks (gait speed, posture angle) rather than generalizable activity patterns. This yields falsely inflated test accuracy ($>99\%$) that fails completely on unseen users. **Subject-wise splitting** trains on one set of participants (e.g. Subjects 101–106) and tests exclusively on withheld participants (Subjects 108–109), evaluating true cross-subject generalization.

### Q47: What is Leave-One-Subject-Out (LOSO) Cross-Validation, and why is it the gold standard in wearable ML?
**Answer:** In LOSO cross-validation with $K$ subjects, the system runs $K$ distinct folds. In fold $k$, subject $k$ is completely held out as the test set, while the model is trained on the remaining $K-1$ subjects. The process repeats $K$ times, and performance is reported as mean $\pm$ standard deviation across all folds. It directly measures how well the system generalizes to a completely new human being.

### Q48: What is Expected Calibration Error (ECE) and why does posterior confidence matter?
**Answer:** Accuracy measures if the top prediction is correct, but not whether the probability score reflects real risk. A model that predicts 99% confidence on an uncertain boundary is overconfident. ECE bins predictions by confidence $B_m$ and computes the weighted average gap between mean confidence and empirical accuracy:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
In medical or athletic monitoring, calibrated probabilities ensure downstream decision systems know when the model is uncertain.

### Q49: What is non-parametric bootstrap resampling for confidence intervals?
**Answer:** Rather than assuming asymptotic normality for accuracy or F1-scores, bootstrap resampling draws $B = 1,000$ samples with replacement from the test set, computing accuracy on each resample. The $95\%$ empirical confidence interval is determined by the $2.5\text{th}$ and $97.5\text{th}$ percentiles of the bootstrap distribution, providing rigorous non-parametric error bounds.

### Q50: Where do Walking and Running confuse each other, and what is the physical biomechanical cause?
**Answer:** Confusions occur at the transition boundary between **brisk fast walking** ($\sim 5.5\text{ km/h}$) and **slow fatigue jogging** ($\sim 7\text{ km/h}$). At high walking speeds, arm swing and heel-strike accelerations increase in amplitude, approaching the lower-end ground reaction force spikes of slow jogging ($14$–$18\text{ m/s}^2$). Furthermore, cadence harmonics overlap near $2.2$–$2.5\text{ Hz}$.

### Q51: Oral Trap: "Why did you use hmmlearn in your project if you claim pure NumPy?"
**Answer:** `hmmlearn` is **strictly prohibited in the core inference code** and is used **exclusively inside `tests/test_oracle_hmmlearn.py` as an external oracle benchmark**. Our core Forward, Backward, Posterior, Gaussian emission, and Viterbi modules are written from scratch in pure NumPy. The test file asserts that our custom implementation matches the independent oracle to within $10^{-8}$, proving that our manual implementation is both independent and correct.

### Q52: Oral Trap: "Is Forward-Backward an online or offline algorithm?"
**Answer:** The Forward pass alone can operate in an **online (streaming)** manner up to the current time step $t$. However, the full **Forward-Backward algorithm** is fundamentally an **offline (batch)** algorithm because the Backward recursion begins at the final time step $T$ ($\beta_T = 1$) and recurses backwards to $t=1$. Posterior smoothing $\gamma_t(i)$ requires the complete sequence $O_{1:T}$.

### Q53: What are the primary failure modes of this HMM HAR system?
**Answer:** 
1. **Sensor Displacement / Inversion:** If the user mounts the chest sensor upside down, $a_y$ inverts sign ($+9.8 \rightarrow -9.8\text{ m/s}^2$), causing Sitting to be misclassified.
2. **Elevator / Vehicle Motion:** Passive vertical acceleration in an elevator can produce acceleration patterns resembling locomotion while the subject is standing or sitting.
3. **Complex Mixed Activities:** Multi-tasking (e.g. standing while gesticulating violently) can cause transient confusion between stationary and ambulatory states.

### Q54: How could this HMM project be extended for future academic work?
**Answer:** 
1. **Autoregressive HMMs (AR-HMM):** Modeling temporal correlations within observation dimensions across consecutive steps.
2. **Hierarchical HMMs (HHMM):** Modeling macro-activities (e.g., commute, workout) at an upper layer and micro-movements (walk, run, sit) at a sub-layer.
3. **Multi-Modal Fusion:** Combining chest accelerometer data with heart-rate physiological response and gyroscope angular velocity.

### Q55: Summarize the complete mathematical workflow from raw sensor voltage to activity prediction.
**Answer:** 
1. Colibri IMU measures proper acceleration at $100\text{ Hz}$ on the chest ($\pm 16g$ sensor).
2. 4th-order zero-phase Butterworth filter ($10\text{ Hz}$ cutoff) eliminates high-frequency noise.
3. Signal is decimated by $M=4$ to $25\text{ Hz}$ ($\Delta t = 0.04\text{ s}$).
4. Observation vector $O_t = [a_x, a_y, a_z, a_{\text{mag}}]^T$ is standardized via training-fitted `SubjectScaler`.
5. Cholesky factorization evaluates log-Gaussian emissions $\log b_j(O_t)$.
6. Log-Forward pass computes $\log \alpha_t(i)$ and total sequence log-likelihood $\log P(O \mid \lambda)$.
7. Log-Backward pass computes $\log \beta_t(i)$.
8. Posterior smoothing computes $\gamma_t(i) = \exp(\log \alpha_t(i) + \log \beta_t(i) - \log P(O \mid \lambda))$.
9. Activity prediction is generated via pointwise posterior argmax $\hat{q}_t = \arg\max_i \gamma_t(i)$ or globally optimal Viterbi path.

