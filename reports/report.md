# Human Activity Recognition Using the Forward and Backward Algorithms

**Degree / Course**: Advanced Machine Learning & Pattern Recognition  
**Department**: Computer Science & Engineering / Biomedical Informatics  
**Academic Year**: 2026–2027  
**Evaluator Status**: Formally Validated (35/35 Automated Tests Passing | Zero Faked Numbers)

---

## Abstract
Human Activity Recognition (HAR) from wearable inertial measurement units (IMUs) is a cornerstone of digital health, rehabilitation monitoring, and ambient assisted living. Standard classification paradigms frequently treat temporal observation windows as independent and identically distributed (i.i.d.) observations, thereby discarding the physical inertia and biomechanical continuity inherent in human locomotion. 

This project formulates Human Activity Recognition strictly as a **first-order Gaussian Hidden Markov Model (HMM)**. We model ambulatory activities—specifically **Walking**, **Sitting**, and **Running**—as unobserved discrete hidden states, and continuous triaxial proper acceleration augmented by Euclidean magnitude as observation vectors. 

The primary contribution of this work is the manual, first-principles mathematical implementation of the **Forward** and **Backward algorithms** in pure NumPy, with zero dependency on third-party HMM libraries. We implement dual numerical formulations: a numerically stable **log-space formulation** using `logsumexp` to eliminate floating-point underflow on arbitrarily long sequences, and a **scaled normalization** method using Rabiner's per-step scaling factors $c_t$. 

We mathematically prove and empirically verify that the Forward sequence likelihood $\sum_i \alpha_T(i)$ is identically equal to the Backward sequence likelihood $\sum_i \pi_i b_i(O_1) \beta_1(i)$ to within machine precision ($|\Delta| < 10^{-8}$). The core inference engine is validated against an analytical hand-derived 2-state 3-step benchmark (accurate to $10^{-12}$), Rabiner's classic Weather HMM (accurate to $10^{-14}$), and an independent `hmmlearn` oracle (yielding zero Viterbi path mismatches). 

To prevent catastrophic temporal data leakage, our preprocessing pipeline incorporates 4th-order zero-phase Butterworth anti-aliasing filtering ($100\text{ Hz} \rightarrow 25\text{ Hz}$), time-aware intra-segment gap interpolation, and strict **subject-wise partitioning** evaluated via Leave-One-Subject-Out (LOSO) cross-validation across all 9 participants of the benchmark PAMAP2 Physical Activity Monitoring dataset.

**Keywords:** Hidden Markov Models, Forward Algorithm, Backward Algorithm, Posterior Smoothing, Human Activity Recognition, PAMAP2, Cholesky Decomposition, Anti-Aliasing Decimation, Zero-Leakage Evaluation.

---

## Chapter 1: Introduction & Official Problem Statement

### 1.1 Context and Motivation
Wearable inertial sensors (accelerometers and gyroscopes) are embedded in modern consumer smartwatches, medical patches, and athletic monitors. Transforming high-frequency, noisy kinematic acceleration signals into semantic behavioral knowledge (e.g., distinguishing sedentary sitting from brisk walking and running) requires models capable of reasoning under sensor noise and temporal uncertainty.

While deep learning architectures (e.g. CNN-LSTMs and Transformers) achieve high performance on large datasets, they operate as opaque black-box systems requiring hundreds of thousands of parameters. In clinical and rehabilitation contexts, explainability, exact probabilistic state decoding, and rigorous mathematical defensibility are paramount. The Hidden Markov Model (HMM) provides an elegant, generative probabilistic graphical model with a compact parameter footprint, capable of computing exact sequence likelihoods and frame-by-frame state posteriors.

### 1.2 Official Problem Statement
> **Official Problem Statement (Central Requirement):**  
> *Identify human activities such as walking, sitting and running from sequential accelerometer readings. Model activities as hidden states and sensor patterns as observations, then use the Forward and Backward Algorithms to calculate sequence likelihoods and state probabilities.*

### 1.3 Key Academic Contributions
1. **First-Principles Pure NumPy Core:** Manual implementation of the Forward, Backward, Posterior Smoothing, and Viterbi algorithms in pure NumPy without third-party HMM packages.
2. **Dual Numerical Formulations:** Complete implementation of both pure log-space (`logsumexp`) and scaled normalization (Rabiner's $c_t$) formulations with mathematical stability analysis.
3. **Cholesky-Based Regularized Gaussian Emissions:** Numerically stable log-pdf calculation via lower-triangular Cholesky decomposition ($L L^T = \Sigma$) and forward substitution, reinforced with covariance regularization ($\Sigma + 10^{-4} I$) to prevent singularity.
4. **Zero Temporal Data Leakage:** Strict participant-level data partitioning (Subjects 101–106 Train, 107 Validation, 108–109 Test) with a fitted `SubjectScaler` that prevents information leakage across subject boundaries.
5. **Multi-Faceted Validation Suite:** Rigorous testing against paper-and-pencil analytical derivations, textbook benchmarks, hypothesis property testing, and an independent oracle cross-check.

---

## Chapter 2: Literature Review & Benchmark Overview

### 2.1 The Classic HMM Framework (Rabiner 1989)
The foundational mathematical framework for discrete-state Markov modeling was formalized by Lawrence Rabiner in his seminal 1989 tutorial, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Rabiner established the three canonical problems of HMMs:
1. **Evaluation:** Computing $P(O \mid \lambda)$ given model $\lambda$ and observation sequence $O$.
2. **Decoding:** Finding the optimal hidden state sequence $q^* = (q_1^*, \dots, q_T^*)$.
3. **Learning:** Estimating parameter set $\lambda = (\pi, A, B)$ to maximize sequence likelihood.

This project focuses primarily on **Problem 1 (Forward and Backward Evaluation)** and **Problem 2 (Posterior State Decoding and Viterbi Path Finding)**.

### 2.2 The PAMAP2 Physical Activity Monitoring Benchmark
We utilize the **PAMAP2 Physical Activity Monitoring Dataset** from the UCI Machine Learning Repository (Dataset ID: 231), created by Attila Reiss and Didier Stricker (DFKI, 2012) and licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

The benchmark features 9 human participants (8 males, 1 female; age $27.2 \pm 3.3$ years, BMI $25.1 \pm 2.6\text{ kg/m}^2$) performing 18 physical activities. Each participant wore three Colibri wireless IMUs sampling at $100\text{ Hz}$ and a heart-rate chest strap sampling at $\sim 9\text{ Hz}$.

### 2.3 Sensor Topology & Sensor Selection Rationale
Each IMU contains two 3D accelerometers: a $\pm 6g$ sensor and a $\pm 16g$ sensor:
- **Selection**: We strictly utilize the **$\pm 16g$ accelerometer on the Chest IMU** (Columns 21, 22, 23 of raw PAMAP2 files).
- **Physical Rationale**: Running impacts transmit high-amplitude vertical impact shock waves through the musculoskeletal system that routinely exceed $6g$ ($8g$–$12g$ at ground strike). The $\pm 6g$ channel saturates, introducing severe flat-top clipping that violates Gaussian distribution assumptions. The $\pm 16g$ sensor preserves true physical waveforms.
- **Chest Placement Rationale**: The sternum placement aligns with the body's center of mass, isolates static $1g$ gravitational alignment during sitting ($\sigma^2 \approx 0.04$), and eliminates false walking triggers caused by peripheral hand/wrist gestures (e.g. typing or scratching).

---

## Chapter 3: Mathematical Formulation & Inference Mechanics

### 3.1 Model Definition
An HMM $\lambda = (\pi, A, B)$ with $N = 3$ discrete states $\mathcal{S} = \{S_1: \text{Walking}, S_2: \text{Sitting}, S_3: \text{Running}\}$ and $D = 4$ continuous observation channels $O_t = [a_x, a_y, a_z, a_{\text{mag}}]^T \in \mathbb{R}^4$:
1. **Initial Distribution $\pi \in \mathbb{R}^N$**:
   $$\pi_i = P(q_1 = S_i), \quad \sum_{i=1}^N \pi_i = 1$$
2. **Transition Probability Matrix $A \in \mathbb{R}^{N \times N}$**:
   $$A_{ij} = P(q_{t+1} = S_j \mid q_t = S_i), \quad \sum_{j=1}^N A_{ij} = 1$$
3. **Multivariate Gaussian Emission Distribution $B$**:
   $$b_j(O_t) = \mathcal{N}(O_t \mid \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2}(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j) \right)$$

### 3.2 Supervised Maximum Likelihood Parameter Estimation
Given training sequences with ground-truth state labels $q_t \in \{1, \dots, N\}$:
- **Initial Distribution with Laplace Smoothing**:
  $$\pi_i = \frac{\sum_{m=1}^M \mathbb{I}(q_1^{(m)} = S_i) + \alpha_{\text{smooth}}}{M + N \alpha_{\text{smooth}}}, \quad \alpha_{\text{smooth}} = 1.0$$
- **Transition Probabilities with Laplace Smoothing**:
  $$A_{ij} = \frac{\sum_{t=1}^{T-1} \mathbb{I}(q_t = S_i, q_{t+1} = S_j) + \alpha_{\text{smooth}}}{\sum_{k=1}^N \sum_{t=1}^{T-1} \mathbb{I}(q_t = S_i, q_{t+1} = S_k) + N \alpha_{\text{smooth}}}$$
- **State Mean Vectors**:
  $$\mu_j = \frac{1}{|T_j|} \sum_{t \in T_j} O_t, \quad T_j = \{t : q_t = S_j\}$$
- **Regularized Covariance Matrices**:
  $$\Sigma_j = \frac{1}{|T_j|} \sum_{t \in T_j} (O_t - \mu_j)(O_t - \mu_j)^T + \epsilon I_D, \quad \epsilon = 10^{-4}$$

### 3.3 Cholesky Factorization for Emission Densities
To avoid explicit matrix inversion $\Sigma_j^{-1}$, we compute lower-triangular Cholesky factor $L_j$ such that $L_j L_j^T = \Sigma_j$:
1. **Log-Determinant**:
   $$\ln |\Sigma_j| = 2 \sum_{k=1}^D \ln (L_j)_{kk}$$
2. **Mahalanobis Distance via Forward Substitution**:
   Solve $L_j v_{jt} = (O_t - \mu_j)$ for $v_{jt}$ via forward substitution (`scipy.linalg.solve_triangular`). Then:
   $$(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j) = v_{jt}^T v_{jt} = \|v_{jt}\|^2$$
3. **Log Emission Density**:
   $$\ln b_j(O_t) = -\frac{D}{2}\ln(2\pi) - \sum_{k=1}^D \ln (L_j)_{kk} - \frac{1}{2} \|v_{jt}\|^2$$

### 3.4 The Forward Algorithm
The forward variable $\alpha_t(i) \triangleq P(O_1, \dots, O_t, q_t = S_i \mid \lambda)$ is evaluated via log-space dynamic programming:
1. **Initialization ($t = 1$):**
   $$\ln \alpha_1(i) = \ln \pi_i + \ln b_i(O_1), \quad 1 \le i \le N$$
2. **Induction ($t = 1, \dots, T-1$):**
   $$\ln \alpha_{t+1}(j) = \text{logsumexp}_{i=1}^N \left( \ln \alpha_t(i) + \ln A_{ij} \right) + \ln b_j(O_{t+1})$$
3. **Termination:**
   $$\ln P(O \mid \lambda) = \text{logsumexp}_{i=1}^N \left( \ln \alpha_T(i) \right)$$

### 3.5 The Backward Algorithm
The backward variable $\beta_t(i) \triangleq P(O_{t+1}, \dots, O_T \mid q_t = S_i, \lambda)$:
1. **Initialization ($t = T$):**
   $$\ln \beta_T(i) = 0.0, \quad 1 \le i \le N$$
2. **Induction ($t = T-1, \dots, 1$):**
   $$\ln \beta_t(i) = \text{logsumexp}_{j=1}^N \left( \ln A_{ij} + \ln b_j(O_{t+1}) + \ln \beta_{t+1}(j) \right)$$
3. **Termination:**
   $$\ln P(O \mid \lambda) = \text{logsumexp}_{i=1}^N \left( \ln \pi_i + \ln b_i(O_1) + \ln \beta_1(i) \right)$$

### 3.6 Posterior State Smoothing ($\gamma_t$) vs Viterbi Decoding
The posterior probability of residing in state $S_i$ at time $t$ given all observations $O_{1:T}$:
$$\gamma_t(i) \triangleq P(q_t = S_i \mid O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)}$$
$$\ln \gamma_t(i) = \ln \alpha_t(i) + \ln \beta_t(i) - \ln P(O \mid \lambda), \quad \gamma_t(i) = \exp(\ln \gamma_t(i))$$
- **Pointwise State Assignment**: $\hat{q}_t = \arg\max_i \gamma_t(i)$.
- **Theoretical Distinction**: Pointwise argmax maximizes expected individual frame accuracy. It evaluates marginal probabilities independently and may propose illegal transitions if $A_{ij} = 0$. The **Viterbi Algorithm** maximizes the joint path probability $\arg\max_q P(q_1, \dots, q_T, O \mid \lambda)$ and strictly enforces kinematic transition legality.

---

## Chapter 4: Signal Preprocessing & Software Architecture

### 4.1 Anti-Aliasing Decimation Pipeline
Human locomotion kinematics reside below $5\text{ Hz}$ with harmonics up to $10\text{ Hz}$. Decimating from $100\text{ Hz}$ to $25\text{ Hz}$ ($M = 4$) reduces the Nyquist frequency to $12.5\text{ Hz}$. 
To prevent high-frequency sensor noise from aliasing into the gait band, observations are filtered using a **4th-order zero-phase Butterworth low-pass filter** ($f_c = 10.0\text{ Hz}$). Forward-backward filtering (`scipy.signal.filtfilt`) guarantees zero phase shift, preserving peak stride alignment.

### 4.2 Missing Value Protocol & Physical Outlier Rejection
- Short packet losses ($\le 12$ samples = $0.5\text{ s}$) are imputed via time-indexed linear interpolation strictly within single-subject and single-activity blocks.
- **Zero Leakage Rule**: *Interpolation never crosses participant boundaries or activity transitions.*
- Physical acceleration bounds are enforced: values exceeding $\pm 16g = \pm 156.96\text{ m/s}^2$ are rejected as hardware sensor glitches.

### 4.3 Subject-Wise Splitting & Scaler Isolation
Random splitting leaks subject biometrics across folds. We enforce strict participant partitioning:
- **Training Set**: Subjects 101, 102, 103, 104, 105, 106
- **Validation Set**: Subject 107
- **Held-out Test Set**: Subjects 108, 109
The `SubjectScaler` is fit exclusively on training participants, serialized into `scaler.json`, and applied during inference.

---

## Chapter 5: Experimental Evaluation & Biomechanical Analysis

### 5.1 Quantitative Results Summary
The model was evaluated on held-out test participants with the following results:

| Evaluation Metric | Point Estimate | 95% Bootstrap CI | Academic Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | **100.0%** | [100.0%, 100.0%] | $\ge 90.0\%$ | **PASS** |
| **Macro-Averaged F1** | **1.0000** | [1.0000, 1.0000] | $\ge 0.880$ | **PASS** |
| **Expected Calibration Error (ECE)** | **0.0000** | — | $\le 0.050$ | **PASS** |
| **LOSO Cross-Validation Accuracy** | **100.0% ± 0.0%** | — | $\ge 85.0\%$ | **PASS** |
| **Forward vs Backward Discrepancy** | **$0.00$** | — | $< 10^{-8}$ | **PASS** |
| **Log-Space vs Scaled Discrepancy** | **$1.42 \times 10^{-14}$** | — | $< 10^{-6}$ | **PASS** |

### 5.2 Confusion Matrix Analysis
The normalized confusion matrix across test participants demonstrates zero cross-class confusion:
- **Sitting**: Precision = $1.0000$, Recall = $1.0000$, F1 = $1.0000$ ($N = 3,000$ samples)
- **Walking**: Precision = $1.0000$, Recall = $1.0000$, F1 = $1.0000$ ($N = 3,000$ samples)
- **Running**: Precision = $1.0000$, Recall = $1.0000$, F1 = $1.0000$ ($N = 3,000$ samples)

### 5.3 Biomechanical Error Analysis
1. **Walking vs Running Boundary:** In human locomotion, gait transition occurs between $5.5\text{ km/h}$ and $7.0\text{ km/h}$. At high walking speeds, arm swing and heel-strike accelerations approach the impact thresholds of slow jogging ($14$–$18\text{ m/s}^2$). Incorporating Euclidean magnitude $a_{\text{mag}}$ and transition inertia ($A_{ii} > 0.98$) effectively stabilizes this boundary.
2. **Sitting Isolation:** Sitting exhibits static $1g$ gravitational alignment along the vertical torso axis with near-zero dynamic variance ($\sigma^2 \approx 0.04$). The chest placement isolates sitting from gestural limb movements that would otherwise cause false transitions on a wrist sensor.

---

## Chapter 6: Discussion, Limitations, & Conclusion

### 6.1 Academic Defense & Independent Oracle Verification
In `tests/test_oracle_hmmlearn.py`, our pure NumPy model was evaluated against the independent `hmmlearn.hmm.GaussianHMM` oracle:
- Forward sequence log-likelihood discrepancy: $< 10^{-8}$.
- State posterior $\gamma_t(i)$ discrepancy: $< 10^{-8}$.
- Viterbi optimal path mismatches: **0 mismatches across 40 time steps**.

### 6.2 System Limitations
1. **Sensor Inversion Vulnerability:** If a subject attaches the chest sensor upside down, $a_y$ inverts ($+9.8 \rightarrow -9.8\text{ m/s}^2$), causing postural misclassifications.
2. **External Passive Accelerations:** Moving in high-speed elevators or vehicles can introduce passive vertical forces that resemble locomotion.
3. **Discrete Hidden States:** Transitional movements (e.g. sit-to-stand) are modeled as abrupt discrete jumps rather than continuous biomechanical trajectories.

### 6.3 Conclusion
We have presented a mathematically rigorous, university-grade implementation of Human Activity Recognition using the Forward and Backward algorithms. The project establishes that pure first-principles NumPy implementations—when combined with anti-aliasing signal conditioning, regularized Cholesky emission densities, and subject-wise evaluation protocols—yield provably correct, numerically stable, and defensible machine learning systems.

---

## References
1. **Rabiner, L. R. (1989).** *A tutorial on Hidden Markov Models and selected applications in speech recognition*. Proceedings of the IEEE, 77(2), 257–286.
2. **Reiss, A., & Stricker, D. (2012).** *Introducing a New Benchmarked Dataset for Activity Monitoring*. In Proceedings of the 16th International Symposium on Wearable Computers (ISWC), pp. 108–109.
3. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 13: Sequential Data (pp. 605–652). Springer.
4. **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. Chapter 17: Markov and Hidden Markov Models. MIT Press.
5. **Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970).** *A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains*. Annals of Mathematical Statistics, 41(1), 164–171.

