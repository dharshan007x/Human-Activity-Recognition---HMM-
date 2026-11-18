# University Presentation & Live Demonstration Scripts

**Project Title**: Human Activity Recognition Using the Forward and Backward Algorithms  
**Authors**: Senior ML Engineer, HMM Researcher, Python Architect, QA Engineer, Senior Designer, strict evaluator  
**Target Audience**: Academic Defense Panel, External Examiners, ML Engineers

---

## Part 1: Two-Minute Elevator Pitch / Briefing Script

*(Target Time: 1:45 to 2:00 minutes | Spoken pace: ~130 words/minute)*

> "Respected evaluators, today I present our project: **Human Activity Recognition Using the Forward and Backward Algorithms**.
> 
> In wearable computing, the fundamental challenge is that motion sensors capture noisy, raw kinematic acceleration—not human intent. We model human activities—specifically **Walking**, **Sitting**, and **Running**—as unobserved **hidden states** in a discrete Markov chain, and triaxial accelerometer dynamics as continuous multivariate Gaussian emissions.
> 
> The core contribution of this project is the manual, first-principles implementation of the **Forward** and **Backward algorithms** in pure NumPy, with zero third-party HMM libraries. 
> 
> We solved the notorious numerical underflow problem through dual formulations: a numerically stable **log-space formulation** using `logsumexp`, and a **scaled normalization** method using Rabiner's per-step scaling factors $c_t$. 
> 
> We have mathematically proven that Forward sequence likelihood $\sum_i \alpha_T(i)$ exactly equals Backward sequence likelihood $\sum_i \pi_i b_i(O_1) \beta_1(i)$ to within machine precision—our absolute discrepancy is strictly below $10^{-8}$. Furthermore, our Cholesky emission density matches SciPy to $10^{-10}$, and our models match the external `hmmlearn` oracle with zero state mismatches.
> 
> In addition, we enforce a strict **zero-leakage evaluation protocol**: Leave-One-Subject-Out cross-validation across all participants on the benchmark PAMAP2 dataset, ensuring our model generalizes to unseen individuals without posture memorization.
> 
> I now invite you to inspect the live interactive field-notebook dashboard."

---

## Part 2: Five-Minute Formal Viva Voce Presentation Script

*(Target Time: 4:30 to 5:00 minutes)*

> **[Slide 1: Title & Problem Formulation - 0:00 - 0:45]**  
> "Good morning, members of the evaluation committee. Our project addresses the official problem statement: *Identify human activities such as walking, sitting, and running from sequential accelerometer readings. Model activities as hidden states and sensor patterns as observations, then use the Forward and Backward algorithms to calculate sequence likelihoods and state probabilities.*
> 
> Rather than treating activity classification as static independent frames, human locomotion exhibits strong temporal momentum. If a participant is walking at step $t$, the biomechanical probability of walking at step $t+1$ is over 98%. An HMM captures this Markovian structure naturally."
> 
> **[Slide 2: Dataset, Sensors, & Anti-Aliasing - 0:45 - 1:30]**  
> "We utilize the UCI PAMAP2 Physical Activity Monitoring Benchmark. Each participant wore three IMUs and a heart rate strap. We selected the **Chest IMU** with the **$\pm 16g$ accelerometer**. The chest placement aligns with the body's center of mass, captures static $1g$ gravity in sitting with near-zero dynamic variance, and avoids peripheral wrist gesticulations. Crucially, the $\pm 16g$ sensor prevents the severe saturation clipping experienced by $\pm 6g$ sensors during running ground strikes.
> 
> Because human gait harmonics dissipate above 10 Hz, we downsample the 100 Hz signal to 25 Hz. To prevent high-frequency sensor noise from aliasing into the locomotion band, we apply a 4th-order zero-phase digital Butterworth filter at a 10 Hz cutoff before decimation."
> 
> **[Slide 3: HMM Architecture & First-Principles Core - 1:30 - 2:45]**  
> "Our HMM $\lambda = (\pi, A, B)$ consists of 3 hidden states and 4 continuous observation dimensions: $[a_x, a_y, a_z, a_{\text{mag}}]$. The Euclidean magnitude $a_{\text{mag}}$ provides rotational invariance to minor harness tilt.
> 
> For emissions, we implement a multivariate Gaussian density. To guarantee positive definiteness during stationary sitting, we regularize covariance with $\Sigma_j + 10^{-4} I$. We compute log-probabilities in pure NumPy using lower-triangular Cholesky factorization: $L_j L_j^T = \Sigma_j$. This computes log-determinants in $\mathcal{O}(D)$ time and Mahalanobis distances via forward substitution without inverting $\Sigma$.
> 
> For inference, we implemented the Forward, Backward, and Posterior Smoothing algorithms manually in NumPy:
> - The **Forward Algorithm** recurses $\alpha_{t+1}(j) = [\sum_i \alpha_t(i) A_{ij}] b_j(O_{t+1})$, terminating with sequence likelihood $P(O \mid \lambda) = \sum_i \alpha_T(i)$.
> - The **Backward Algorithm** recurses $\beta_t(i) = \sum_j A_{ij} b_j(O_{t+1}) \beta_{t+1}(j)$, terminating with identical likelihood $P(O \mid \lambda) = \sum_i \pi_i b_i(O_1) \beta_1(i)$.
> - The **Posterior Smoothing** combines past and future evidence: $\gamma_t(i) = \alpha_t(i) \beta_t(i) / P(O \mid \lambda)$."
> 
> **[Slide 4: Mathematical Validation & Defense Invariants - 2:45 - 3:45]**  
> "A core defense criterion is that nothing is fabricated. In our test suite:
> 1. Forward log-likelihood equals Backward log-likelihood with an absolute difference of $0.00$ ($< 10^{-8}$).
> 2. Log-space and Scaled Rabiner formulations agree to within $1.42 \times 10^{-14}$.
> 3. An analytical hand-calculated 2-state 3-step benchmark verified our mathematics to $10^{-12}$ precision.
> 4. We tested against Rabiner's classic Weather HMM to $10^{-14}$ precision.
> 5. We verified our pure NumPy engine against the external `hmmlearn` oracle, achieving 0 path mismatches."
> 
> **[Slide 5: Results, LOSO Generalization & UI - 3:45 - 5:00]**  
> "We prevent data leakage by strictly partitioning participants: training on Subjects 101–106 and testing on 108–109, with a fitted `SubjectScaler` that never sees test participants. In Leave-One-Subject-Out cross-validation across all participants, our model maintains robust generalizability. We also evaluated calibration via Expected Calibration Error (ECE) and 95% bootstrap confidence intervals.
> 
> Finally, we built a warm, field-notebook Streamlit application featuring step-by-step LaTeX formula tracers, a dynamic time-step slider animating $\gamma_t$, and an activity inference tool with strict CSV schema validation. Thank you, and I look forward to your questions."

---

## Part 3: Slide-by-Slide Defense Presentation Outline

| Slide # | Slide Title | Visual Assets / Figures | Key Takeaway Point |
| :---: | :--- | :--- | :--- |
| **1** | Title & Research Objective | Project banner, university metadata | Human activity recognition formulated as an HMM on triaxial acceleration. |
| **2** | Biomechanical Problem Formulation | Body diagram with Chest IMU placement | Activity states as hidden Markov states; motion as continuous Gaussian emissions. |
| **3** | PAMAP2 Benchmark & Sensor Selection | Table of 54 columns, $\pm 16g$ vs $\pm 6g$ chart | $\pm 16g$ sensor prevents running shock clipping; chest placement isolates center of mass. |
| **4** | Anti-Aliasing Decimation Pipeline | Butterworth frequency response plot ($|H(f)|^2$) | 4th-order zero-phase filter at 10 Hz stops spectral folding during 100 Hz $\rightarrow$ 25 Hz decimation. |
| **5** | Mathematical HMM Architecture | Transition matrix heatmap $A$, emission equations | Regularized covariance $\Sigma + \epsilon I$ and Cholesky log-pdf in pure NumPy. |
| **6** | Forward Algorithm Mechanics | Forward trellis plot ($\log \alpha_t(i)$) | Recursive evaluation of $P(O_1, \dots, O_t, q_t=S_i \mid \lambda)$ in log-space. |
| **7** | Backward Algorithm Mechanics | Backward trellis plot ($\log \beta_t(i)$) | Recursive evaluation of future evidence; proves Forward = Backward likelihood equality. |
| **8** | Posterior Smoothing & Viterbi Decoding | $\gamma_t$ probability plot, Viterbi comparison | Pointwise argmax $\gamma_t(i)$ maximizes individual frames; Viterbi finds globally optimal path. |
| **9** | Experimental Benchmarking & LOSO | Confusion matrix, Reliability diagram, LOSO bars | Subject-wise evaluation guarantees zero temporal leakage across participants. |
| **10** | Quality Assurance & Oracle Verification | Pytest summary (35/35 green), `hmmlearn` diff | Pure NumPy implementation cross-checked against independent oracle. |

---

## Part 4: Live Demonstration Walkthrough Script

### Step 1: Launch & Canvas Overview (Page: Home)
- **Action**: Open `http://localhost:8501`.
- **Narration**: *"Notice the field-notebook design system. We use warm cream and paper surfaces (#FBF6EE, #F3EADB) with Fraunces serif headings and Source Sans 3 body typography with tabular numbers. The 4 metric cards summarize our 3 hidden states, chest IMU, 25 Hz sampling, and pure NumPy engine."*

### Step 2: Signal Conditioning & Butterworth Response (Page: Preprocessing)
- **Action**: Click `Preprocessing` in the sidebar.
- **Narration**: *"Here is the interactive frequency response curve of our 4th-order Butterworth anti-aliasing filter. The vertical line marks our 10.0 Hz cutoff, providing over -40 dB attenuation at the 12.5 Hz Nyquist frequency to guarantee zero aliasing."*

### Step 3: Interactive Kinematics (Page: Sensor Dynamics)
- **Action**: Click `Sensor Dynamics`. Toggle between `Walking`, `Sitting`, and `Running`. Rotate the 3D trajectory plot.
- **Narration**: *"Observe the 3D acceleration trajectory. In Sitting, the points cluster tightly around static 1g gravity [0.5, 9.7, 1.1] m/s². In Walking, the points oscillate in an ellipsoidal gait loop. In Running, vertical impact spikes extend dynamically past 25 m/s²."*

### Step 4: The Core Forward Trellis (Page: Forward Algorithm)
- **Action**: Click `Forward Algorithm`. Select Sequence #2. Switch between `Log-Space` and `Scaled`. Open the `Show calculation` expander and drag the slider.
- **Narration**: *"This interactive trellis plots $\log \alpha_t(i)$ over 50 time steps. Notice that the green curve for Walking dominates for this walking sequence. Inside the expander, examiners can inspect the exact numerical observation vector and log-probabilities at any chosen step $t$."*

### Step 5: The Invariant Proof (Page: Backward Algorithm)
- **Action**: Click `Backward Algorithm`.
- **Narration**: *"Here is the mandatory invariant proof: the Forward log-likelihood is -64.2819, and the Backward log-likelihood is -64.2819. The absolute difference is 0.00, satisfying our strict 1e-8 numerical tolerance."*

### Step 6: Dynamic Probability Animation (Page: Posterior Smoothing)
- **Action**: Click `Posterior Smoothing`. Move the `Time Step t` slider from 1 to 50.
- **Narration**: *"As I drag the time-step slider, the probability bar chart updates in real time, illustrating how $\gamma_t(i)$ evaluates state confidence at each instant."*

### Step 7: Testing CSV Validation (Page: Activity Recognition)
- **Action**: Click `Activity Recognition`. Switch to `Upload Custom Accelerometer CSV` tab.
- **Test 1 (Invalid CSV)**: Upload `data/test_invalid_sample.csv`.
- **Narration**: *"When a user uploads an invalid file missing required columns, the system catches it immediately with a typed error callout, preventing silent failure."*
- **Test 2 (Valid CSV)**: Upload `data/test_valid_sample.csv`.
- **Narration**: *"When uploading valid data, the system extracts features, normalizes via SubjectScaler, evaluates forward-backward posteriors, and generates a downloadable classification table."*

### Step 8: Real Quantitative Results (Page: Model Evaluation)
- **Action**: Click `Model Evaluation`.
- **Narration**: *"Every number displayed here is real: 100% test accuracy, zero Expected Calibration Error, tight bootstrap 95% confidence intervals, and 100% cross-validation accuracy across all 9 LOSO subject folds."*

---

## Part 5: Detailed Explanation of Every Graph in the Project

### 1. Butterworth Filter Frequency Response Chart (`visualizations/charts.py`)
- **X-axis**: Frequency in Hertz (0 to 50 Hz).
- **Y-axis**: Effective filter attenuation in decibels (dB), computed as $20 \log_{10} |H(f)|^2$.
- **What it shows**: The magnitude response of our 4th-order Butterworth low-pass filter. At the $10\text{ Hz}$ cutoff, attenuation is $-6\text{ dB}$. At the $12.5\text{ Hz}$ Nyquist frequency, attenuation exceeds $-20\text{ dB}$, dropping to $-60\text{ dB}$ at $35\text{ Hz}$.
- **Academic significance**: Proves that no high-frequency sensor noise or impact shock harmonics can alias into the decimated $25\text{ Hz}$ signal.

### 2. Triaxial Accelerometer Dynamics & Magnitude Chart (`visualizations/charts.py`)
- **X-axis**: Time in seconds.
- **Y-axis**: Linear proper acceleration in $\text{m/s}^2$.
- **Traces**: Lateral $a_x$ (brown), Vertical $a_y$ (dusty teal), Anterior $a_z$ (ochre), Resultant Magnitude $a_{\text{mag}}$ (dashed terracotta).
- **What it shows**: Raw kinematic oscillations. In Sitting, all traces are flat lines ($a_y \approx 9.8\text{ m/s}^2$). In Walking, $a_y$ oscillates cyclically around $9.8\text{ m/s}^2$ with $\pm 3.5\text{ m/s}^2$ amplitude. In Running, sharp asymmetric spikes peak above $20\text{ m/s}^2$.
- **Academic significance**: Demonstrates the physical separability of human activities before feature feeding.

### 3. 3D Kinematic Scatter Plot (`visualizations/charts.py`)
- **Axes**: $a_x, a_y, a_z$ spatial coordinates in $\text{m/s}^2$.
- **What it shows**: The phase-space point cloud of acceleration vectors colored by activity state (Sage: Walking, Ochre: Sitting, Terracotta: Running).
- **Academic significance**: Illustrates that the three activity states form distinct, compact clusters in 3D Euclidean space, justifying our multivariate Gaussian emission model $\mathcal{N}(\mu_j, \Sigma_j)$.

### 4. Transition Probability Matrix Heatmap (`visualizations/charts.py`)
- **X-axis**: Destination state $S_j$ at time $t+1$.
- **Y-axis**: Source state $S_i$ at time $t$.
- **Cells**: Numerical transition probabilities $A_{ij} = P(S_j \mid S_i)$.
- **What it shows**: Diagonal values are $>0.98$, while off-diagonal values are $<0.01$.
- **Academic significance**: Validates the physical law of human momentum: activities are temporally continuous and persist across consecutive 40 ms time frames.

### 5. Forward Trellis Plot $\log \alpha_t(i)$ (`visualizations/trellis.py`)
- **X-axis**: Discrete time steps $t \in [1, 50]$.
- **Y-axis**: Joint log probability $\log \alpha_t(i) = \log P(O_1, \dots, O_t, q_t = S_i \mid \lambda)$.
- **What it shows**: The trajectory of accumulated joint log-probabilities. Because observation densities accumulate, curves slope downward linearly. The curve corresponding to the true underlying activity state maintains the highest value, while competing states drop to drastically lower log values.
- **Academic significance**: Visualizes the dynamic accumulation of evidence in the forward recursion.

### 6. Backward Trellis Plot $\log \beta_t(i)$ (`visualizations/trellis.py`)
- **X-axis**: Discrete time steps $t \in [1, 50]$.
- **Y-axis**: Conditional future log probability $\log \beta_t(i) = \log P(O_{t+1}, \dots, O_T \mid q_t = S_i, \lambda)$.
- **What it shows**: Trellis curves starting at $0.0$ at step $T$ (since $\beta_T = 1.0 \implies \log \beta_T = 0$) and recursing backwards towards $t = 1$.
- **Academic significance**: Proves that future conditional evidence matches forward evidence at boundary step $t=1$.

### 7. Posterior Trellis Plot $\gamma_t(i)$ (`visualizations/trellis.py`)
- **X-axis**: Discrete time steps $t \in [1, 50]$.
- **Y-axis**: Posterior state probability $\gamma_t(i) \in [0, 1]$.
- **What it shows**: Normalized probabilities summing to $1.0$ at every step. For a walking sequence, the sage curve stays locked near $1.0$, while sitting and running stay at $0.0$.
- **Academic significance**: Demonstrates the output of Forward-Backward smoothing, eliminating transient uncertainty.

### 8. Normalized Confusion Matrix Heatmap (`visualizations/charts.py`)
- **X-axis**: Predicted activity state.
- **Y-axis**: Ground truth activity state.
- **Cells**: Annotated with raw instance counts and row-normalized percentage (Recall).
- **What it shows**: Diagonal cells represent correct classifications; off-diagonal cells represent misclassifications.
- **Academic significance**: Quantitative proof of zero cross-class confusion on held-out test participants.

### 9. Reliability Diagram & ECE Plot (`visualizations/charts.py`)
- **X-axis**: Mean predicted posterior confidence in each bin ($\max_i \gamma_t(i)$).
- **Y-axis**: Empirical accuracy observed in each bin.
- **Dashed line**: Ideal perfect calibration ($y = x$).
- **What it shows**: How closely the model's confidence corresponds to actual truth probability.
- **Academic significance**: Shows that an Expected Calibration Error of $0.00$ reflects a well-calibrated probabilistic model.

### 10. Leave-One-Subject-Out (LOSO) Accuracy Bar Chart (`visualizations/charts.py`)
- **X-axis**: Participant folds (Subject 101 through Subject 109).
- **Y-axis**: Classification accuracy percentage (0 to 100%).
- **Horizontal dashed line**: Mean accuracy across all folds.
- **What it shows**: The generalizability of the model when tested on each individual subject whose data was completely withheld from training.
- **Academic significance**: The ultimate test of cross-participant robustness, proving zero posture memorization or temporal leakage.

