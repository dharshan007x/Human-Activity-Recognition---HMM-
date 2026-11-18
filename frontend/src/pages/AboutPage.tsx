import React, { useState } from 'react';
import { BookOpen, HelpCircle, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Callout } from '../components/common/Callout';
import { MathBlock } from '../components/common/MathBlock';

interface FAQItem {
  question: string;
  answer: React.ReactNode;
  tag: string;
}

export const AboutPage: React.FC = () => {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  const faqs: FAQItem[] = [
    {
      question:
        'Q1: Why is the Forward Algorithm necessary? Why can we not compute sequence likelihood directly by summing all hidden state paths?',
      tag: 'Computational Complexity',
      answer: (
        <div>
          <p>
            For a sequence of length <MathBlock math="T" /> with <MathBlock math="N" /> hidden states,
            there are <MathBlock math="N^T" /> distinct state sequences. Evaluating the joint probability
            for each individual sequence requires <MathBlock math="2T" /> calculations, resulting in a brute-force
            time complexity of:
          </p>
          <MathBlock block math="\mathcal{O}(2T \cdot N^T)" />
          <p>
            For our modest window of <MathBlock math="N=3" /> states and <MathBlock math="T=50" /> time steps,
            <MathBlock math="3^{50} \approx 7.18 \times 10^{23}" /> paths exist! At 1 billion paths evaluated per
            second, direct summation would take over <strong>22 million years</strong>.
          </p>
          <p style={{ marginTop: '8px' }}>
            The <strong>Forward Algorithm</strong> utilizes dynamic programming to store intermediate forward variables
            <MathBlock math="\alpha_t(i)" />, reducing the computational cost to:
          </p>
          <MathBlock block math="\mathcal{O}(N^2 \cdot T) = 3^2 \cdot 50 = 450 \text{ operations}" />
          <p>
            This executes in less than <strong>0.1 milliseconds</strong> on standard hardware—a computational speedup factor
            of over <MathBlock math="10^{21}" />!
          </p>
        </div>
      ),
    },
    {
      question:
        'Q2: How do you prevent numerical underflow when multiplying dozens of small probabilities in the Forward and Backward algorithms?',
      tag: 'Numerical Stability',
      answer: (
        <div>
          <p>
            Because each emission density and transition probability is strictly bounded in magnitude, the raw forward
            variable <MathBlock math="\alpha_t(i)" /> decays exponentially with sequence length <MathBlock math="t" />:
          </p>
          <MathBlock block math="\alpha_t(i) \sim \mathcal{O}(p^t) \quad \text{as } t \to 50, \quad \alpha_{50} \approx 10^{-65}" />
          <p>
            In standard 64-bit IEEE 754 floating-point arithmetic, numbers smaller than <MathBlock math="10^{-308}" />
            underflow to hard zero, destroying all gradient and probability calculations.
          </p>
          <p style={{ marginTop: '8px' }}>
            Our platform deploys two mathematically defensible solutions:
          </p>
          <ol style={{ paddingLeft: '20px', marginTop: '6px', lineHeight: 1.6 }}>
            <li>
              <strong>Rabiner (1989) Scaled Normalization:</strong> At each step <MathBlock math="t" />, compute scaling
              factor <MathBlock math="c_t = \sum_{i=1}^N \alpha_t(i)" /> and normalize <MathBlock math="\hat{\alpha}_t(i) = \alpha_t(i) / c_t" />.
              Then exact log-likelihood is recovered via <MathBlock math="\ln P(O \mid \lambda) = -\sum_{t=1}^T \ln c_t" />.
            </li>
            <li>
              <strong>Pure Log-Space Arithmetic:</strong> Perform all additions using the numerically stable
              <MathBlock math="\text{logsumexp}(x, y) = \max(x, y) + \ln(1 + e^{-|x - y|})" /> routine.
            </li>
          </ol>
        </div>
      ),
    },
    {
      question:
        'Q3: Why was the wrist/hand IMU sensor chosen over the chest or ankle sensors in the PAMAP2 dataset?',
      tag: 'Biomechanics & Sensor Placement',
      answer: (
        <div>
          <p>
            The PAMAP2 dataset recorded simultaneous telemetry from 3 IMU units: hand/wrist, chest, and ankle.
            We selected the <strong>hand/wrist</strong> placement because:
          </p>
          <ul style={{ paddingLeft: '20px', marginTop: '6px', lineHeight: 1.6 }}>
            <li>
              <strong>Arm-Swing Dynamics:</strong> Walking exhibits rhythmic alternating pendulum arm swings at ~1.8 Hz,
              producing distinct sinusoidal waveforms in <MathBlock math="a_x" /> and <MathBlock math="a_z" />.
            </li>
            <li>
              <strong>Vigorous Running Acceleration:</strong> Running amplifies both frequency (~2.8 Hz) and magnitude
              due to pronounced forward kinetic propulsions, exceeding 25–30 m/s².
            </li>
            <li>
              <strong>Resting Gravitational Quiescence:</strong> Sitting keeps the hand largely stationary on the lap or desk,
              registering near-zero dynamic acceleration and an invariant static 1g Earth gravity vector (~9.8 m/s²).
            </li>
            <li>
              <strong>Consumer Wearable Relevance:</strong> Modern commercial fitness trackers (smartwatches, wristbands)
              operate almost exclusively on wrist-mounted IMUs.
            </li>
          </ul>
        </div>
      ),
    },
    {
      question:
        'Q4: Why is random cross-validation forbidden in human activity recognition, and how did you enforce zero data leakage?',
      tag: 'Scientific Integrity & Leakage',
      answer: (
        <div>
          <p>
            Randomly shuffling time-steps or sliding windows across the entire dataset causes catastrophic
            <strong> subject leakage and temporal autocorrelation leakage</strong>. Because human gait signatures
            are highly unique to individual biometric traits (height, weight, stride frequency), a model trained on
            randomly sampled frames of Subject A will simply "memorize" Subject A's identity rather than learning
            generalized activity dynamics.
          </p>
          <p style={{ marginTop: '8px' }}>
            <strong>Our Protocol:</strong>
          </p>
          <ul style={{ paddingLeft: '20px', marginTop: '6px', lineHeight: 1.6 }}>
            <li>
              <strong>Subject-Wise Split:</strong> Training subjects (101–106), validation subject (107), and held-out
              test subjects (108 & 109) are strictly disjoint.
            </li>
            <li>
              <strong>SubjectScaler Pipeline:</strong> Mean and variance statistics were fit strictly on training subjects
              101–106 and applied out-of-sample to validation and test subjects.
            </li>
            <li>
              <strong>Preserved Time Order:</strong> Accelerometer frames within each subject sequence are strictly chronological.
            </li>
          </ul>
        </div>
      ),
    },
    {
      question:
        'Q5: Why is covariance regularisation necessary in Gaussian emissions, and what does Σ_i + 10⁻⁴I guarantee?',
      tag: 'Numerical Linear Algebra',
      answer: (
        <div>
          <p>
            Evaluating continuous multivariate Gaussian log-density requires computing:
          </p>
          <MathBlock block math="\ln b_i(O_t) = -\frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln |\Sigma_i| - \frac{1}{2}(O_t - \mu_i)^T \Sigma_i^{-1} (O_t - \mu_i)" />
          <p>
            If two sensor features are collinear (such as <MathBlock math="a_z" /> during static sitting) or variance
            along an axis is negligible, the sample covariance matrix <MathBlock math="\Sigma_i" /> becomes singular
            (<MathBlock math="|\Sigma_i| \approx 0" />) or ill-conditioned. Inverting <MathBlock math="\Sigma_i" /> directly
            results in catastrophic numerical explosion or division by zero.
          </p>
          <p style={{ marginTop: '8px' }}>
            Adding a Tikhonov regularizer <MathBlock math="\Sigma_i \leftarrow \Sigma_i + 10^{-4} I" /> guarantees that all
            eigenvalues satisfy <MathBlock math="\lambda_k \ge 10^{-4} > 0" />, ensuring <MathBlock math="\Sigma_i" /> is strictly
            positive-definite. This enables stable Cholesky decomposition <MathBlock math="\Sigma_i = L L^T" />, allowing
            quadratic Mahalanobis distance to be solved with forward-substitution <MathBlock math="L y = (O_t - \mu_i)" /> in
            <MathBlock math="\mathcal{O}(D^2)" /> time without computing an explicit matrix inverse.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Mathematical Foundations &amp; Viva Voce Defense Guide
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Defensible mathematical derivations, proofs, algorithmic complexity analyses, and comprehensive viva
          examination questions with model academic answers.
        </p>
      </div>

      {/* Proof of Forward-Backward Identity */}
      <Card
        title="Theorem & Proof: Total Probability Conservation Identity"
        subtitle="Proving that the sum of α_t(i) β_t(i) equals sequence likelihood P(O|λ) for any t"
        badge={<Badge variant="accent">Formal Proof</Badge>}
      >
        <div style={{ fontSize: '0.88rem', lineHeight: 1.65 }}>
          <p>
            <strong>Theorem:</strong> For any valid Hidden Markov Model <MathBlock math="\lambda = (\pi, A, B)" />
            and observation sequence <MathBlock math="O = (O_1, \dots, O_T)" />, the inner product of the forward
            and backward variables is independent of time step <MathBlock math="t" /> and identically equals the
            sequence likelihood:
          </p>
          <MathBlock
            block
            math="\sum_{i=1}^N \alpha_t(i) \beta_t(i) = P(O \mid \lambda) \quad \forall t \in \{1, 2, \dots, T\}"
          />

          <p style={{ marginTop: '10px' }}>
            <strong>Proof:</strong> By the definition of the forward variable <MathBlock math="\alpha_t(i)" /> and
            the backward variable <MathBlock math="\beta_t(i)" />:
          </p>
          <MathBlock
            block
            math="\alpha_t(i) = P(O_1, \dots, O_t, q_t = S_i \mid \lambda), \quad \beta_t(i) = P(O_{t+1}, \dots, O_T \mid q_t = S_i, \lambda)"
          />
          <p>
            By conditional independence of past and future given current state <MathBlock math="q_t" />:
          </p>
          <MathBlock
            block
            math="\alpha_t(i) \beta_t(i) = P(O_1, \dots, O_t, q_t = S_i \mid \lambda) \cdot P(O_{t+1}, \dots, O_T \mid q_t = S_i, \lambda) = P(O, q_t = S_i \mid \lambda)"
          />
          <p>
            Summing across all mutually exclusive and exhaustive hidden states <MathBlock math="S_i \in \{S_1, \dots, S_N\}" />
            by the Law of Total Probability:
          </p>
          <MathBlock
            block
            math="\sum_{i=1}^N \alpha_t(i) \beta_t(i) = \sum_{i=1}^N P(O, q_t = S_i \mid \lambda) = P(O \mid \lambda) \quad \blacksquare"
          />
        </div>
      </Card>

      {/* Interactive Viva Voce FAQ Accordion */}
      <div>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '14px' }}>
          Frequently Asked Viva Voce Questions &amp; Defensible Answers
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {faqs.map((faq, idx) => {
            const isOpen = openFaq === idx;
            return (
              <div
                key={idx}
                style={{
                  backgroundColor: 'var(--surface-card)',
                  border: isOpen ? '1.5px solid var(--accent-terracotta)' : '1px solid var(--border-sand)',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  transition: 'all 0.2s ease',
                }}
              >
                <button
                  onClick={() => setOpenFaq(isOpen ? null : idx)}
                  style={{
                    width: '100%',
                    padding: '16px 20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    textAlign: 'left',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    gap: '12px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <HelpCircle size={18} style={{ color: 'var(--accent-terracotta)', flexShrink: 0 }} />
                    <span style={{ fontWeight: 600, fontSize: '0.96rem', color: 'var(--text-primary)' }}>
                      {faq.question}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Badge variant="neutral" size="sm">
                      {faq.tag}
                    </Badge>
                    {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </div>
                </button>

                {isOpen && (
                  <div
                    style={{
                      padding: '0 20px 20px 48px',
                      fontSize: '0.88rem',
                      color: 'var(--text-primary)',
                      borderTop: '1px solid var(--border-light)',
                      marginTop: '4px',
                      paddingTop: '14px',
                    }}
                  >
                    {faq.answer}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Academic References */}
      <Card title="Primary Academic References" subtitle="Rigorous foundations cited in research defense">
        <ul style={{ paddingLeft: '20px', fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
          <li>
            <strong>Rabiner, L. R. (1989).</strong> "A tutorial on hidden Markov models and selected applications in
            speech recognition." <em>Proceedings of the IEEE</em>, 77(2), 257–286.
          </li>
          <li>
            <strong>Reiss, A., &amp; Stricker, D. (2012).</strong> "Introducing a new benchmarked dataset for activity
            monitoring." <em>16th International Symposium on Wearable Computers (ISWC)</em>, IEEE.
          </li>
          <li>
            <strong>Bishop, C. M. (2006).</strong> <em>Pattern Recognition and Machine Learning</em>. Springer (Chapter 13:
            Sequential Data &amp; Hidden Markov Models).
          </li>
        </ul>
      </Card>
    </div>
  );
};
