# Dataset Card: PAMAP2 Physical Activity Monitoring Benchmark

## 1. Overview & Provenance
- **Dataset Name**: PAMAP2 Physical Activity Monitoring Dataset
- **Repository**: UCI Machine Learning Repository (Dataset ID: 231)
- **Direct UCI URL**: `https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring`
- **Creators**: Attila Reiss and Didier Stricker (DFKI / TU Kaiserslautern)
- **Publication**: Reiss, A., & Stricker, D. (2012). *Introducing a New Benchmarked Dataset for Activity Monitoring*. IEEE ISWC, pp. 108–109.
- **Licence**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).

## 2. Participant Demographics
- **Sample Size**: 9 human subjects (8 male, 1 female).
- **Age**: $27.2 \pm 3.31$ years.
- **Height**: $179.6 \pm 5.10\text{ cm}$.
- **Weight**: $80.9 \pm 8.78\text{ kg}$.
- **Body Mass Index (BMI)**: $25.1 \pm 2.62\text{ kg/m}^2$.
- **Dominant Arm**: 8 right-handed, 1 left-handed.
- **Data Partitions**:
  - `Train`: Subjects 101, 102, 103, 104, 105, 106
  - `Validation`: Subject 107
  - `Test`: Subjects 108, 109

## 3. Sensor Topology & Physical Characteristics
- **Sensory Units**: 3 Colibri wireless Inertial Measurement Units (IMUs).
  - Placement 1: Dominant wrist (Hand IMU)
  - Placement 2: Chest IMU (sternum level)
  - Placement 3: Dominant ankle (lateral above shoe)
  - Heart rate monitor: Chest strap ($\sim 9\text{ Hz}$).
- **IMU Sampling Frequency**: $100\text{ Hz}$ ($10\text{ ms}$ interval).
- **Decimated Frequency**: $25\text{ Hz}$ ($40\text{ ms}$ interval) after 4th-order zero-phase Butterworth low-pass anti-aliasing filtering ($f_c = 10\text{ Hz}$).

## 4. Accelerometer Selection Rationale ($\pm 16g$ vs $\pm 6g$)
Each IMU records from two accelerometers:
- $\pm 6g$ sensor (resolution $\approx 0.001g$, sensitive but saturates at $58.86\text{ m/s}^2$).
- $\pm 16g$ sensor (resolution $\approx 0.004g$, range up to $156.96\text{ m/s}^2$).

During vigorous locomotion (Running), ground-strike vertical accelerations transmit through the skeleton, causing instantaneous acceleration peaks of $8g$–$12g$. The $\pm 6g$ channel saturates, introducing hard flat-top clipping that violates Gaussianity. The **$\pm 16g$ sensor is strictly required** and utilized for all analysis.

## 5. Primary Hidden State Labels
| State Name | PAMAP2 Activity ID | Description | Characteristic Signal Dynamics |
| :--- | :---: | :--- | :--- |
| **Walking** | `4` | Self-paced walking ($4$–$5\text{ km/h}$) | Rhythmic oscillation ($1.5$–$2.0\text{ Hz}$), magnitude $10$–$15\text{ m/s}^2$. |
| **Sitting** | `2` | Static sitting on an office chair | Static $1g$ gravitational alignment, near-zero dynamic variance ($\sigma^2 < 0.2$). |
| **Running** | `5` | Jogging/running ($8$–$9\text{ km/h}$) | High-amplitude cyclic spikes ($2.5$–$3.5\text{ Hz}$), magnitude $18$–$35\text{ m/s}^2$. |
| *Transient* | `0` | Inter-activity pauses / transitions | Excluded from hidden states. |

## 6. Raw File Format (54 Columns)
Raw files (`subject101.dat` to `subject109.dat`) contain 54 numeric columns:
- `Col 0`: Timestamp (s)
- `Col 1`: Activity ID
- `Col 2`: Heart rate (bpm)
- `Cols 3-19`: Hand IMU (temp, $\pm 16g$ acc $x,y,z$, $\pm 6g$ acc $x,y,z$, gyro $x,y,z$, mag $x,y,z$, orientation $q_0..q_3$)
- `Cols 20-36`: Chest IMU (temp, **$\pm 16g$ acc $x,y,z$ [Cols 21, 22, 23]**, $\pm 6g$ acc $x,y,z$, gyro, mag, orientation)
- `Cols 37-53`: Ankle IMU (temp, $\pm 16g$ acc $x,y,z$, $\pm 6g$ acc $x,y,z$, gyro, mag, orientation)

## 7. Known Data Limitations & Preprocessing Protocols
1. **Clock Discrepancy & Heart Rate NaNs**: HR is sampled at $9\text{ Hz}$, generating $\sim 91\%$ missing values in Column 2. HR is omitted from this accelerometer-only study.
2. **Packet Dropouts**: Colibri wireless packets drop occasionally ($\le 0.5\text{ s}$). Addressed via time-indexed linear interpolation strictly within uninterrupted activity segments of individual subjects.
3. **No Leakage Policy**: Normalization scalers are fit strictly on training participants (`101`–`106`). Time sequences are never shuffled.

