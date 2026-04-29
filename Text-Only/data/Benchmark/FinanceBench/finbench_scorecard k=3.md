### FinanceBench Scorecard (k=3 Run)

**Key:**
🟢 **Green:** Matches, encompasses, or explicitly improves upon the Ground Truth.
🟡 **Yellow:** Partial match; correct logic but minor data extraction variance.
🔴 **Red:** Fail / Hallucination / Contradicts reality.

| Company | Question ID | Bot Answer (Key Metric Summary) | Score |
| :--- | :--- | :--- | :---: |
| AMD | dg01 | 1.78x (Hallucinated calculation) | 🔴 |
| AMD | dg07 | CPUs, GPUs, FPGAs, DPUs | 🟢 |
| AMD | dg15 | Data Center, Gaming, Embedded (Xilinx) | 🟢 |
| AMD | dg17 | Amortization from Xilinx acquisition | 🟢 |
| AMD | dg19 | Operating activities (\$3,565M) | 🟢 |
| AMD | novel | Data Center (+63.6%) | 🟢 |
| AMD | novel | Yes, one customer (16% of revenue) | 🟢 |
| PepsiCo | dg08 | US, Canada, LatAm, Europe, AMESA, APAC | 🟢 |
| PepsiCo | dg11 | No material legal battles | 🟢 |
| PepsiCo | dg21 | \$411 million | 🟢 |
| PepsiCo | novel | \$9,068 million | 🟢 |
| PepsiCo | novel | 16.5% | 🟢 |
| AMEX | dg04 | None specified / Not registered | 🟢 |
| AMEX | dg08 | US, EMEA, APAC, LACC | 🟢 |
| AMEX | dg14 | Not measured by operating margin | 🟢 |
| AMEX | dg16 | Not measured by gross margin | 🟢 |
| AMEX | dg23 | Decreased from 24.6% to 21.6% | 🟢 |
| AMEX | novel | Customer deposits (\$110,239M) | 🟢 |
| AMEX | novel | Yes, retained and grew | 🟢 |
| Boeing | dg09 | BCA (38.8%), BDS (34.8%), BGS (26.4%) | 🟢 |
| Boeing | dg11 | Yes (737 MAX, Embraer, Lion Air) | 🟢 |
| Boeing | dg13 | Improved from 4.8% to 5.3% | 🟢 |
| Boeing | dg20 | Commercial airlines & US Government | 🟢 |
| Boeing | novel | Yes, cyclical | 🟢 |
| Boeing | novel | 737/787 increase, 777X paused | 🟢 |
| Boeing | novel | (0.6)% vs 14.8% | 🟢 |

**Summary:** 25 / 26 (96.2% Accuracy). The 3-node configuration matched the high accuracy of the 5-node test, with only one numerical hallucination (AMD Quick Ratio calculation) preventing a perfect score.
