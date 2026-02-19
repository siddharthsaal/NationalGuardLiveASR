## Performance Metrics

| Metric | Average Value |
| :--- | :--- |
| **Average Latency** | 7.95 seconds |
| **Average Processing Duration** | 6.09 seconds |
| **Total Samples Analyzed** | 9 |

## Detailed Breakdown

> [!NOTE]
> Latency is measured from the start of the audio segment processing to the completion of the transcription.

| Sample ID | Duration (s) | Latency (s) | Status |
| :--- | :--- | :--- | :--- |
| 0 | 4.32 | 5.78 | Partial |
| 1 (p1) | 4.00 | 5.47 | Partial |
| 1 (p2) | 7.62 | 7.68 | Partial |
| 1 (p3) | 5.00 | 11.91 | Completed |
| 1 (p4) | 2.78 | 10.69 | Partial |
| 3 | 7.87 | 7.44 | Partial |
| 4 (p1) | 4.00 | 5.26 | Partial |
| 4 (p2) | 7.62 | 7.43 | Partial |
| 4 (p3) | 11.58 | 9.89 | Partial |

## Observations
- The average latency is slightly higher than the segment duration, indicating some overhead in processing or network communication.
- Longer segments generally result in higher latency, as expected for sequential 
