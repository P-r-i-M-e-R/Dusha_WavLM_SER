# Speech emotion recognition Dusha-WavLM
This repository contains the code for a deep learning project comparing two approaches to Speech Emotion Recognition (SER) on Russian-language audio using the Dusha corpus (SberDevices).
The task is 4-class classification — neutral / sad / angry / positive — evaluated by macro-F1 to account for heavy class imbalance (neutral makes up 66% of the data).
Two models are trained and compared:

Baseline — a lightweight 3-layer 1-D CNN with attention pooling over 40 MFCC coefficients (44k parameters, class-weighted cross-entropy).
Transformer — Microsoft wavlm-base (95M parameters) fine-tuned end-to-end on raw waveforms with mixed precision, cosine LR schedule, gradient accumulation, and early stopping.

Results: WavLM achieves macro-F1 0.656 vs. 0.553 for the CNN baseline — a +18.7% relative gain across all four emotion classes, with the largest improvements on minority classes (angry, positive).
Training was run on Kaggle (2× Tesla T4, 30 GB RAM) on the full dataset of ~197k clips (~270 h) split 70/15/15 stratified.

## Difference in versions:
- v1 consist of first attempt of training models. It has lower f1 metric comparing to others, but it most stable with fully completed pipeline.
- v2 has changed configuration to achieve better metrics. It has interrupted transformer training on second epoch (becase of GPU Kaggle limits), but still shows better results for both models (Also WavLM still better than CNN)
- v3 bonus attempt with trying to achieve better results. It gets better f1 metric than v2 version but interrupted by early stopping during the training (Also the best metric acieved only on 3 epoch out of 8)
