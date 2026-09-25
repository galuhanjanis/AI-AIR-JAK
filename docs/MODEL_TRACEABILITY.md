# Model Integration & Scientific Traceability Note

## Uploaded research model
The research notebook defines:
- targets: PM10, PM2.5, SO2, CO, O3, NO2
- 22 inputs
- lag: 1 hour
- 2-head Transformer block
- feed-forward dimension: 16
- dense head: 16 -> 6 outputs
- checkpoint: best_model_Transformer.weights.h5

Notebook-reported aggregate metrics:
- Training: RMSE 4.8880, MAE 3.6867, R2 0.9617, sMAPE 11.8197%
- Validation: RMSE 3.0050, MAE 1.9885, R2 0.9866, sMAPE 6.6327%
- Test: RMSE 5.9670, MAE 3.2696, R2 0.9486, sMAPE 9.7465%

## Sequence-shape issue
The notebook reshapes model inputs to `(N, 1, 22)`, but then declares
`Input(shape=(24, 22))`. The training outputs show that a model was trained and a
checkpoint saved, but the data-preparation code itself represents a single timestep.

Therefore, for TKT-6 scientific defensibility the application:
1. uses the uploaded H5 weights directly;
2. reproduces the one-timestep inference pathway;
3. does not label the model as a validated 24-hour-sequence Transformer;
4. labels scenario outputs as model sensitivity, not guaranteed causal effects.

## Explainability
SHAP global importances displayed in the app are taken from the research notebook
outputs. They explain model behavior, not direct causality.

## PCMCI
PCMCI results displayed in the app are the aggregated lag-1 conditional relationships
reported by the notebook. The interface uses cautious terminology rather than treating
each link as direct physical causation.
