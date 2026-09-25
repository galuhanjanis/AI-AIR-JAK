# JAK-AIR AI v0.6 — TKT 6 Ready Prototype

User-friendly demonstrator for Jakarta urban air-quality prediction and decision support.

## Fastest way to run on Windows
1. Extract this ZIP.
2. Install Python 3.10–3.12 if Python is not already installed.
3. Double-click `RUN_JAK_AIR_AI.bat`.
4. The browser will open the JAK-AIR AI dashboard.

## Main modules
- Executive dashboard
- DLH DKI monitoring mode
- Dishub DKI traffic scenario mode
- Six-pollutant Transformer model output
- Five AQMS station map
- Historical monitoring
- SHAP global explainability
- PCMCI lagged conditional relationship panel
- Functional/TKT-6 evidence page

## Model provenance
The package uses the uploaded:
- `best_model_Transformer.weights.h5`
- `Transformer_AQI_Jakarta_2024_L.ipynb`
- Jakarta 2024 AQI + traffic + meteorology + spatial dataset

### Important methodological note
The notebook explicitly reshapes X to `(samples, 1, 22)` but subsequently declares
a Transformer input shape `(24, 22)`. The saved weights themselves are compatible
with arbitrary sequence length because no positional-embedding weights are present.

For scientific defensibility, this prototype reproduces the **one-timestep pathway**
actually prepared by the notebook. It therefore does **not** claim that the uploaded
weights constitute a trained 24-hour sequence Transformer.

## TKT 6
Running software alone is not sufficient for a defensible TKT-6 claim. Use:
- `docs/TKT6_EVIDENCE_CHECKLIST.md`
- `docs/FUNCTIONAL_TEST_REPORT.md`
- `docs/UAT_DLH_DISHUB.md`

Then perform a documented demonstration in a relevant environment and attach the
evidence to the TKT self-assessment.
