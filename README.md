# Ridge Audio
Estimation and Analysis of Audio Spectra Using Ridge Regression

## How it works
1. Generate 300 synthetic half-second clips with varying frequencies gain and noise.
2. Estimate each clip's power spectrum with Welch's method and extract 24 overlapping frequency-band features.
3. Train an ordinary linear regression baseline and a Ridge model on 60 clips. Choose Ridge's regularization strength with five-fold cross-validation.
4. Evaluate both models on 240 held-out clips.

## Run locally
Use Python 3.12.

run:
```bash
python -m pip install -r requirements.txt
python ridge_audio.py
```

The script generates the data and prints a JSON report with both models' errors and an actual-versus-estimated power share for one test clip.

```bash
python ridge_audio.py --sample 12 --output results/report.json
```
