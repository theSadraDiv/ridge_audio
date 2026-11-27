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

<img width="540" height="675" alt="رگرشن_01" src="https://github.com/user-attachments/assets/611c613f-81bc-417a-a498-243f99bea931" />
<img width="540" height="675" alt="رگرشن_02" src="https://github.com/user-attachments/assets/7ea36f65-e71f-458d-a9ce-9472ba438908" />
<img width="540" height="675" alt="رگرشن_03" src="https://github.com/user-attachments/assets/484e1ea3-950c-434f-bc13-7d796402d468" />
<img width="540" height="675" alt="رگرشن_04" src="https://github.com/user-attachments/assets/92fc46d8-3889-4150-b35f-986a67f3f73c" />
<img width="540" height="675" alt="رگرشن_05" src="https://github.com/user-attachments/assets/82d607a1-a075-43e8-8d74-f89bb7d62a30" />
<img width="540" height="675" alt="رگرشن_06" src="https://github.com/user-attachments/assets/d1710f4b-8e40-4b45-856c-21c7ce844305" />


















