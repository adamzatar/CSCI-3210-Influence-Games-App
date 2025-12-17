# CSCI 3210 Influence Games

Explore linear-threshold influence games, cascades, PSNE, and forcing sets. The core engine lives in `src/` and the Streamlit dashboard is in `web/streamlit_app/app.py`.

## Running locally
- Python 3.11+ is recommended.
- Optional: create a virtual environment.
- `pip install -r requirements.txt`
- `python -m streamlit run web/streamlit_app/app.py`

## How this maps to the paper
- Baseline Kuran mapping: fully connected, weight=1 edges. Thresholds θ are constants and count how many neighbors must be active (because weights are uniform). PSNE listing highlights the lowest vs highest participation equilibria.
- Latent bandwagon: mix of low and higher θ values so a low-participation PSNE coexists with all-ones. The ε slider in the preset subtracts a small constant from every θ_i; once high thresholds drop enough, the low PSNE disappears.
- Extensions: a sparse structure (star) and a weighted hub example. We allow non-uniform weights as an extension beyond Kuran’s uniform baseline; negative weights are future work.
- Forcing sets: our “most influential nodes” are the minimal sets that make the all-ones profile the only PSNE when fixed to 1.
- Irfan’s indicative nodes: defined as the smallest set whose actions uniquely identify a PSNE; not implemented in the app yet.

## Deployment (Streamlit Community Cloud)
1. Push or fork the repo to GitHub (e.g., `adamzatar/CSCI-3210-Influence-Games-App`).
2. Go to https://share.streamlit.io/ (Streamlit Community Cloud) and create a new app.
3. Select the repo and branch.
4. Set the entry point to `web/streamlit_app/app.py`.
5. Confirm Python 3.11+ and point to `requirements.txt` (included in the repo).
6. Click Deploy. The live URL will be shareable once the build finishes.

## Optional: Docker
To run on hosts that prefer containers:
- `docker build -t influence-games .`
- `docker run -p 8501:8501 influence-games`
The container CMD runs `streamlit run web/streamlit_app/app.py` on port 8501.
