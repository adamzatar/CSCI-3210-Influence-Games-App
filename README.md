# CSCI 3210 Influence Games

Explore linear-threshold influence games, cascades, PSNE, and forcing sets. The core engine lives in `src/` and the Streamlit dashboard is in `web/streamlit_app/app.py`.

## Running locally
- Python 3.11+ is recommended.
- Optional: create a virtual environment.
- `pip install -r requirements.txt`
- `python -m streamlit run web/streamlit_app/app.py`

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
