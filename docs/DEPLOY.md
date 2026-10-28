# Deploy Streamlit and Vercel

Two public demos for the Stage 1 finalist
`HammadHARahim/urdu-headline-stage1-cosine`.

| Target | Runtime | Model loading |
| --- | --- | --- |
| Streamlit Community Cloud | Python + ONNX Runtime | Downloads quantized ONNX from the Hub |
| Vercel | Static browser app | Transformers.js + chunked Hub downloads |

## Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Create app from `HammadHARahim/Infinitic-AI-Urdu-News-Headline-Classifier`.
3. Set:
   - Branch: `main`
   - Main file path: `apps/streamlit/app.py`
   - Python version: 3.12
4. Deploy. First boot downloads the ONNX file (~279 MB) into the app cache.

Local check from the repo root:

```bash
pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

## Vercel

The app lives in `apps/vercel/` and serves `public/index.html`.
Model weights stay on Hugging Face; Vercel only hosts the UI + `manifest.json`.

### CLI deploy

```bash
cd apps/vercel
npx vercel login
npx vercel --prod
```

When prompted, set the project root to `apps/vercel` (or run the commands
from that folder).

### GitHub integration

1. Import the GitHub repo in the Vercel dashboard.
2. Set **Root Directory** to `apps/vercel`.
3. Framework Preset: Other.
4. Deploy. No build command is required for this static app.

## Smoke checks after deploy

1. Paste `پاکستان کرکٹ ٹیم نے میچ جیت لیا` → expect **sports**.
2. Paste `کراچی سٹاک ایکسچینج میں اضافہ` → expect **business**.
3. Confirm the page links back to GitHub and the Hub model card.
