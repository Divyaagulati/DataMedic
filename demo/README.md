# Presentation demo provisioning

Reproducibly prepares the DataMedic presentation account and every demo artifact (dataset,
diagnostics, biomedical summary, recommendations, Experiment Lab, committed version, unapproved
live-demo pipeline, version comparison, training runs and audits, insights, dashboard, export)
through the **public API only**: no database access, no authentication bypass, no copied data.

Data: UCI *Diabetes 130-US Hospitals for Years 1999–2008* (<https://doi.org/10.24432/C5230J>,
CC BY 4.0). The original archive and the generated subset stay **outside** the repository.
See [docs/PRESENTATION_GUIDE.md](../docs/PRESENTATION_GUIDE.md) for the dataset card and the talk.

## 1. Build the dataset (once, deterministic)

```bash
# download the original archive (3.2 MB) into a folder outside the repository
curl -fsSL -o diabetes_130_us_hospitals.zip \
  "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
# archive sha256 at download time (2026-11-27): f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a
python -I demo/build_presentation_dataset.py diabetes_130_us_hospitals.zip <out_dir>
# -> <out_dir>/diabetes_readmission_presentation.csv (4,000 rows, sha256 4c082a1c...) + provenance.json
```

Run it with any Python that has pandas and NumPy (for example `backend/.venv`).

## 2. Credentials file (never commit it)

```
DEMO_EMAIL=datamedic.demo@example.com      # local: Mailpit catches the verification email
DEMO_DISPLAY_NAME=DataMedic Presentation
DEMO_PASSWORD=<long random password>
```

On a public server use an address you can receive mail at: `example.com` cannot receive email.

## 3. Provision

Local stack (`docker compose up -d`):

```bash
backend/.venv/Scripts/python -I demo/provision_presentation.py \
  --web http://localhost:3000 --api http://localhost:8000 --mailpit http://localhost:8025 \
  --credentials <credentials.env> --dataset <out_dir>/diabetes_readmission_presentation.csv \
  --state <out_dir>/demo_state.json \
  --stages upload,roles_and_diagnostics,biomedical,recommendations,experiment,commit_pipeline,live_demo_draft,version_comparison,training,training_class_weight,insights,dashboard,exports
```

Cloud (`https://<domain>`, real SMTP): same command with `--web https://<domain>` and no
`--api`/`--mailpit`. After sign-up the script waits; click the link in the verification email.

The state file records completed stages, so a re-run skips them; the dataset, committed pipeline
and dashboard are also looked up by name, so nothing is duplicated. Total time on the local stack:
about 15–25 minutes, almost all of it model training.
