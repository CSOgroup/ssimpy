# ssimpy web service

A browser-based interface for running ssimpy analyses by uploading input files.

## Installation

```bash
pip install -r webapp/requirements.txt
```

## Start the server

From the `selectsim/` directory:

```bash
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

To make it accessible on a lab server, replace `localhost` with the server's IP address or hostname. Use `--host 0.0.0.0` to accept connections from any machine on the network.

## Usage

1. Upload one or more **GAM files** (binary gene × sample TSV matrices)
2. Optionally upload matching **TMB files** (sample/tmb[/class] TSV) — file order must match GAM files
3. Adjust parameters if needed (number of simulations, FDR threshold, etc.)
4. Click **Run ssimpy**
5. Browse the results table and download the full TSV

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/run` | POST | Upload files + params, run ssimpy, return job ID and summary |
| `/results/{job_id}` | GET | Results as JSON (`?all=true` for all pairs, default = significant only) |
| `/download/{job_id}` | GET | Download full results TSV |
