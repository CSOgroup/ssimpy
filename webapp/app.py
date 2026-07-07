"""ssimpy web service — FastAPI backend.

Start with:
    cd selectsim/
    uvicorn webapp.app:app --host 0.0.0.0 --port 8000

Data retention policy:
    - Uploaded input files are deleted immediately after ssimpy finishes.
    - Result TSV is deleted as soon as the user downloads it.
    - Any job not downloaded within JOB_TTL_SECONDS is purged automatically.
"""
import asyncio
import os
import sys
import time
import uuid
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure the selectsim package directory is on the path
SELECTSIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SELECTSIM_DIR))

from ssimpy import main as ssimpy_main          # noqa: E402
from maf_to_ssimpy import maf_to_ssimpy as convert_maf  # noqa: E402

JOB_TTL_SECONDS = 3600  # jobs not downloaded within 1 hour are purged

# In-memory job store: job_id -> {df, tsv_path, created_at}
_jobs: dict = {}


def _delete_job(job_id: str) -> None:
    """Remove a job's result TSV and evict it from the in-memory store."""
    job = _jobs.pop(job_id, None)
    if job:
        tsv = job.get('tsv_path')
        if tsv and os.path.exists(tsv):
            try:
                os.remove(tsv)
            except OSError:
                pass


async def _cleanup_loop() -> None:
    """Background task: purge jobs older than JOB_TTL_SECONDS every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        cutoff = time.time() - JOB_TTL_SECONDS
        stale = [jid for jid, job in list(_jobs.items()) if job['created_at'] < cutoff]
        for jid in stale:
            _delete_job(jid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title='ssimpy web service', lifespan=lifespan,
              docs_url=None, redoc_url=None)  # disable Swagger UI; we serve /docs ourselves

STATIC_DIR = Path(__file__).parent / 'static'
EXAMPLE_DIR = SELECTSIM_DIR / 'example_data'
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
app.mount('/example_data', StaticFiles(directory=str(EXAMPLE_DIR)), name='example_data')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_upload(f: UploadFile, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    with open(path, 'wb') as out:
        out.write(f._file.read() if hasattr(f, '_file') else b'')
    return path


def _build_summary(results_df):
    n_sig = int(results_df['significant'].sum())
    n_co  = int(((results_df['direction'] == 'co-occurrence') & results_df['significant']).sum())
    n_me  = int(((results_df['direction'] == 'mutual_exclusivity') & results_df['significant']).sum())
    return {'n_significant': n_sig, 'n_co_occurrence': n_co,
            'n_mutual_exclusivity': n_me, 'n_total_pairs': len(results_df)}


def _ssimpy_argv(gam_paths, tmb_paths, output_tsv,
                 n_simulations, fdr, min_mut, tau, lam, filter_pct, seed):
    argv = ['--gam'] + gam_paths
    if tmb_paths:
        argv += ['--tmb'] + tmb_paths
    argv += ['--N', str(n_simulations), '--fdr', str(fdr),
             '--min-mut', str(min_mut), '--tau', str(tau),
             '--lam', str(lam), '--filter-pct', str(filter_pct),
             '--output', output_tsv]
    if seed and str(seed).strip():
        argv += ['--seed', str(seed).strip()]
    return argv


def _register_job(job_id, results_df, output_tsv):
    _jobs[job_id] = {'df': results_df, 'tsv_path': output_tsv, 'created_at': time.time()}
    summary = _build_summary(results_df)
    return {'job_id': job_id, **summary}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
def index():
    return (STATIC_DIR / 'index.html').read_text()


@app.get('/docs', response_class=HTMLResponse)
def docs():
    return (STATIC_DIR / 'docs.html').read_text()


@app.post('/run')
async def run_ssimpy(
    gam_files: list[UploadFile] = File(...),
    tmb_files: list[UploadFile] = File(default=[]),
    n_simulations: int = Form(default=1000),
    fdr: float = Form(default=0.1),
    min_mut: int = Form(default=5),
    tau: float = Form(default=1.0),
    lam: float = Form(default=0.3),
    filter_pct: float = Form(default=0.10),
    seed: Optional[str] = Form(default=None),
):
    if tmb_files and len(tmb_files) != len(gam_files):
        raise HTTPException(status_code=400,
            detail=f'Number of TMB files ({len(tmb_files)}) must match GAM files ({len(gam_files)}).')

    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f'ssimpy_{job_id}_')
    gam_paths, tmb_paths = [], []

    try:
        for i, f in enumerate(gam_files):
            p = os.path.join(tmp_dir, f'gam_{i}_{f.filename}')
            with open(p, 'wb') as out:
                out.write(await f.read())
            gam_paths.append(p)

        for i, f in enumerate(tmb_files):
            p = os.path.join(tmp_dir, f'tmb_{i}_{f.filename}')
            with open(p, 'wb') as out:
                out.write(await f.read())
            tmb_paths.append(p)

        output_tsv = os.path.join(tmp_dir, 'results.tsv')
        argv = _ssimpy_argv(gam_paths, tmb_paths, output_tsv,
                            n_simulations, fdr, min_mut, tau, lam, filter_pct, seed)
        results_df = ssimpy_main(argv)

    except SystemExit as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f'ssimpy error: {e}')
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in gam_paths + tmb_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return _register_job(job_id, results_df, output_tsv)


@app.post('/run-from-maf')
async def run_from_maf(
    maf_file: UploadFile = File(...),
    metadata_file: Optional[UploadFile] = File(default=None),
    gene_list_file: Optional[UploadFile] = File(default=None),
    split_by_type: bool = Form(default=False),
    min_samples: int = Form(default=2),
    min_mutations: int = Form(default=1),
    n_simulations: int = Form(default=1000),
    fdr: float = Form(default=0.1),
    min_mut: int = Form(default=5),
    tau: float = Form(default=1.0),
    lam: float = Form(default=0.3),
    filter_pct: float = Form(default=0.10),
    seed: Optional[str] = Form(default=None),
):
    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f'ssimpy_{job_id}_')
    input_files, gam_paths, tmb_paths = [], [], []

    try:
        # Save MAF
        maf_path = os.path.join(tmp_dir, maf_file.filename or 'input.maf')
        with open(maf_path, 'wb') as out:
            out.write(await maf_file.read())
        input_files.append(maf_path)

        # Save optional metadata
        metadata_path = None
        if metadata_file and metadata_file.filename:
            metadata_path = os.path.join(tmp_dir, metadata_file.filename)
            with open(metadata_path, 'wb') as out:
                out.write(await metadata_file.read())
            input_files.append(metadata_path)

        # Save optional gene list
        gene_list_path = None
        if gene_list_file and gene_list_file.filename:
            gene_list_path = os.path.join(tmp_dir, gene_list_file.filename)
            with open(gene_list_path, 'wb') as out:
                out.write(await gene_list_file.read())
            input_files.append(gene_list_path)

        # Convert MAF → GAM + TMB files
        convert_maf(
            maf_path=maf_path,
            output_dir=tmp_dir,
            prefix='converted',
            split_by_type=split_by_type,
            metadata_path=metadata_path,
            gene_list_path=gene_list_path,
            min_samples=min_samples,
            min_mutations=min_mutations,
        )

        # Build GAM/TMB paths for ssimpy
        if split_by_type:
            gam_paths = [os.path.join(tmp_dir, 'converted_gam_missense.tsv'),
                         os.path.join(tmp_dir, 'converted_gam_truncating.tsv')]
            tmb_paths = [os.path.join(tmp_dir, 'converted_tmb_missense.tsv'),
                         os.path.join(tmp_dir, 'converted_tmb_truncating.tsv')]
        else:
            gam_paths = [os.path.join(tmp_dir, 'converted_gam.tsv')]
            tmb_paths = [os.path.join(tmp_dir, 'converted_tmb.tsv')]

        output_tsv = os.path.join(tmp_dir, 'results.tsv')
        argv = _ssimpy_argv(gam_paths, tmb_paths, output_tsv,
                            n_simulations, fdr, min_mut, tau, lam, filter_pct, seed)
        results_df = ssimpy_main(argv)

    except SystemExit as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f'ssimpy error: {e}')
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Delete all input and intermediate files; keep only result TSV
        for p in input_files + gam_paths + tmb_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return _register_job(job_id, results_df, output_tsv)


@app.get('/results/{job_id}')
def get_results(job_id: str, all: bool = False):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    df = job['df'] if all else job['df'][job['df']['significant']]
    cols = ['gene1', 'gene2', 'n_mut_gene1', 'n_mut_gene2', 'n_comut',
            'freq_gene1', 'freq_gene2', 'nES', 'direction', 'FDR', 'significant']
    return df[cols].to_dict(orient='records')


@app.get('/download/{job_id}')
def download_results(job_id: str, background_tasks: BackgroundTasks):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    background_tasks.add_task(_delete_job, job_id)
    return FileResponse(
        job['tsv_path'],
        media_type='text/tab-separated-values',
        filename='ssimpy_results.tsv',
        background=background_tasks,
    )
