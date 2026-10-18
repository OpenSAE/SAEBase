# CPU-only deployment checklist

The OpenSAE Research Lab can run as a read-only CPU service. This is a
deployment recipe, not a claim that the existing Hugging Face Space changed.

## Local isolated build

Run from the SAEBase repository root:

    docker build -f apps/research-lab/Dockerfile -t opensae-lab:cpu .
    docker run --rm -p 7860:7860 opensae-lab:cpu

Open http://127.0.0.1:7860. The API summary should show the original 48
historical sites, zero new curated prompt releases, and GPU online as false.
No model weights download.

## Hugging Face Docker Space

Before replacing an existing OpenSAE Space, back up its exact previous
revision and test rollback. An experimental Docker Space can use the metadata
in apps/research-lab/hf/README.md.

The Dockerfile uses SAEBase repository root as build context and copies
the application plus conf/coverage/stage_d_evidence. Copying only the
application into a different Space will lack the 48-site evidence and
must fail closed rather than create pretend results.

The Space is read-only: POST /api/jobs returns HTTP 503, because neither
an authenticated job queue nor a GPU worker is configured.

Do not store experiment artifacts in the ephemeral Space filesystem.
Attach durable object storage later when results can be published. Keep
any user inputs private by default and outside normal request logging.

## CPU release acceptance

1. GET /api/health returns online.
2. GET /api/summary reports 48 historical evidence sites and truthful scope.
3. GET /api/results?metric=core.l0 returns 48 historical rows linking GitHub.
4. GET /api/datasets lists ten candidate sources and actual imported counts.
5. GET /api/features shows no false cards before ingestion.
6. POST /api/jobs returns 503; preflight does not run inference.
7. Neither tokens nor weight binaries are included in the container.
8. Restart preserves browsing, since evidence is copied and read-only.
9. Existing HF Space functionality remains available until replacement has
   been independently tested.

The later GPU worker is a separate, explicitly authenticated service.
