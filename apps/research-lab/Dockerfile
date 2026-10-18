# docker build -f apps/research-lab/Dockerfile -t opensae-lab:cpu .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENSAE_LAB_DATA=/workspace/runs/research-lab
WORKDIR /workspace
COPY apps/research-lab/requirements.txt /workspace/apps/research-lab/requirements.txt
RUN python -m pip install --no-cache-dir -r apps/research-lab/requirements.txt
COPY apps/research-lab /workspace/apps/research-lab
COPY conf/coverage/stage_d_evidence /workspace/conf/coverage/stage_d_evidence
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin lab && mkdir -p /workspace/runs/research-lab && chown -R lab:lab /workspace/runs
USER lab
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health',timeout=3)" || exit 1
CMD ["uvicorn", "backend.api:app", "--app-dir", "/workspace/apps/research-lab", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
