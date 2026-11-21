# docker build -f assets/research_lab/application/files/Dockerfile -t opensae-lab:cpu .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENSAE_LAB_DATA=/workspace/runs/research-lab
WORKDIR /workspace
COPY assets/research_lab/application/files/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY base/research_lab ./base/research_lab
COPY assets/research_lab/application/files/data ./assets/research_lab/application/files/data
COPY doc/code/research_lab/source ./doc/code/research_lab/source
COPY assets/history_experiment/benchmark_evidence/files/stage_d_evidence ./assets/history_experiment/benchmark_evidence/files/stage_d_evidence
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin lab && mkdir -p /workspace/runs/research-lab && chown -R lab:lab /workspace/runs
USER lab
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health',timeout=3)" || exit 1
CMD ["uvicorn", "service.service:app", "--app-dir", "/workspace/base/research_lab", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
