
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	HOST=0.0.0.0 \
	PORT=8003 \
	WEB_CONCURRENCY=1 \
	PRELOAD_MODEL_ON_STARTUP=false

WORKDIR /app

# Pull in latest security fixes available for the base image packages.
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install --no-install-recommends -y ca-certificates tini \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

RUN useradd --create-home --uid 10001 appuser \
	&& chown -R appuser:appuser /app

USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
	CMD python -c "import os,sys,urllib.request; url=f'http://127.0.0.1:{os.getenv(\"PORT\",\"8003\")}/health'; urllib.request.urlopen(url, timeout=3); sys.exit(0)" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["hallucination-lens-api"]
