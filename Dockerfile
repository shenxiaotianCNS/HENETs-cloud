FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
WORKDIR /opt/henets
ARG WITH_ANALYSIS=0
COPY requirements-inference.txt requirements-analysis.txt ./
RUN python -m pip install --no-cache-dir -r requirements-inference.txt && \
    if [ "$WITH_ANALYSIS" = "1" ]; then python -m pip install --no-cache-dir -r requirements-analysis.txt; fi && \
    python -m pip check
COPY henets/ henets/
COPY examples/ examples/
COPY tools/licenses/ tools/licenses/
RUN python -c "import cv2, openslide, torch, torchvision; from henets import _engine, infer"
ENTRYPOINT ["python", "-m", "henets.cli"]
CMD ["--help"]
