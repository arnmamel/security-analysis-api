# Utilitzar una sola etapa per a tot el procés de construcció
FROM python:3.10.13

# Establir el directori de treball
WORKDIR /usr/src/sec-ai

# Instal·lar Git
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Copiar tots els fitxers de requisits i instal·lar les dependències
COPY requirements_base.txt requirements_ml.txt requirements_ml2.txt /usr/src/sec-ai/
RUN pip install --no-cache-dir -r requirements_base.txt \
    && pip install --no-cache-dir -r requirements_ml.txt \
    && pip install --no-cache-dir -r requirements_ml2.txt

# Copiar el directori 'dep' i instal·lar les seves dependències
COPY dep /usr/src/sec-ai/dep
RUN pip install /usr/src/sec-ai/dep/nlp-gym

# Configurar PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/sec-ai/dep"

# Copiar l'aplicació i altres arxius necessaris directament del context de construcció
COPY app /usr/src/sec-ai/app
COPY logs /usr/src/sec-ai/logs
COPY datasets /usr/src/sec-ai/datasets
COPY tmp /usr/src/sec-ai/tmp

# Instal·lar Uvicorn
RUN pip install uvicorn

# Exposar el port
EXPOSE 8000

# Establir el comandament per executar l'aplicació
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Afegir les etiquetes de metadades
LABEL author="Arnau Mata Melià" \
      version="1.0" \
      description="Aplicació per interactuar amb un model LLM local (embedit) i un model RL."
