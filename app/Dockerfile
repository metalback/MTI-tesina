FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1. Instala utilidades básicas y unixODBC (para compilar pyodbc)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      gnupg \
      apt-transport-https \
      unixodbc \
      unixodbc-dev \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2. Añade la llave y repo de Microsoft para el driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor \
      > /etc/apt/trusted.gpg.d/microsoft.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/microsoft.gpg] \
      https://packages.microsoft.com/debian/11/prod bullseye main" \
      > /etc/apt/sources.list.d/mssql-release.list

# 3. Instala msodbcsql17 forzando la sobreescritura de archivos en conflicto
RUN apt-get update \
 && ACCEPT_EULA=Y apt-get install -y \
      -o Dpkg::Options::="--force-overwrite" \
      msodbcsql17 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Python deps y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
EXPOSE 5000

CMD ["python", "main.py"]
