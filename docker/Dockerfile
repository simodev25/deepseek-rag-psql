# Utilisation d'une image Python avec PostgreSQL et les dépendances
FROM python:3.11

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Exécuter l'application (remplace "app.py" par ton script principal)
CMD ["python"]
