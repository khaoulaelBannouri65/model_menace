# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio torch-geometric networkx joblib matplotlib flask

# Exposer le port utilisé par Flask
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
