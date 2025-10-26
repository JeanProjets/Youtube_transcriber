# YouTube Transcriber avec Whisper

Un programme Python avec interface graphique pour transcrire automatiquement des vidéos YouTube en utilisant OpenAI Whisper. Optimisé pour GPU NVIDIA (RTX 3090).

## 🚀 Fonctionnalités

- **Interface graphique intuitive** avec Tkinter
- **Transcription haute qualité** avec Whisper large-v3
- **Support GPU CUDA** pour une transcription rapide
- **Traitement en batch** de plusieurs vidéos
- **Détection automatique de la langue** (français/anglais)
- **Logs détaillés** avec barre de progression
- **Nettoyage automatique** des fichiers temporaires
- **Timestamps inclus** dans les transcriptions

## 📋 Prérequis

- Python 3.8+
- FFmpeg installé sur le système
- GPU NVIDIA avec CUDA (recommandé) ou CPU
- 8GB+ de RAM (16GB+ recommandé pour le modèle large)
- ~10GB d'espace disque pour le modèle Whisper large-v3

## 🔧 Installation

### 1. Cloner ou télécharger les fichiers

Créez un nouveau dossier et placez-y les fichiers :
- `main.py`
- `transcriber.py`
- `requirements.txt`

### 2. Installer FFmpeg

**Windows:**
1. Télécharger depuis [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extraire et ajouter le dossier `bin` au PATH système

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Installer PyTorch avec CUDA (pour RTX 3090)

```bash
# Pour CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1 (plus récent)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Installer les autres dépendances

```bash
pip install openai-whisper yt-dlp ffmpeg-python
```

### 6. Premier lancement (téléchargement du modèle)

Le premier lancement téléchargera automatiquement le modèle Whisper large-v3 (~3GB). Cela ne se fera qu'une seule fois.

## 💻 Utilisation

### Lancer le programme

```bash
python main.py
```

### Interface graphique

1. **Coller les URLs** : Entrez une ou plusieurs URLs YouTube (une par ligne)
2. **Lancer la transcription** : Cliquez sur "🎬 Lancer la transcription"
3. **Suivre la progression** : 
   - Barre globale : nombre de vidéos traitées
   - Barre détaillée : étape actuelle (téléchargement/transcription)
   - Logs : informations détaillées en temps réel
4. **Résultats** : Les transcriptions sont sauvegardées dans le dossier `transcriptions/`

### Format de sortie

Les fichiers de transcription incluent :
- Titre de la vidéo
- Date et heure de transcription  
- Langue détectée
- Texte complet
- Version avec timestamps (minutes:secondes)

## ⚡ Optimisation des performances

### Avec RTX 3090 (configuration actuelle)

- **Modèle large-v3** : ~2-3x plus rapide que le temps réel
- Une vidéo de 10 minutes est transcrite en 3-4 minutes
- Utilisation de FP16 pour accélérer les calculs

### Choix du modèle selon votre matériel

| Modèle | VRAM nécessaire | Qualité | Vitesse (RTX 3090) |
|--------|----------------|---------|---------------------|
| tiny | ~1 GB | ⭐⭐ | ~10x temps réel |
| base | ~1 GB | ⭐⭐⭐ | ~7x temps réel |
| small | ~2 GB | ⭐⭐⭐⭐ | ~5x temps réel |
| medium | ~5 GB | ⭐⭐⭐⭐ | ~3x temps réel |
| large-v3 | ~10 GB | ⭐⭐⭐⭐⭐ | ~2-3x temps réel |

Pour changer le modèle, modifiez la ligne dans `main.py` :
```python
self.transcriber = YouTubeTranscriber(
    output_dir=self.output_dir,
    model_size='large-v3',  # Changer ici
    device='cuda',
    message_queue=self.message_queue
)
```

## 🐛 Résolution des problèmes

### "CUDA non disponible"

Vérifiez votre installation CUDA :
```python
python -c "import torch; print(torch.cuda.is_available())"
```

Si False, réinstallez PyTorch avec la bonne version CUDA.

### "ffmpeg not found"

Assurez-vous que FFmpeg est dans le PATH système :
```bash
ffmpeg -version
```

### Erreur de mémoire GPU

Si vous manquez de VRAM, utilisez un modèle plus petit ou passez sur CPU :
```python
device='cpu'  # Dans main.py
```

### Vidéo privée ou géo-bloquée

Le programme continuera avec les autres vidéos et indiquera l'erreur dans les logs.

## 📁 Structure des fichiers

```
youtube-transcriber/
├── main.py              # Interface graphique
├── transcriber.py       # Logique de transcription
├── requirements.txt     # Dépendances
├── README.md           # Documentation
└── transcriptions/     # Dossier de sortie (créé automatiquement)
    ├── video1_20240115_143022.txt
    ├── video2_20240115_143512.txt
    └── ...
```

## 🔒 Considérations légales

- Respectez les droits d'auteur des vidéos
- Utilisez uniquement pour un usage personnel ou avec permission
- Vérifiez les conditions d'utilisation de YouTube

## 📊 Exemples de cas d'usage

- Créer des notes de cours à partir de vidéos éducatives
- Archiver des podcasts ou interviews
- Générer des sous-titres pour vos propres vidéos
- Analyser le contenu de vidéos pour la recherche
- Créer des résumés de conférences

## 🚀 Améliorations futures possibles

- [ ] Support de plus de langues
- [ ] Export en formats SRT/VTT pour sous-titres
- [ ] Résumé automatique avec LLM
- [ ] Traduction automatique
- [ ] Support des playlists YouTube
- [ ] Mode CLI pour automatisation
- [ ] API REST pour intégration

## 📝 Notes

- La première transcription sera plus lente car le modèle doit être chargé en mémoire
- Les vidéos très longues (>1h) peuvent prendre du temps
- La qualité de transcription dépend de la qualité audio de la vidéo source

## 🤝 Support

Pour toute question ou problème, vérifiez d'abord que :
1. FFmpeg est correctement installé
2. Vous avez assez d'espace disque
3. Votre GPU est reconnu (si utilisation CUDA)
4. Les URLs YouTube sont valides et publiques

Bon transcribing! 🎉