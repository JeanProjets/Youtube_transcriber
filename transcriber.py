import os
import re
import tempfile
from pathlib import Path
import yt_dlp
import whisper
import torch
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

class YouTubeTranscriber:
    def __init__(self, output_dir='transcriptions', model_size='large-v3', 
                 device='cuda', message_queue=None):
        """
        Initialise le transcripteur YouTube
        
        Args:
            output_dir: Dossier de sortie pour les transcriptions
            model_size: Taille du modèle Whisper (tiny, base, small, medium, large, large-v3)
            device: 'cuda' pour GPU ou 'cpu'
            message_queue: Queue pour envoyer des messages à l'interface
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.message_queue = message_queue
        
        # Vérifier si CUDA est disponible
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            self.send_message('log', 'CUDA non disponible, utilisation du CPU', 'WARNING')
        else:
            self.device = device
            if device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                self.send_message('log', f'GPU détecté: {gpu_name}', 'SUCCESS')
        
        # Charger le modèle Whisper
        self.send_message('log', f'Chargement du modèle Whisper {model_size}...', 'INFO')
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            self.send_message('log', 'Modèle chargé avec succès!', 'SUCCESS')
        except Exception as e:
            self.send_message('log', f'Erreur lors du chargement du modèle: {str(e)}', 'ERROR')
            raise
        
        # Configuration yt-dlp
        # Utiliser les formats audio directs qui fonctionnent avec les vidéos YouTube
        # 251 (webm, 138k) et 250 (webm, 70k) ou 140 (m4a, 129k) et 139 (m4a, 49k)
        self.ydl_opts = {
            'format': '251/250/140/139/bestaudio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
    
    def send_message(self, msg_type, content, extra=''):
        """Envoyer un message à l'interface via la queue"""
        if self.message_queue:
            self.message_queue.put((msg_type, content, extra))
    
    def clean_filename(self, filename):
        """Nettoyer le nom de fichier en remplaçant les caractères spéciaux"""
        # Caractères à remplacer
        invalid_chars = r'[<>:"/\\|?*]'
        cleaned = re.sub(invalid_chars, '_', filename)
        # Limiter la longueur du nom de fichier
        if len(cleaned) > 200:
            cleaned = cleaned[:200]
        return cleaned.strip()

    def clean_repetitions(self, text):
        """
        Nettoyer les répétitions excessives dans le texte (hallucinations Whisper)
        Par exemple: "word word word word..." répété plus de 3 fois
        """
        if not text:
            return text

        words = text.split()
        if len(words) < 4:
            return text

        cleaned_words = []
        repetition_count = 0
        last_word = None

        for word in words:
            # Comparer le mot normalisé (minuscules)
            word_lower = word.lower().strip('.,!?;:')
            last_word_lower = last_word.lower().strip('.,!?;:') if last_word else None

            if word_lower == last_word_lower:
                repetition_count += 1
                # Si plus de 2 répétitions du même mot, sauter
                if repetition_count > 2:
                    continue
            else:
                repetition_count = 0
                last_word = word

            cleaned_words.append(word)

        return ' '.join(cleaned_words)
    
    def download_audio(self, url):
        """
        Télécharger l'audio d'une vidéo YouTube
        
        Returns:
            tuple: (chemin_audio, titre_video) ou (None, None) si échec
        """
        temp_audio_path = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.send_message('detail', 'Récupération des informations de la vidéo...', 10)
                
                # Récupérer les infos de la vidéo
                with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get('title', 'Unknown')
                    duration = info.get('duration', 0)
                    video_id = info.get('id', 'unknown')
                    
                    # Informer de la durée
                    duration_min = duration // 60
                    duration_sec = duration % 60
                    self.send_message('log', f'Titre: {video_title}', 'INFO')
                    self.send_message('log', f'Durée: {duration_min}:{duration_sec:02d}', 'INFO')
                
                self.send_message('detail', 'Téléchargement de l\'audio...', 20)
                
                # Configurer le téléchargement avec un nom de fichier simple
                ydl_opts = self.ydl_opts.copy()
                # Utiliser l'ID de la vidéo pour éviter les problèmes de caractères spéciaux
                ydl_opts['outtmpl'] = os.path.join(temp_dir, f'{video_id}.%(ext)s')
                ydl_opts['keepvideo'] = False  # Ne pas garder la vidéo
                ydl_opts['no_write_info_json'] = True  # Pas de fichier .info.json
                
                # Télécharger l'audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Trouver le fichier audio téléchargé
                audio_files = list(Path(temp_dir).glob(f'{video_id}.*'))
                # Filtrer pour ne garder que les fichiers audio
                audio_extensions = ['.mp3', '.m4a', '.opus', '.webm', '.wav']
                audio_files = [f for f in audio_files if f.suffix.lower() in audio_extensions]
                
                if audio_files:
                    audio_path = audio_files[0]
                    
                    # Créer un fichier temporaire unique avec un nom propre
                    temp_audio_path = Path(tempfile.gettempdir()) / f"whisper_temp_{video_id}.mp3"
                    temp_audio_path.write_bytes(audio_path.read_bytes())
                    
                    self.send_message('detail', 'Audio téléchargé avec succès!', 40)
                    return str(temp_audio_path), video_title
                else:
                    self.send_message('log', 'Aucun fichier audio trouvé après téléchargement', 'ERROR')
                    return None, None
                    
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            if 'Private video' in error_msg:
                self.send_message('log', 'Vidéo privée ou non disponible', 'ERROR')
            elif 'Video unavailable' in error_msg:
                self.send_message('log', 'Vidéo non disponible', 'ERROR')
            else:
                self.send_message('log', f'Erreur de téléchargement: {error_msg[:100]}', 'ERROR')
            return None, None
        except Exception as e:
            self.send_message('log', f'Erreur inattendue lors du téléchargement: {str(e)}', 'ERROR')
            return None, None
        finally:
            # Nettoyer le fichier temporaire si une erreur s'est produite
            if temp_audio_path and Path(temp_audio_path).exists() and not audio_files:
                try:
                    Path(temp_audio_path).unlink()
                except:
                    pass
    
    def transcribe_audio(self, audio_path):
        """
        Transcrire un fichier audio avec Whisper
        
        Returns:
            dict: Résultat de la transcription ou None si échec
        """
        try:
            self.send_message('detail', 'Début de la transcription avec Whisper...', 50)
            self.send_message('log', 'Transcription en cours (cela peut prendre quelques minutes)...', 'INFO')
            
            # Options de transcription
            options = {
                'language': None,  # Détection automatique de la langue
                'task': 'transcribe',  # 'transcribe' ou 'translate' (vers anglais)
                'temperature': 0.1,  # Un peu de variation pour éviter les hallucinations
                'no_speech_threshold': 0.8,  # Plus agressif pour détecter le silence
                'logprob_threshold': -1.0,
                'compression_ratio_threshold': 2.4,
                'condition_on_previous_text': False,  # Désactiver pour éviter les boucles de répétition
                'fp16': self.device == 'cuda',  # FP16 seulement sur GPU
                'verbose': False
            }

            # Transcrire
            result = self.model.transcribe(audio_path, **options)

            # Informer de la langue détectée
            detected_language = result.get('language', 'unknown')
            self.send_message('log', f'Langue détectée: {detected_language}', 'INFO')

            # Post-traitement: nettoyer les répétitions excessives
            result['text'] = self.clean_repetitions(result['text'])

            self.send_message('detail', 'Transcription terminée!', 80)

            return result
            
        except Exception as e:
            self.send_message('log', f'Erreur lors de la transcription: {str(e)}', 'ERROR')
            return None
    
    def save_transcription(self, transcription, video_title):
        """
        Sauvegarder la transcription dans un fichier texte
        
        Returns:
            Path: Chemin du fichier sauvegardé ou None si échec
        """
        try:
            # Nettoyer le titre pour le nom de fichier
            clean_title = self.clean_filename(video_title)
            
            # Créer le nom de fichier avec timestamp pour éviter les doublons
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{clean_title}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            # Écrire la transcription (titre + texte uniquement)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Titre: {video_title}\n")
                f.write("="*80 + "\n\n")
                f.write(transcription['text'].strip())
            
            self.send_message('log', f'Transcription sauvegardée: {filename}', 'SUCCESS')
            return filepath
            
        except Exception as e:
            self.send_message('log', f'Erreur lors de la sauvegarde: {str(e)}', 'ERROR')
            return None
    
    def process_video(self, url):
        """
        Traiter une vidéo complète : télécharger, transcrire, sauvegarder
        
        Returns:
            bool: True si succès, False sinon
        """
        audio_path = None
        try:
            # Nettoyer les anciens fichiers temporaires au début
            self.cleanup_temp_files()
            
            # Télécharger l'audio
            audio_path, video_title = self.download_audio(url)
            if not audio_path:
                return False
            
            # Transcrire
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                return False
            
            # Sauvegarder
            self.send_message('detail', 'Sauvegarde de la transcription...', 90)
            saved_path = self.save_transcription(transcription, video_title)
            
            if saved_path:
                self.send_message('detail', 'Terminé!', 100)
                return True
            else:
                return False
                
        except Exception as e:
            self.send_message('log', f'Erreur inattendue: {str(e)}', 'ERROR')
            return False
            
        finally:
            # Nettoyer le fichier audio temporaire
            if audio_path and Path(audio_path).exists():
                try:
                    Path(audio_path).unlink()
                    self.send_message('log', 'Fichier audio temporaire supprimé', 'INFO')
                except Exception as e:
                    self.send_message('log', f'Impossible de supprimer {Path(audio_path).name}: {e}', 'WARNING')
            
            # Nettoyer tous les fichiers temporaires Whisper
            self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Nettoyer les fichiers temporaires de yt-dlp et Whisper"""
        try:
            temp_dir = Path(tempfile.gettempdir())
            
            # Patterns de fichiers à nettoyer
            patterns = [
                'whisper_temp_*.mp3',  # Nos fichiers temporaires
                '*.part',               # Fichiers partiels yt-dlp
                '*.ytdl',               # Fichiers de verrouillage yt-dlp
                '*.info.json',          # Métadonnées yt-dlp
                'tmp*.mp3',             # Autres fichiers temporaires
                'tmp*.m4a',
                'tmp*.webm'
            ]
            
            cleaned_count = 0
            for pattern in patterns:
                for file in temp_dir.glob(pattern):
                    try:
                        # Ne supprimer que les fichiers de plus de 1 heure
                        if (Path(file).stat().st_mtime < (Path(file).stat().st_mtime - 3600)):
                            file.unlink()
                            cleaned_count += 1
                    except:
                        pass
            
            if cleaned_count > 0:
                self.send_message('log', f'Nettoyage: {cleaned_count} fichiers temporaires supprimés', 'INFO')
                
        except Exception as e:
            # Ne pas faire échouer le processus si le nettoyage échoue
            pass

# Test standalone
if __name__ == "__main__":
    # Pour tester le module directement
    transcriber = YouTubeTranscriber(model_size='base', device='cpu')
    url = input("Entrez une URL YouTube: ")
    success = transcriber.process_video(url)
    print(f"Succès: {success}")