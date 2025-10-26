import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from pathlib import Path
import queue
import sys
from datetime import datetime
import torch
from transcriber import YouTubeTranscriber

def get_best_device():
    """Détecte automatiquement le meilleur device disponible"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Transcriber - Whisper")
        self.root.geometry("800x600")
        
        # Queue pour la communication entre threads
        self.message_queue = queue.Queue()
        
        # Créer le dossier de sortie
        self.output_dir = Path("transcriptions")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        self.transcriber = None
        self.is_processing = False
        
        # Démarrer la vérification de la queue
        self.check_queue()
    
    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Section URLs
        url_frame = ttk.Frame(main_frame)
        url_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        url_frame.columnconfigure(0, weight=1)

        url_label = ttk.Label(url_frame, text="URLs YouTube (une par ligne):",
                             font=('Arial', 10, 'bold'))
        url_label.grid(row=0, column=0, sticky=tk.W)

        clear_btn = ttk.Button(url_frame, text="Effacer", command=self.clear_urls)
        clear_btn.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))

        # Zone de texte pour les URLs
        self.url_text = scrolledtext.ScrolledText(main_frame, height=8, width=70)
        self.url_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Frame pour les boutons et barres de progression
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(3, weight=1)
        
        # Bouton de transcription
        self.transcribe_btn = ttk.Button(control_frame, text="🎬 Lancer la transcription",
                                        command=self.start_transcription)
        self.transcribe_btn.grid(row=0, column=0, pady=10)
        
        # Barre de progression globale
        global_progress_label = ttk.Label(control_frame, text="Progression globale:")
        global_progress_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 2))
        
        self.global_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.global_progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.global_label = ttk.Label(control_frame, text="En attente...")
        self.global_label.grid(row=3, column=0, sticky=tk.W)
        
        # Barre de progression détaillée
        detail_progress_label = ttk.Label(control_frame, text="Progression détaillée:")
        detail_progress_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        
        self.detail_progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.detail_progress.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.detail_label = ttk.Label(control_frame, text="")
        self.detail_label.grid(row=6, column=0, sticky=tk.W)
        
        # Zone de logs
        log_header_frame = ttk.Frame(control_frame)
        log_header_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(15, 5))
        log_header_frame.columnconfigure(0, weight=1)

        log_label = ttk.Label(log_header_frame, text="Logs:", font=('Arial', 10, 'bold'))
        log_label.grid(row=0, column=0, sticky=tk.W)

        copy_logs_btn = ttk.Button(log_header_frame, text="📋 Copier", command=self.copy_logs)
        copy_logs_btn.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        # Frame pour les logs avec scrollbar
        log_frame = ttk.Frame(control_frame)
        log_frame.grid(row=8, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        control_frame.rowconfigure(8, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=70, 
                                                  bg='#2b2b2b', fg='#00ff00',
                                                  font=('Consolas', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tags pour la coloration des logs
        self.log_text.tag_config('INFO', foreground='#00ff00')
        self.log_text.tag_config('WARNING', foreground='#ffaa00')
        self.log_text.tag_config('ERROR', foreground='#ff5555')
        self.log_text.tag_config('SUCCESS', foreground='#55ff55')
        
        # Bouton pour ouvrir le dossier de sortie
        self.open_folder_btn = ttk.Button(control_frame, text="📁 Ouvrir le dossier de sortie",
                                         command=self.open_output_folder)
        self.open_folder_btn.grid(row=9, column=0, pady=(5, 0))
    
    def log(self, message, level='INFO'):
        """Ajouter un message au log avec timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)

    def clear_urls(self):
        """Effacer le contenu de la zone de texte des URLs"""
        self.url_text.delete('1.0', tk.END)

    def copy_logs(self):
        """Copier le contenu des logs dans le presse-papier"""
        logs_content = self.log_text.get('1.0', tk.END)
        if logs_content.strip():  # Vérifier qu'il y a du contenu
            self.root.clipboard_clear()
            self.root.clipboard_append(logs_content)
            self.root.update()  # Mettre à jour le presse-papier
            messagebox.showinfo("Succès", "Les logs ont été copiés dans le presse-papier!")
        else:
            messagebox.showwarning("Vide", "Il n'y a aucun log à copier.")

    def start_transcription(self):
        """Démarrer le processus de transcription dans un thread séparé"""
        if self.is_processing:
            messagebox.showwarning("En cours", "Une transcription est déjà en cours!")
            return
        
        # Récupérer les URLs
        urls_text = self.url_text.get('1.0', tk.END).strip()
        urls = [url.strip() for url in urls_text.split('\n') if url.strip() and 'youtube.com' in url or 'youtu.be' in url]
        
        if not urls:
            messagebox.showerror("Erreur", "Veuillez entrer au moins une URL YouTube valide!")
            return
        
        self.is_processing = True
        self.transcribe_btn.config(state='disabled')
        
        # Réinitialiser les barres de progression
        self.global_progress['maximum'] = len(urls)
        self.global_progress['value'] = 0
        
        # Lancer la transcription dans un thread
        thread = threading.Thread(target=self.process_videos, args=(urls,))
        thread.daemon = True
        thread.start()
    
    def process_videos(self, urls):
        """Traiter toutes les vidéos"""
        try:
            # Initialiser le transcripteur
            self.message_queue.put(('log', 'Initialisation de Whisper...', 'INFO'))
            self.message_queue.put(('detail', 'Chargement du modèle Whisper large-v3...', 0))
            
            self.transcriber = YouTubeTranscriber(
                output_dir=self.output_dir,
                model_size='large-v3',
                device=get_best_device(),  # Auto-détection: cuda (NVIDIA) / mps (M1) / cpu
                message_queue=self.message_queue
            )
            
            self.message_queue.put(('log', f'Modèle chargé avec succès! GPU détecté: {self.transcriber.device}', 'SUCCESS'))
            
            successful = 0
            failed = 0
            
            for i, url in enumerate(urls, 1):
                self.message_queue.put(('global', f'Traitement vidéo {i}/{len(urls)}', i-1))
                self.message_queue.put(('log', f'\n--- Vidéo {i}/{len(urls)} ---', 'INFO'))
                self.message_queue.put(('log', f'URL: {url}', 'INFO'))
                
                success = self.transcriber.process_video(url)
                
                if success:
                    successful += 1
                    self.message_queue.put(('log', f'✓ Vidéo {i} transcrite avec succès!', 'SUCCESS'))
                else:
                    failed += 1
                    self.message_queue.put(('log', f'✗ Échec de la vidéo {i}', 'ERROR'))
                
                self.message_queue.put(('global', f'Vidéo {i}/{len(urls)} terminée', i))
            
            # Résumé final
            self.message_queue.put(('log', f'\n========== RÉSUMÉ ==========', 'INFO'))
            self.message_queue.put(('log', f'Réussies: {successful}/{len(urls)}', 'SUCCESS'))
            if failed > 0:
                self.message_queue.put(('log', f'Échouées: {failed}/{len(urls)}', 'WARNING'))
            self.message_queue.put(('log', f'Fichiers sauvegardés dans: {self.output_dir.absolute()}', 'INFO'))
            
            self.message_queue.put(('detail', 'Transcription terminée!', 100))
            
        except Exception as e:
            self.message_queue.put(('log', f'Erreur critique: {str(e)}', 'ERROR'))
        finally:
            self.message_queue.put(('finished', '', ''))
    
    def check_queue(self):
        """Vérifier la queue pour les messages du thread de transcription"""
        try:
            while True:
                msg_type, content, extra = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    self.log(content, extra)
                elif msg_type == 'global':
                    self.global_label.config(text=content)
                    self.global_progress['value'] = extra
                elif msg_type == 'detail':
                    self.detail_label.config(text=content)
                    if extra == 0:
                        self.detail_progress.start(10)
                    elif extra == 100:
                        self.detail_progress.stop()
                    else:
                        if self.detail_progress['mode'] == 'indeterminate':
                            self.detail_progress.stop()
                            self.detail_progress['mode'] = 'determinate'
                        self.detail_progress['value'] = extra
                elif msg_type == 'finished':
                    self.is_processing = False
                    self.transcribe_btn.config(state='normal')
                    self.detail_progress.stop()
                    messagebox.showinfo("Terminé", "Toutes les transcriptions sont terminées!")
                    
        except queue.Empty:
            pass
        
        # Revérifier dans 100ms
        self.root.after(100, self.check_queue)
    
    def open_output_folder(self):
        """Ouvrir le dossier de sortie dans l'explorateur de fichiers"""
        import os
        import platform
        
        system = platform.system()
        if system == 'Windows':
            os.startfile(self.output_dir)
        elif system == 'Darwin':  # macOS
            os.system(f'open "{self.output_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{self.output_dir}"')

def main():
    root = tk.Tk()
    app = TranscriberGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()