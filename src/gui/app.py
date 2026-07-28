import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import soundfile as sf

from src.pipeline.notes_pipeline import NotesPipeline
from src.gui.recorder import AudioRecorder
from src.storage.db import Database
from src.gui import styles

class MeetNotesApp:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root = root
        self.root.title("MeetNotes — Edge Meeting Notes Maker")
        self.root.geometry("820x650")
        self.root.minsize(750, 580)
        
        # Load styles
        styles.apply_dark_theme(self.root)
        
        # Models and database
        self.model_path = model_path
        self.db = Database()
        self.recorder = AudioRecorder()
        self.pipeline = NotesPipeline(model_path)
        
        # Variables
        self.mic_enabled = tk.BooleanVar(value=True)
        self.sys_enabled = tk.BooleanVar(value=False)
        self.timer_running = False
        self.selected_recording_id = None
        self.playback_active = False
        
        self.build_ui()
        self.refresh_past_recordings()

    def build_ui(self):
        # 1. Main Header
        header = tk.Frame(self.root, bg=styles.BG_COLOR, height=70)
        header.pack(fill="x", padx=20, pady=(15, 10))
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header,
            text="MeetNotes",
            font=styles.TITLE_FONT,
            fg=styles.ACCENT_COLOR,
            bg=styles.BG_COLOR
        )
        title_label.pack(side="left", anchor="center")
        
        subtitle_label = tk.Label(
            header,
            text="• Offline CPU-based notes assistant",
            font=("Inter", 10),
            fg=styles.TEXT_MUTED,
            bg=styles.BG_COLOR
        )
        subtitle_label.pack(side="left", padx=10, pady=(4, 0), anchor="center")
        
        # 2. Main Tabbed Container
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        # Tab Frames
        self.tab_record = ttk.Frame(self.notebook, style="TFrame")
        self.tab_past = ttk.Frame(self.notebook, style="TFrame")
        self.tab_notes = ttk.Frame(self.notebook, style="TFrame")
        
        self.notebook.add(self.tab_record, text=" RECORD ")
        self.notebook.add(self.tab_past, text=" PAST RECORDINGS ")
        self.notebook.add(self.tab_notes, text=" NOTES ")

        
        self.build_record_tab()
        self.build_past_recordings_tab()
        self.build_notes_tab()

    # -------------------------------------------------------------------------
    # TAB 1: RECORD TAB
    # -------------------------------------------------------------------------
    def build_record_tab(self):
        # Layout: Left column (Config card), Right column (Controls card)
        self.tab_record.columnconfigure(0, weight=1)
        self.tab_record.columnconfigure(1, weight=1)
        self.tab_record.rowconfigure(0, weight=1)
        
        # Left Panel: Config Card
        config_card = ttk.Frame(self.tab_record, style="Card.TFrame")
        config_card.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="nsew")
        
        config_title = ttk.Label(config_card, text="Audio Source Setup", style="CardTitle.TLabel")
        config_title.pack(anchor="w", padx=20, pady=(20, 10))
        
        config_desc = ttk.Label(
            config_card,
            text="Select where you want to record the meeting audio. You can mix both microphone and computer output.",
            style="Card.TLabel",
            wraplength=300
        )
        config_desc.pack(anchor="w", padx=20, pady=(0, 20))
        
        # Audio checkbuttons with styling overrides
        self.chk_mic = tk.Checkbutton(
            config_card,
            text="Record Microphone (default mic)",
            variable=self.mic_enabled,
            bg=styles.CARD_BG,
            fg=styles.TEXT_MAIN,
            selectcolor=styles.BG_COLOR,
            activebackground=styles.CARD_BG,
            activeforeground=styles.TEXT_MAIN,
            font=styles.TEXT_FONT,
            bd=0
        )
        self.chk_mic.pack(anchor="w", padx=20, pady=10)
        
        self.chk_sys = tk.Checkbutton(
            config_card,
            text="Record System Audio (speakers loopback)",
            variable=self.sys_enabled,
            bg=styles.CARD_BG,
            fg=styles.TEXT_MAIN,
            selectcolor=styles.BG_COLOR,
            activebackground=styles.CARD_BG,
            activeforeground=styles.TEXT_MAIN,
            font=styles.TEXT_FONT,
            bd=0
        )
        self.chk_sys.pack(anchor="w", padx=20, pady=10)
        
        # Right Panel: Controls Card
        controls_card = ttk.Frame(self.tab_record, style="Card.TFrame")
        controls_card.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="nsew")
        
        controls_title = ttk.Label(controls_card, text="Meeting Control Room", style="CardTitle.TLabel")
        controls_title.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Timer Display
        self.timer_label = tk.Label(
            controls_card,
            text="00:00:00",
            font=("Outfit", 28, "bold"),
            fg=styles.TEXT_MUTED,
            bg=styles.CARD_BG
        )
        self.timer_label.pack(pady=(30, 10))
        
        # Live Volume Meter
        self.volume_bar = ttk.Progressbar(controls_card, length=200, mode="determinate", maximum=1.0)
        self.volume_bar.pack(pady=(0, 20))
        
        # Start/Stop Button
        self.btn_record = tk.Button(
            controls_card,
            text="Start Recording",
            font=("Inter", 12, "bold"),
            bg=styles.ACCENT_COLOR,
            fg=styles.TEXT_MAIN,
            activebackground=styles.ACCENT_HOVER,
            activeforeground=styles.TEXT_MAIN,
            bd=0,
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.toggle_recording
        )
        self.btn_record.pack(pady=10, fill="x", padx=40)
        
        # Sub-actions Frame (notes up to now, end recording)
        self.sub_action_frame = ttk.Frame(controls_card, style="Card.TFrame")
        self.sub_action_frame.pack(fill="x", padx=40, pady=10)
        
        self.btn_notes_now = tk.Button(
            self.sub_action_frame,
            text="Notes Up To Now",
            font=("Inter", 9, "bold"),
            bg=styles.CARD_BG,
            fg=styles.TEXT_MUTED,
            activebackground=styles.BORDER_COLOR,
            activeforeground=styles.TEXT_MAIN,
            bd=1,
            relief="solid",
            highlightthickness=0,
            cursor="hand2",
            state="disabled",
            command=self.generate_notes_up_to_now
        )
        self.btn_notes_now.pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=5)
        
        self.btn_end = tk.Button(
            self.sub_action_frame,
            text="End Meeting",
            font=("Inter", 9, "bold"),
            bg=styles.CARD_BG,
            fg=styles.TEXT_MUTED,
            activebackground=styles.BORDER_COLOR,
            activeforeground=styles.TEXT_MAIN,
            bd=1,
            relief="solid",
            highlightthickness=0,
            cursor="hand2",
            state="disabled",
            command=self.end_meeting
        )
        self.btn_end.pack(side="right", fill="x", expand=True, padx=(5, 0), ipady=5)

    # -------------------------------------------------------------------------
    # TAB 2: PAST RECORDINGS TAB
    # -------------------------------------------------------------------------
    def build_past_recordings_tab(self):
        # Treeview (table)
        cols = ("id", "filename", "duration", "source", "date")
        self.tree = ttk.Treeview(self.tab_past, columns=cols, show="headings", selectmode="browse")
        
        self.tree.heading("id", text="ID")
        self.tree.heading("filename", text="Name")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("source", text="Source")
        self.tree.heading("date", text="Recorded At")
        
        self.tree.column("id", width=40, anchor="center")
        self.tree.column("filename", width=250, anchor="w")
        self.tree.column("duration", width=80, anchor="center")
        self.tree.column("source", width=80, anchor="center")
        self.tree.column("date", width=150, anchor="center")
        
        self.tree.pack(fill="both", expand=True, padx=0, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_recording_selected)
        
        # Action Panel
        actions = ttk.Frame(self.tab_past, style="TFrame")
        actions.pack(fill="x", pady=10)
        
        self.btn_play = ttk.Button(actions, text="Play Audio", command=self.play_selected_audio, style="Muted.TButton")
        self.btn_play.pack(side="left", padx=(0, 10))
        
        self.btn_make_notes = ttk.Button(actions, text="Make Notes", command=self.generate_notes_selected)
        self.btn_make_notes.pack(side="left", padx=10)
        
        self.btn_delete = ttk.Button(actions, text="Delete", command=self.delete_selected_recording, style="Muted.TButton")
        self.btn_delete.pack(side="right")

    # -------------------------------------------------------------------------
    # TAB 3: NOTES VIEWER TAB
    # -------------------------------------------------------------------------
    def build_notes_tab(self):
        self.tab_notes.rowconfigure(0, weight=1)
        self.tab_notes.columnconfigure(0, weight=1)
        
        # Container frame
        container = ttk.Frame(self.tab_notes, style="Card.TFrame")
        container.grid(row=0, column=0, sticky="nsew", pady=10)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        
        # Title & Copy Button header
        header = tk.Frame(container, bg=styles.CARD_BG)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        
        self.notes_title = tk.Label(
            header,
            text="Structured Meeting Notes",
            font=("Outfit", 12, "bold"),
            fg=styles.TEXT_MAIN,
            bg=styles.CARD_BG
        )
        self.notes_title.pack(side="left")
        
        btn_copy = tk.Button(
            header,
            text="📋 Copy Notes",
            font=("Inter", 9, "bold"),
            bg=styles.ACCENT_COLOR,
            fg=styles.TEXT_MAIN,
            activebackground=styles.ACCENT_HOVER,
            activeforeground=styles.TEXT_MAIN,
            bd=0,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self.copy_notes_to_clipboard
        )
        btn_copy.pack(side="right")
        
        # Scrollable Text Area
        scrollbar = ttk.Scrollbar(container)
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        self.notes_text = tk.Text(
            container,
            bg=styles.BG_COLOR,
            fg=styles.TEXT_MAIN,
            insertbackground=styles.TEXT_MAIN,
            font=styles.MONOSPACE_FONT,
            bd=0,
            padx=15,
            pady=15,
            yscrollcommand=scrollbar.set,
            wrap="word"
        )
        self.notes_text.grid(row=1, column=0, sticky="nsew", padx=(20, 0), pady=(0, 20))
        scrollbar.config(command=self.notes_text.yview)

    # -------------------------------------------------------------------------
    # RECORDING ACTIONS
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        if not self.recorder.is_recording:
            # Start
            if not (self.mic_enabled.get() or self.sys_enabled.get()):
                messagebox.showerror("Setup Error", "You must select at least one audio source.")
                return
                
            try:
                self.recorder.start(
                    record_mic=self.mic_enabled.get(),
                    record_sys=self.sys_enabled.get()
                )
                self.btn_record.config(text="Stop Recording", bg="#EF4444")  # red
                self.chk_mic.config(state="disabled")
                self.chk_sys.config(state="disabled")
                self.btn_notes_now.config(state="normal")
                self.btn_end.config(state="normal")
                
                # Start timer thread
                self.timer_running = True
                threading.Thread(target=self.update_timer_loop, daemon=True).start()
                self.update_status("Recording meeting...")
            except Exception as e:
                messagebox.showerror("Recording Error", str(e))
        else:
            # Stop
            self.stop_recording_and_save()

    def update_timer_loop(self):
        while self.timer_running and self.recorder.is_recording:
            elapsed = self.recorder.get_elapsed_seconds()
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            timer_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            
            self.root.after(0, self.timer_label.config, {"text": timer_str, "fg": "#EF4444"})
            
            # Update volume meter
            try:
                vol = self.recorder.get_current_volume()
                vol = min(1.0, vol * 2.5) # amplify visually
                self.root.after(0, lambda v=vol: self.volume_bar.config(value=v))
            except: pass
            
            time.sleep(0.1)

    def stop_recording_and_save(self) -> str:
        self.timer_running = False
        self.timer_label.config(fg=styles.TEXT_MUTED)
        
        # Save path config
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_{timestamp}.wav"
        save_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        # Stop recorder
        duration = self.recorder.stop(save_path=save_path)
        
        # Reset UI
        self.btn_record.config(text="Start Recording", bg=styles.ACCENT_COLOR)
        self.chk_mic.config(state="normal")
        self.chk_sys.config(state="normal")
        self.btn_notes_now.config(state="disabled")
        self.btn_end.config(state="disabled")
        
        if duration > 0 and os.path.exists(save_path):
            source_desc = "Mixed" if (self.mic_enabled.get() and self.sys_enabled.get()) else ("Mic" if self.mic_enabled.get() else "System")
            # Save to Database
            rec_id = self.db.add_recording(
                filename=filename,
                path=save_path,
                duration_sec=duration,
                source=source_desc
            )
            self.refresh_past_recordings()
            self.update_status(f"Recording saved: {filename}")
            return save_path
        return ""

    def generate_notes_up_to_now(self):
        """
        Retrieves the buffer of audio recorded so far, saves it, and compiles notes
        without stopping the main meeting recording.
        """
        temp_dir = os.path.join(os.getcwd(), "recordings")
        temp_wav = os.path.join(temp_dir, "temp_snapshot.wav")
        
        success = self.recorder.get_current_wav_snapshot(temp_wav)
        if not success:
            messagebox.showwarning("Buffer Empty", "Not enough audio recorded yet.")
            return
            
        self.update_status("Running pipeline on current audio buffer...")
        # Launch pipeline in background thread
        threading.Thread(target=self.run_pipeline_thread, args=(temp_wav, None), daemon=True).start()

    def end_meeting(self):
        """
        Stops the recording, saves it, and immediately prompts the user to generate notes.
        """
        wav_path = self.stop_recording_and_save()
        if not wav_path:
            return
            
        ans = messagebox.askyesno("Meeting Ended", "Would you like to compile notes for this meeting now?")
        if ans:
            # Switch to notes tab, find the newly added recording row index
            self.notebook.select(self.tab_notes)
            # Find the ID of the recording we just saved (the most recent in DB)
            recordings = self.db.get_all_recordings()
            if recordings:
                rec_id = recordings[0]["id"]
                self.run_notes_generation(wav_path, rec_id)

    # -------------------------------------------------------------------------
    # DATABASE & TABLE UI ACTIONS
    # -------------------------------------------------------------------------
    def refresh_past_recordings(self):
        # Clear
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Load all from db
        recordings = self.db.get_all_recordings()
        for r in recordings:
            duration_str = f"{int(r['duration_sec'] // 60)}m {int(r['duration_sec'] % 60)}s"
            
            # Format filename to "Rec YYYY-MM-DD HH:MM" based on created_at
            # created_at is usually "YYYY-MM-DD HH:MM:SS"
            dt = r["created_at"][:16] if r["created_at"] else "Unknown"
            formatted_name = f"Rec {dt}"
            
            self.tree.insert("", "end", values=(
                r["id"],
                formatted_name,
                duration_str,
                r["source"],
                r["created_at"]
            ))

    def on_recording_selected(self, event):
        selected = self.tree.selection()
        if selected:
            item = self.tree.item(selected[0])
            self.selected_recording_id = item["values"][0]
            
            # Dynamically update the button
            notes_entry = self.db.get_notes_for_recording(self.selected_recording_id)
            if notes_entry:
                self.btn_make_notes.config(text="See Notes")
            else:
                self.btn_make_notes.config(text="Make Notes")
        else:
            self.selected_recording_id = None
            self.btn_make_notes.config(text="Make Notes")

    def play_selected_audio(self):
        if not self.selected_recording_id:
            messagebox.showwarning("Select Recording", "Please select a recording to play.")
            return
            
        rec = self.db.get_recording_by_id(self.selected_recording_id)
        if not rec:
            return
            
        wav_path = rec["path"]
        if not os.path.exists(wav_path):
            messagebox.showerror("File Error", f"Audio file not found at: {wav_path}")
            return
            
        if self.playback_active:
            # Stop playback
            sd.stop()
            self.playback_active = False
            self.btn_play.config(text="Play Audio")
            self.update_status("Playback stopped", 0.0)
        else:
            # Play in background thread
            self.playback_active = True
            self.btn_play.config(text="Stop Playback")
            self.update_status(f"Playing...")
            
            def _play_thread():
                try:
                    data, fs = sf.read(wav_path)
                    sd.play(data, fs)
                    sd.wait()
                except Exception as e:
                    self.root.after(0, messagebox.showerror, "Playback Error", str(e))
                finally:
                    self.playback_active = False
                    self.root.after(0, self.btn_play.config, {"text": "Play Audio"})
                    self.root.after(0, self.update_status, "Playback finished")
                    
            threading.Thread(target=_play_thread, daemon=True).start()

    def delete_selected_recording(self):
        if not self.selected_recording_id:
            messagebox.showwarning("Select Recording", "Please select a recording to delete.")
            return
            
        ans = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this recording and its notes permanently?")
        if ans:
            # Stop playback if active
            if self.playback_active:
                sd.stop()
                
            success = self.db.delete_recording(self.selected_recording_id)
            if success:
                self.refresh_past_recordings()
                self.selected_recording_id = None
                self.update_status("Recording deleted successfully")
            else:
                messagebox.showerror("Error", "Could not delete recording.")

    def generate_notes_selected(self):
        if not self.selected_recording_id:
            messagebox.showwarning("Select Recording", "Please select a recording.")
            return
            
        rec = self.db.get_recording_by_id(self.selected_recording_id)
        if not rec:
            return
            
        # Check if notes already exist in db
        notes_entry = self.db.get_notes_for_recording(self.selected_recording_id)
        if notes_entry:
            # Load notes into Viewer immediately
            self.display_notes(notes_entry["notes_text"])
            self.notebook.select(self.tab_notes)
            return
                
        wav_path = rec["path"]
        self.notebook.select(self.tab_notes)
        self.run_notes_generation(wav_path, self.selected_recording_id)

    # -------------------------------------------------------------------------
    # PIPELINE THREAD EXECUTION
    # -------------------------------------------------------------------------
    def run_notes_generation(self, wav_path: str, recording_id: int):
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("end", "Starting notes compilation engine...\nPlease wait, loading models...")
        
        # Start pipeline process in background thread
        threading.Thread(
            target=self.run_pipeline_thread,
            args=(wav_path, recording_id),
            daemon=True
        ) .start()

    def run_pipeline_thread(self, wav_path: str, recording_id: int):
        def pipeline_callback(stage_text: str, percentage: float):
            def _append_text():
                self.notes_text.insert("end", f"{stage_text}\n")
                self.notes_text.see("end")
            self.root.after(0, _append_text)
            
        # Execute orchestrator
        final_notes = self.pipeline.process_audio(wav_path, progress_callback=pipeline_callback)
        
        # Save results to DB and text files
        if recording_id and "Error" not in final_notes:
            # Check if notes exist and delete to avoid duplicate
            existing = self.db.get_notes_for_recording(recording_id)
            if existing:
                # delete
                with self.db.get_connection() as conn:
                    conn.cursor().execute("DELETE FROM notes WHERE recording_id = ?", (recording_id,))
                    conn.commit()
            
            # Save notes in db
            # Calculate chunks based on separator
            chunk_count = len(final_notes.split("\n\n---\n\n"))
            self.db.add_notes(recording_id, final_notes, chunk_count)
            
            # Save notes in notes/ folder alongside WAV
            rec = self.db.get_recording_by_id(recording_id)
            if rec:
                notes_filename = rec["filename"].replace(".wav", "_notes.txt")
                notes_dir = os.path.join(os.getcwd(), "notes")
                os.makedirs(notes_dir, exist_ok=True)
                notes_path = os.path.join(notes_dir, notes_filename)
                
                try:
                    with open(notes_path, "w", encoding="utf-8") as f:
                        f.write(final_notes)
                    self.root.after(0, self.update_status, f"Saved notes to notes/{notes_filename}")
                except Exception as e:
                    print(f"Error saving notes text file: {e}")
                    
        # Update Viewer in GUI thread
        self.root.after(0, self.display_notes, final_notes)

    def display_notes(self, notes_content: str):
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("end", notes_content)

    def copy_notes_to_clipboard(self):
        content = self.notes_text.get("1.0", "end-1c").strip()
        if not content:
            return
            
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.update_status("Notes copied to clipboard!")

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def update_status(self, text: str, progress: float = 0.0):
        """
        Status updates are now routed to the console since the UI status bar is removed.
        """
        print(f"[Status] {text}")
            
    def cleanup(self):
        """
        Stops background processes during exit.
        """
        print("Cleaning up resources...")
        sd.stop()
        self.recorder.stop()
        self.pipeline.shutdown()
