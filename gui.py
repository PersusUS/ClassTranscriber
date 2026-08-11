"""ClassTranscriber — ventana de escritorio.

Interfaz mínima en Tkinter (incluido en Python, sin dependencias nuevas ni
navegador) para grabar una clase y obtener la transcripción del profesor
sin tocar el terminal.

    python gui.py

El trabajo pesado corre en un hilo aparte y se comunica con la ventana por
una cola: Tkinter no es seguro desde varios hilos, así que solo el hilo
principal toca los widgets.
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    HORIZONTAL,
    LEFT,
    RIGHT,
    W,
    X,
    BooleanVar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    scrolledtext,
    ttk,
)

from dotenv import load_dotenv

import config
import main as cli
from modules import gui_state, pipeline
from modules.gui_state import STAGE_LABELS, STATUS_MARKS, FormState

load_dotenv()

logger = logging.getLogger("classtranscriber.gui")

POLL_MS = 100          # How often the window drains the worker's message queue
PADDING = {"padx": 8, "pady": 4}


class ClassTranscriberApp:
    """The whole window."""

    def __init__(self, root: Tk):
        self.root = root
        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.outputs: dict[str, Path] = {}
        self.stage_labels: dict[str, ttk.Label] = {}
        self.devices: list[dict] = []

        root.title("ClassTranscriber")
        root.minsize(620, 620)

        self._build_form()
        self._build_meter()
        self._build_stages()
        self._build_log()
        self._build_buttons()

        self._load_devices()
        self.name_var.set(gui_state.default_session_name())
        self.root.after(POLL_MS, self._drain)

    # -- Construcción de la interfaz ---------------------------------------

    def _build_form(self) -> None:
        frame = ttk.LabelFrame(self.root, text="La clase")
        frame.pack(fill=X, **PADDING)

        ttk.Label(frame, text="Nombre").grid(row=0, column=0, sticky=W, **PADDING)
        self.name_var = StringVar()
        ttk.Entry(frame, textvariable=self.name_var, width=34).grid(
            row=0, column=1, columnspan=2, sticky=W, **PADDING
        )

        ttk.Label(frame, text="Micrófono").grid(row=1, column=0, sticky=W, **PADDING)
        self.device_var = StringVar()
        self.device_box = ttk.Combobox(
            frame, textvariable=self.device_var, width=32, state="readonly"
        )
        self.device_box.grid(row=1, column=1, columnspan=2, sticky=W, **PADDING)

        ttk.Label(frame, text="Duración (min)").grid(row=2, column=0, sticky=W, **PADDING)
        self.duration_var = StringVar(value="")
        ttk.Entry(frame, textvariable=self.duration_var, width=8).grid(
            row=2, column=1, sticky=W, **PADDING
        )
        ttk.Label(frame, text="vacío = hasta que pulses Parar").grid(
            row=2, column=2, sticky=W, **PADDING
        )

        ttk.Label(frame, text="Personas que hablan").grid(row=3, column=0, sticky=W, **PADDING)
        self.speakers_var = StringVar(value="")
        ttk.Entry(frame, textvariable=self.speakers_var, width=8).grid(
            row=3, column=1, sticky=W, **PADDING
        )
        ttk.Label(frame, text="opcional, mejora la separación").grid(
            row=3, column=2, sticky=W, **PADDING
        )

        ttk.Label(frame, text="Consumo").grid(row=4, column=0, sticky=W, **PADDING)
        self.profile_var = StringVar(value=config.DEFAULT_PROFILE)
        ttk.Combobox(
            frame,
            textvariable=self.profile_var,
            values=["auto", "low", "balanced", "quality"],
            width=12,
            state="readonly",
        ).grid(row=4, column=1, sticky=W, **PADDING)
        ttk.Label(frame, text="low = más rápido y menos batería").grid(
            row=4, column=2, sticky=W, **PADDING
        )

        options = ttk.Frame(frame)
        options.grid(row=5, column=0, columnspan=3, sticky=W)
        self.denoise_var = BooleanVar(value=True)
        self.diarize_var = BooleanVar(value=True)
        self.clean_var = BooleanVar(value=True)
        ttk.Checkbutton(options, text="Reducir el ruido del aula",
                        variable=self.denoise_var).pack(anchor=W, **PADDING)
        ttk.Checkbutton(options, text="Separar voces del profesor y los alumnos",
                        variable=self.diarize_var).pack(anchor=W, **PADDING)
        ttk.Checkbutton(options, text="Corregir el texto con IA",
                        variable=self.clean_var).pack(anchor=W, **PADDING)

    def _build_meter(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Nivel de entrada")
        frame.pack(fill=X, **PADDING)

        self.meter = ttk.Progressbar(frame, orient=HORIZONTAL, mode="determinate", maximum=1.0)
        self.meter.pack(fill=X, **PADDING)

        self.meter_text = StringVar(value="Sin grabar")
        ttk.Label(frame, textvariable=self.meter_text).pack(anchor=W, **PADDING)

    def _build_stages(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Progreso")
        frame.pack(fill=X, **PADDING)

        for row, (key, label) in enumerate(STAGE_LABELS):
            widget = ttk.Label(frame, text=f"{STATUS_MARKS['pending']}  {label}")
            widget.grid(row=row // 4, column=row % 4, sticky=W, **PADDING)
            self.stage_labels[key] = widget

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Detalle")
        frame.pack(fill=BOTH, expand=True, **PADDING)

        self.log = scrolledtext.ScrolledText(frame, height=9, wrap="word", state="disabled")
        self.log.pack(fill=BOTH, expand=True, **PADDING)

    def _build_buttons(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=X, **PADDING)

        self.record_button = ttk.Button(frame, text="Grabar la clase", command=self.start_recording)
        self.record_button.pack(side=LEFT, **PADDING)

        self.stop_button = ttk.Button(
            frame, text="Parar y procesar", command=self.stop_recording, state="disabled"
        )
        self.stop_button.pack(side=LEFT, **PADDING)

        self.file_button = ttk.Button(
            frame, text="Procesar un audio…", command=self.process_file
        )
        self.file_button.pack(side=LEFT, **PADDING)

        self.open_button = ttk.Button(
            frame, text="Abrir transcripción", command=self.open_result, state="disabled"
        )
        self.open_button.pack(side=RIGHT, **PADDING)

    # -- Utilidades de la interfaz -----------------------------------------

    def _load_devices(self) -> None:
        """Fills the microphone dropdown, tolerating a machine with none."""
        try:
            self.devices = cli.list_input_devices()
        except Exception as exc:      # noqa: BLE001 - PortAudio varies by OS
            self.devices = []
            self._append_log(f"No se pudo leer la lista de micrófonos: {exc}")

        labels = [gui_state.describe_device(device) for device in self.devices]
        self.device_box["values"] = labels
        if labels:
            default = next(
                (label for label, device in zip(labels, self.devices) if device["default"]),
                labels[0],
            )
            self.device_var.set(default)
        else:
            self.device_box["values"] = ["sin micrófono detectado"]
            self.device_var.set("sin micrófono detectado")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text.rstrip() + "\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _set_stage(self, key: str, status: str) -> None:
        label = self.stage_labels.get(key)
        if label is None:
            return
        text = dict(STAGE_LABELS).get(key, key)
        label.configure(text=f"{STATUS_MARKS.get(status, '·')}  {text}")

    def _set_running(self, recording: bool, working: bool) -> None:
        self.record_button.configure(state="disabled" if working else "normal")
        self.file_button.configure(state="disabled" if working else "normal")
        self.stop_button.configure(state="normal" if recording else "disabled")

    def read_form(self) -> FormState:
        """Reads the widgets into a `FormState`, validating as it goes.

        Raises:
            ValueError: If the duration or speaker count cannot be parsed.
        """
        return FormState(
            name=self.name_var.get(),
            device=gui_state.device_index_from_label(self.device_var.get()),
            profile=self.profile_var.get(),
            duration_minutes=gui_state.parse_duration_minutes(self.duration_var.get()),
            num_speakers=gui_state.parse_speakers(self.speakers_var.get()),
            denoise=self.denoise_var.get(),
            diarize=self.diarize_var.get(),
            clean=self.clean_var.get(),
        )

    # -- Acciones -----------------------------------------------------------

    def start_recording(self) -> None:
        """Grabar la clase: records now, then runs the whole pipeline."""
        self._start(input_path=None)

    def process_file(self) -> None:
        """Procesar un audio: skips recording and uses an existing file."""
        chosen = filedialog.askopenfilename(
            title="Elige la grabación de la clase",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.aiff"), ("Todos", "*.*")],
        )
        if chosen:
            self._start(input_path=Path(chosen))

    def _start(self, input_path: Path | None) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        try:
            form = self.read_form()
        except ValueError as exc:
            messagebox.showerror("Revisa los datos", str(exc))
            return

        form.input_path = input_path

        token = os.getenv("HF_TOKEN")
        ollama_ok, _ = cli.check_ollama()
        problems = gui_state.missing_requirements(form, token, ollama_ok)
        if problems:
            messagebox.showerror("Falta algo por configurar", "\n\n".join(problems))
            return

        if not self.devices and input_path is None:
            messagebox.showerror(
                "Sin micrófono",
                "No se ha detectado ningún micrófono. Conecta uno y reabre la ventana, "
                "o usa «Procesar un audio…» con una grabación que ya tengas.",
            )
            return

        self.stop_event = threading.Event()
        self.outputs = {}
        for key, status in gui_state.initial_stages(form).items():
            self._set_stage(key, status)
        self.open_button.configure(state="disabled")
        self.meter.configure(value=0.0)
        self.meter_text.set("Preparando…")
        self._append_log(f"Sesión «{gui_state.sanitise_name(form.name)}» — {form.profile}")
        self._set_running(recording=input_path is None, working=True)

        options = gui_state.build_options(
            form,
            hf_token=token,
            stop_event=self.stop_event,
            on_level=self._on_level,
            on_stage=self._on_stage,
        )

        self.worker = threading.Thread(target=self._run, args=(options,), daemon=True)
        self.worker.start()

    def stop_recording(self) -> None:
        """Parar y procesar: ends the recording, keeps everything captured."""
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.meter_text.set("Grabación detenida, procesando…")
        self._append_log("Grabación detenida. Procesando lo grabado.")

    def open_result(self) -> None:
        """Opens the professor-only transcript in the system's text editor."""
        target = self.outputs.get("professor") or self.outputs.get("full")
        if target is None:
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))       # noqa: S606 - Windows' documented API
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                webbrowser.open(target.as_uri())
        except Exception as exc:      # noqa: BLE001
            messagebox.showinfo("Transcripción guardada", f"{target}\n\n({exc})")

    # -- Hilo de trabajo ----------------------------------------------------

    def _run(self, options: pipeline.PipelineOptions) -> None:
        """Runs the pipeline. Executes in the worker thread, never in the UI."""
        try:
            outputs = pipeline.run_pipeline(options)
            self.messages.put(("done", outputs))
        except Exception as exc:      # noqa: BLE001 - reported in the window
            logger.exception("La sesión ha fallado")
            self.messages.put(("failed", exc))

    def _on_level(self, elapsed: float, level_dbfs: float, peak_dbfs: float) -> None:
        self.messages.put(("level", (elapsed, level_dbfs, peak_dbfs)))

    def _on_stage(self, key: str, status: str) -> None:
        self.messages.put(("stage", (key, status)))

    def _drain(self) -> None:
        """Applies queued worker messages to the widgets, in the UI thread."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "level":
                    self._apply_level(*payload)
                elif kind == "stage":
                    self._apply_stage(*payload)
                elif kind == "done":
                    self._apply_done(payload)
                elif kind == "failed":
                    self._apply_failure(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(POLL_MS, self._drain)

    def _apply_level(self, elapsed: float, level_dbfs: float, peak_dbfs: float) -> None:
        self.meter.configure(value=gui_state.meter_fraction(level_dbfs))
        _, message = gui_state.meter_verdict(peak_dbfs)
        self.meter_text.set(f"{gui_state.format_elapsed(elapsed)} — {message}")

    def _apply_stage(self, key: str, status: str) -> None:
        self._set_stage(key, status)
        if key == "record" and status == "done":
            self._set_running(recording=False, working=True)
            self.meter.configure(value=0.0)
        if status == "running":
            self._append_log(f"{dict(STAGE_LABELS).get(key, key)}…")

    def _apply_done(self, outputs: dict[str, Path]) -> None:
        self.outputs = outputs
        self._set_running(recording=False, working=False)
        self.meter_text.set("Listo")
        for label, path in outputs.items():
            self._append_log(f"{label}: {path}")
        if outputs.get("professor") or outputs.get("full"):
            self.open_button.configure(state="normal")

    def _apply_failure(self, error: Exception) -> None:
        self._set_running(recording=False, working=False)
        self.meter_text.set("Ha fallado")
        self._append_log(f"ERROR: {error}")
        messagebox.showerror("La sesión ha fallado", str(error))


def build_app(root: Tk) -> ClassTranscriberApp:
    """Creates the application on an existing Tk root. Used by the tests."""
    return ClassTranscriberApp(root)


def run() -> None:
    """Opens the window."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    for directory in (config.AUDIO_DIR, config.OUTPUT_DIR, config.SESSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    root = Tk()
    build_app(root)
    root.mainloop()


if __name__ == "__main__":
    run()
