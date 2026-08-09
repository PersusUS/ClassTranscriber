# ClassTranscriber

Graba una clase, separa las voces, transcribe con Whisper y genera **la
transcripción limpia del profesor solo**, lista para hacer apuntes o
pasársela a un LLM.

Todo funciona en local: nada sale del portátil.

---

## Qué produce

Para una sesión llamada `algebra_01`:

| Fichero | Contenido |
|---|---|
| `output/algebra_01_profesor.txt` | **Solo el profesor**, en prosa continua, sin etiquetas, con marcas de tiempo cada 5 min. Este es el que quieres para estudiar. |
| `output/algebra_01_completo.txt` | Transcripción completa con todos los hablantes y marcas de tiempo. |
| `output/algebra_01.md` | Lo mismo en Markdown, con los turnos del profesor destacados. |
| `output/algebra_01.srt` | Subtítulos, para volver a escuchar un momento concreto. |
| `sessions/algebra_01/` | Audio y resultados intermedios, para reanudar sin repetir trabajo. |

---

## Requisitos

- Python 3.11
- **No hace falta GPU.** Si tienes una NVIDIA con CUDA 12 se usa sola.
- [Ollama](https://ollama.com) para la corrección final (opcional: `--no-clean`).
- Cuenta gratuita de HuggingFace para separar voces (opcional: `--no-diarize`).

Sin GPU, en un portátil de 4 núcleos, el perfil `balanced` procesa una clase
de 2 h en aproximadamente 1–2 h. Con `--profile low` baja a menos de la mitad,
a cambio de algo de precisión.

---

## Instalación

### 1. PyTorch (primero, y eligiendo la versión correcta)

Solo CPU — es lo que quieres en un portátil sin gráfica NVIDIA:

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

Con GPU NVIDIA y CUDA 12:

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. El resto de dependencias

```bash
pip install -r requirements.txt
```

### 3. Token de HuggingFace (para separar voces)

1. Crea una cuenta en <https://huggingface.co>.
2. Acepta la licencia en <https://huggingface.co/pyannote/speaker-diarization-3.1>
   y en <https://huggingface.co/pyannote/segmentation-3.0>. Es gratis y se
   hace una sola vez.
3. Crea un token de lectura en <https://huggingface.co/settings/tokens>.
4. Copia la plantilla y pega el token:

```bash
cp .env.example .env      # en Windows: copy .env.example .env
```

```
HF_TOKEN=hf_tu_token_aqui
```

Una vez descargado, el modelo queda en caché y funciona sin conexión.

### 4. Ollama (para la corrección del texto)

```bash
ollama pull gemma3:4b     # con --profile low usa: ollama pull qwen2.5:3b
ollama serve
```

### 5. Comprueba que todo está en su sitio

```bash
python main.py doctor
```

Te dice qué perfil se va a usar, si detecta CUDA, cuántos hilos usará, qué
micrófono hay y qué falta por configurar. No falla: solo informa.

---

## Uso

### Una clase entera

```bash
python main.py run --name algebra_01 --duration 5400
```

Graba 5400 s (90 min) y ejecuta todo el proceso. **Ctrl+C corta la grabación
antes de tiempo y sigue con lo grabado**, así que puedes poner una duración
generosa sin miedo.

### Si ya tienes el audio grabado (móvil, grabadora, Zoom…)

```bash
python main.py run --name algebra_01 --input ~/grabaciones/clase.wav
```

Acepta cualquier formato que lea `soundfile` (WAV, FLAC, OGG) en cualquier
frecuencia de muestreo: se convierte a mono 16 kHz automáticamente. Para MP3 o
M4A, conviértelos antes con ffmpeg:

```bash
ffmpeg -i clase.m4a -ac 1 -ar 16000 clase.wav
```

### Opciones que de verdad importan

```bash
# Portátil justo de recursos, o clase que no necesita máxima precisión
python main.py run --name clase --profile low

# Sabes cuánta gente habló: mejora bastante la separación de voces
python main.py run --name clase --num-speakers 3

# Sin token de HuggingFace: transcribe todo sin separar voces
python main.py run --name clase --no-diarize

# Sin Ollama: transcripción en bruto, sin corrección
python main.py run --name clase --no-clean

# Falló algo a mitad: reanuda sin repetir lo ya hecho
python main.py run --name clase --resume

# Sin preguntas interactivas (para dejarlo desatendido)
python main.py run --name clase --yes
```

### Elegir micrófono

```bash
python main.py devices
python main.py run --name clase --mic 2
```

### Quién es el profesor

Por defecto se identifica como **el que más habla**, que en una clase acierta
casi siempre. Si el reparto está ajustado (un seminario, mucho debate), el
programa te enseña una tabla con el tiempo de cada voz y un par de frases de
cada una para que elijas. Para saltarte la pregunta:

```bash
python main.py run --name clase --professor auto        # siempre el que más habla
python main.py run --name clase --professor SPEAKER_01  # una etiqueta concreta
python main.py run --name clase --professor none        # no filtrar por hablante
```

---

## Consejos para aulas ruidosas

El mayor salto de calidad no está en el software, está en el micrófono. Por
orden de impacto real:

1. **Acerca el micrófono al profesor.** Un micro de solapa o direccional de
   20 € cambia el resultado más que cualquier ajuste. El ruido cae con el
   cuadrado de la distancia; el filtrado digital no puede competir con eso.
2. **Siéntate delante.** Con el portátil, mismo argumento.
3. **Comprueba el nivel.** Durante la grabación se registra el nivel de
   entrada cada 30 s y avisa si está demasiado bajo (micro lejos o silenciado)
   o saturando.
4. **Mira el aviso de SNR.** Al preprocesar se calcula la relación
   señal/ruido; por debajo de 10 dB la separación de voces empeora bastante y
   conviene replantear la colocación del micro.
5. Si el aula está tranquila, `--no-denoise` va más rápido y evita cualquier
   artefacto del filtrado.

---

## Comandos por separado

Cada etapa se puede ejecutar sola; todas guardan JSON reutilizable.

```bash
python main.py record     --output audio/clase.wav --duration 3600
python main.py preprocess --input audio/clase.wav --output audio/clase_limpio.wav
python main.py transcribe --input audio/clase_limpio.wav --output tr.json
python main.py diarize    --input audio/clase_limpio.wav --output dia.json --num-speakers 3
python main.py merge      --diarization dia.json --transcription tr.json --output merged.json
python main.py clean      --input merged.json --professor auto
python main.py export     --input merged.cleaned.json --name clase --formats txt,professor,md,srt
python main.py clear      --name clase          # borra los intermedios de una sesión
```

---

## Perfiles

| Perfil | Whisper | LLM | Cuándo usarlo |
|---|---|---|---|
| `low` | `small` (int8) | `qwen2.5:3b` | Portátil justo, batería, resultado rápido. |
| `balanced` | `large-v3-turbo` (int8) | `gemma3:4b` | Por defecto. Casi la calidad de `large-v3` a una fracción del coste. |
| `quality` | `large-v3` | `gemma3:4b` | Solo merece la pena con GPU. |

`auto` (por defecto) elige `quality` si hay GPU con 6 GB o más de VRAM, y
`balanced` en cualquier otro caso. Se puede fijar en `.env` con `CT_PROFILE`.

---

## Cómo funciona

```
1. Grabar        micrófono -> WAV 16 kHz mono, escrito en streaming al disco
2. Preprocesar   paso alto 80 Hz -> puerta espectral -> normalización RMS
3. Transcribir   faster-whisper con VAD y filtro de alucinaciones
4. Separar voces pyannote (CPU o GPU)
5. Combinar      atribución palabra a palabra + agrupación en turnos
6. Corregir      LLM local, solo sobre el texto del profesor
7. Exportar      profesor / completo / markdown / subtítulos
```

Algunas decisiones que explican por qué funciona en aulas ruidosas:

- **Normalización por RMS, no por pico.** En un aula el pico lo marca una
  silla arrastrada, no la voz. Normalizar por pico deja al profesor igual de
  bajo. El nivel de voz se estima con el percentil 75 de los bloques, que
  ignora tanto los silencios como los golpes.
- **El nivel se mide después del filtro paso alto**, para que el zumbido de
  la red eléctrica y el aire acondicionado no se cuenten como voz.
- **`condition_on_previous_text=False`.** Es lo que evita que Whisper entre
  en bucle repitiendo la misma frase cuando solo oye ruido.
- **Filtro de alucinaciones.** Sobre silencio Whisper inventa frases como
  «Subtítulos realizados por la comunidad de Amara.org»; se detectan y se
  descartan, junto con los segmentos de baja confianza.
- **Atribución palabra a palabra.** Whisper corta por pausas de respiración,
  no por hablante, así que un segmento puede acabar con la pregunta de un
  alumno. Asignando cada palabra por separado y partiendo el segmento donde
  cambia la voz, esa pregunta no acaba en el fichero del profesor.
- **Todo en streaming por bloques.** La memoria no depende de la duración:
  dos horas de clase ocupan lo mismo que dos minutos.

---

## Tests

```bash
pytest
```

Los tests no necesitan GPU, ni micrófono, ni modelos descargados.

## Estructura

```
ClassTranscriber/
├── main.py                 # CLI
├── config.py               # Perfiles y todos los ajustes
├── modules/
│   ├── audio_utils.py      # DSP y utilidades de streaming
│   ├── recorder.py         # M1 — captura de audio
│   ├── preprocessor.py     # M2 — ruido y normalización
│   ├── diarizer.py         # M3 — separación de voces
│   ├── transcriber.py      # M4 — Whisper
│   ├── merger.py           # M5 — alineación voz/texto
│   ├── speaker_id.py       # Identificación del profesor
│   ├── cleaner.py          # M6 — corrección con LLM
│   ├── exporter.py         # M7 — ficheros de salida
│   └── pipeline.py         # Orquestación y reanudación
├── tests/
├── audio/                  # Grabaciones sueltas (gitignored)
├── sessions/               # Intermedios por sesión (gitignored)
└── output/                 # Transcripciones finales (gitignored)
```
