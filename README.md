# Notulen ID

Pipeline notulen otomatis untuk rapat dan podcast berbahasa Indonesia.

> Status: dalam pengembangan. Bagian hasil, demo, dan analisis kesalahan akan dilengkapi setelah evaluasi selesai.

## Alur

```text
audio -> pra-proses (ffmpeg + VAD) -> ASR (faster-whisper) -> diarization (pyannote.audio)
      -> penggabungan kata dan pembicara -> ringkasan dan tindak lanjut (LLM) -> demo Gradio
```

## Instalasi di Windows (PowerShell)

Prasyarat: Anaconda atau Miniconda, Ollama, dan GPU NVIDIA (opsional).

```powershell
conda env create -f environment.yml
conda activate notulen
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,gpu]"
Copy-Item .env.example .env
python scripts/check_device.py
```

Tanpa GPU NVIDIA, lewati baris `pip install torch` dan pakai `pip install -e ".[dev]"`.

## Akses model pyannote

1. Buat token read di <https://huggingface.co/settings/tokens>, lalu simpan sebagai `HF_TOKEN` di `.env`.
2. Setujui syarat penggunaan di <https://huggingface.co/pyannote/speaker-diarization-3.1> dan <https://huggingface.co/pyannote/segmentation-3.0>.

## Konfigurasi

Semua pengaturan pipeline ada di [configs/default.yaml](configs/default.yaml).

## Hasil evaluasi

Belum tersedia.

## Lisensi

MIT
