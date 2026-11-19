---
title: Notulen Otomatis
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Transkrip per pembicara dan notulen rapat berbahasa Indonesia
---

# Notulen Otomatis

Unggah rekaman rapat atau podcast berbahasa Indonesia (maksimal 2 menit di demo ini). Hasilnya:

- transkrip per pembicara dengan penanda waktu,
- ringkasan berisi topik, keputusan, dan poin penting,
- daftar tindak lanjut beserta kutipan sumber dari transkrip.

## Cara kerja

```text
audio -> ffmpeg + Silero VAD -> faster-whisper -> pyannote.audio
      -> penggabungan kata dan pembicara -> LLM (map-reduce) -> notulen
```

Demo ini berjalan di ZeroGPU: GPU dipinjam hanya selama transkripsi dan pengenalan pembicara, lalu ringkasan dibuat lewat Hugging Face Inference Providers. Setiap pengunjung punya kuota GPU harian dari Hugging Face (lebih besar jika login), jadi demo bisa menolak permintaan setelah beberapa kali dicoba.

## Memilih LLM

Buka panel **Pengaturan LLM** untuk memakai penyedia lain (Groq, Google Gemini, OpenRouter, OpenAI, atau endpoint OpenAI-compatible milik Anda) dengan API key Anda sendiri. Key hanya dipakai untuk permintaan itu, tidak disimpan, dan tidak ditulis ke log.

## Privasi

Audio diproses di server Space ini. Transkripnya dikirim ke penyedia LLM yang dipilih (secara default Hugging Face Inference Providers) untuk diringkas. Jangan mengunggah rekaman yang berisi informasi rahasia.
