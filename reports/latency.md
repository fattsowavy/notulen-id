# Perbandingan konfigurasi ASR

Dibuat oleh `scripts/benchmark_latency.py` pada 2026-11-19 dengan 30 ucapan FLEURS (6.2 menit audio).

RTF adalah waktu transkripsi dibagi durasi audio; di bawah 1 berarti lebih cepat dari waktu nyata. VRAM adalah tambahan memori GPU di atas kondisi sebelum model dimuat.

| Model | compute_type | Device | Muat model (dtk) | RTF | WER | CER | VRAM puncak (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | int8_float16 | cuda | 4.2 | 0.114 | 15.2% | 6.1% | 514 |
| medium | int8_float16 | cuda | 7.9 | 0.206 | 11.6% | 5.1% | 1128 |
