# Data

## Struktur

```text
data/
├── raw/            # unduhan dataset mentah (gitignored)
├── samples/        # 2-3 audio contoh pendek untuk demo
└── test_set/
    ├── audio/        # audio uji 16 kHz mono (gitignored, dibuat ulang oleh skrip)
    ├── transcripts/  # transkrip referensi .txt
    ├── rttm/         # giliran bicara referensi .rttm
    └── summaries/    # ringkasan dan tindak lanjut referensi .json
```

Audio dan unduhan mentah tidak di-commit. Bangun ulang dengan:

```powershell
python scripts/build_test_set.py
```

Skrip ini mengunduh sekitar 460 MB ke `data/raw/`. Pengambilan sampelnya deterministik (`evaluation.build.seed` di `configs/default.yaml`), jadi hasilnya selalu sama.

## Sumber dan lisensi

| Prefix | Sumber | Lisensi | Dipakai untuk |
| --- | --- | --- | --- |
| `fleurs_` | [FLEURS](https://huggingface.co/datasets/google/fleurs), bahasa Indonesia, split test | CC-BY 4.0 | WER dan CER |
| `rapat_` | [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia), split test | Public domain (CC-0) | DER saja |
| `naskah_` | Ditulis manual untuk proyek ini | MIT, sama dengan proyek | Ringkasan dan F1 tindak lanjut |

Transkrip FLEURS adalah karya Google yang dirilis dengan CC-BY 4.0: Conneau dkk., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022.

### Mengapa dua dataset audio

FLEURS tidak mencatat identitas pembicara, hanya gender, sehingga tidak bisa dipakai untuk membuat referensi DER yang pasti benar. LibriVox Indonesia mencatat ID pembaca, jadi ucapan dari beberapa pembaca bisa disambung menjadi rapat simulasi dengan label pembicara yang tepat.

### Cara rapat simulasi dibuat

1. Pilih 2-3 pembaca. Setiap pembaca bicara sekali dulu, lalu giliran bergantian secara acak tanpa pembicara yang sama dua kali berturut-turut.
2. Tiap ucapan dipotong dari awal sampai akhir suara dengan Silero VAD, lalu disambung dengan jeda hening acak 0,2-1,0 detik.
3. RTTM referensi berasal dari wilayah ucapan VAD di tiap klip, berlabel ID pembaca.

Keterbatasan: tidak ada ucapan yang tumpang tindih, dan semuanya ucapan yang dibacakan, bukan percakapan spontan.

Transkrip LibriVox berasal dari buku-buku lama dan banyak yang memakai ejaan sebelum EYD, misalnya "itoe", "djadi", dan "pentjoerian". Whisper menulis dengan ejaan modern, jadi rapat simulasi tidak dipakai untuk WER (`evaluation.wer_exclude_subsets`). Transkripnya tetap disimpan sebagai rujukan isi ucapan.

## Format anotasi

Semua file satu item memakai nama dasar yang sama, misalnya `rapat_03.wav`, `rapat_03.txt`, dan `rapat_03.rttm`.

**Transkrip (`.txt`)**: satu ucapan per baris. Label pembicara opsional, dengan format `Pembicara N: `. Label ini diabaikan saat menghitung WER.

```text
Pembicara 1: Selamat pagi, kita mulai rapatnya.
Pembicara 2: Baik, Pak.
```

**RTTM (`.rttm`)**: format standar NIST, satu baris per giliran bicara.

```text
SPEAKER rapat_03 1 0.000 4.120 <NA> <NA> reader_2814 <NA> <NA>
```

**Ringkasan (`.json`)**: skema `Summary` di `src/notulen/schemas.py`. Setiap `kutipan_sumber` harus disalin persis dari transkrip, dan ini dicek oleh `tests/test_evaluation.py`. Nilai `timestamp` di referensi tidak dinilai. Tulis `penanggung_jawab` sebagai nama beserta label pembicaranya, misalnya `Bu Sari (Pembicara 2)`, karena LLM bisa menjawab dengan salah satunya dan keduanya benar.

## Menambah rekaman sendiri

Taruh `audio/<id>.wav`, lalu tambahkan referensi yang tersedia (`transcripts/<id>.txt`, `rttm/<id>.rttm`, `summaries/<id>.json`). `scripts/evaluate.py` memakai setiap item sesuai file yang ada. Pilih prefix baru, misalnya `asli_01`, agar hasilnya tampil sebagai subset tersendiri dan tidak terhapus saat test set dibangun ulang.

Pastikan semua orang dalam rekaman setuju rekamannya dipakai, terutama jika file akan di-commit.
