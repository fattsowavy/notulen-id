# Hasil evaluasi

Dibuat oleh `scripts/evaluate.py` pada 2026-11-19 di NVIDIA GeForce RTX 4050 Laptop GPU.

- ASR: faster-whisper `medium` (int8_float16, cuda), bahasa `id`
- Diarization: `pyannote/speaker-diarization-3.1`
- LLM: `gemma3:4b` lewat ollama

## ASR

Teks dinormalisasi sebelum dinilai: huruf kecil, tanpa tanda baca, dan tanda hubung menjadi spasi. Angka tidak dieja, jadi "60" melawan "enam puluh" dihitung salah.

| Subset | Ucapan | Kata referensi | WER | CER | Substitusi | Hapus | Sisip | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLEURS (ucapan tunggal) | 100 | 1981 | 10.6% | 3.8% | 155 | 28 | 27 | 0.204 |

## Diarization

DER dengan collar 0.25 detik di tiap batas giliran, termasuk area tumpang tindih.

| Jumlah pembicara | File | DER | Salah pembicara | Terlewat | Alarm palsu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dideteksi otomatis | 6 | 13.4% | 10.1% | 2.4% | 0.9% |
| Diberikan (benar) | 6 | 7.6% | 4.3% | 2.4% | 0.9% |

## Ringkasan

LLM membaca transkrip referensi, jadi skor ini mengukur peringkasan saja tanpa kesalahan ASR. Teks yang dibandingkan adalah topik, keputusan, dan poin penting.

| Dokumen | ROUGE-L F1 | BERTScore F1 |
| --- | ---: | ---: |
| naskah_01 | 0.492 | 0.755 |
| naskah_02 | 0.452 | 0.781 |
| naskah_03 | 0.315 | 0.749 |
| naskah_04 | 0.650 | 0.825 |
| **Rata-rata** | **0.477** | **0.777** |

## Tindak lanjut

Item dianggap cocok jika token F1 tugas atau kutipannya minimal 0.5. Pencocokan satu lawan satu.

| Precision | Recall | F1 | Prediksi | Referensi | Cocok | Penanggung jawab benar |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 58.3% | 77.8% | 66.7% | 12 | 9 | 7 | 85.7% |

## Catatan metodologi

- FLEURS dan LibriVox berisi ucapan yang dibacakan, bukan percakapan spontan. Rapat sungguhan biasanya lebih sulit: ada jeda ragu, kata gaul, dan bising.
- Rapat simulasi tidak memiliki ucapan yang tumpang tindih, sehingga DER di sini cenderung lebih rendah daripada rapat sungguhan.
- Rapat simulasi LibriVox hanya dipakai untuk DER. Transkripnya memakai ejaan lama (misalnya "itoe" dan "djadi"), sehingga WER-nya akan menghitung perbedaan ejaan sebagai kesalahan.
- Ringkasan referensi ditulis manual untuk naskah rapat buatan, jadi jumlah dokumennya kecil dan angka ringkasan hanya indikatif.
