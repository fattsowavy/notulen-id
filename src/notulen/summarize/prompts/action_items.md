Kamu adalah notulis rapat yang teliti. Berikut potongan transkrip rapat atau podcast berbahasa Indonesia. Setiap baris berisi waktu, label pembicara, dan ucapannya. Transkrip dibuat oleh mesin, jadi mungkin ada kata yang salah dengar.

Temukan semua tindak lanjut, yaitu tugas konkret yang akan dikerjakan seseorang setelah rapat. Untuk setiap tindak lanjut, isi:

- "tugas": apa yang harus dikerjakan, ditulis sebagai kalimat perintah singkat dalam bahasa Indonesia.
- "penanggung_jawab": siapa yang mengerjakan. Pakai nama jika disebut jelas, atau label pembicara seperti "Pembicara 2". Isi null jika tidak jelas.
- "tenggat": batas waktu persis seperti yang diucapkan, misalnya "Jumat sore". Isi null jika tidak disebut.
- "kutipan_sumber": salinan kata demi kata dari ucapan di transkrip yang menyebut tugas itu, tanpa waktu dan label pembicara. Jangan diparafrasekan, diringkas, atau diterjemahkan.

Aturan:
- Masukkan hanya tugas yang benar-benar diucapkan di transkrip. Tindak lanjut tanpa kutipan yang cocok dengan transkrip akan dibuang.
- Tulis setiap tugas satu kali saja. Permintaan tugas dan jawaban kesediaan dari orang yang ditugasi dihitung sebagai satu tugas.
- Jadwal rapat berikutnya bukan tindak lanjut. Catat hanya jika ada orang yang ditugasi menyiapkannya.
- Jangan menambah kata pada tenggat. Jika transkrip menyebut "hari Jumat", tulis "hari Jumat", bukan "Jumat sore".
- Kembalikan daftar kosong jika tidak ada tindak lanjut.

Jawab hanya dengan JSON dengan format:
{"tindak_lanjut": [{"tugas": "...", "penanggung_jawab": "...", "tenggat": "...", "kutipan_sumber": "..."}]}

Transkrip:

{transcript}
