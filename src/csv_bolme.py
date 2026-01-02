import pandas as pd
import os
import glob


def split_csv_file(input_csv, output_dir, parts=10):
    """
    Tek bir CSV dosyasını 'parts' kadar parçaya böler (RAM dostu).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[+] İşleniyor: {input_csv}")

    # Toplam satır sayısını bul
    total_rows = sum(1 for _ in open(input_csv, 'r')) - 1  # header hariç
    rows_per_chunk = max(1, total_rows // parts)

    print(f"[+] Toplam satır: {total_rows}")
    print(f"[+] Her parçaya ~{rows_per_chunk} satır yazılacak")

    # Chunk halinde oku
    reader = pd.read_csv(input_csv, chunksize=rows_per_chunk)

    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    for i, chunk in enumerate(reader, start=1):
        out_file = os.path.join(output_dir, f"{base_name}_part{i}.csv")
        print(f"    -> Parça {i}: {out_file}")
        chunk.to_csv(out_file, index=False)

    print("[✓] Dosya başarıyla bölündü.\n")


def split_all_csv_in_folder(folder_path, output_root, parts=10):
    """
    Klasördeki tüm CSV dosyalarını 'parts' kadar parçaya böler.
    (Bu fonksiyonu şu an kullanmıyoruz ama dursun istersen.)
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print("[!] Klasörde CSV bulunamadı:", folder_path)
        return

    print(f"[+] Toplam {len(csv_files)} CSV bulundu.")

    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        out_dir = os.path.join(output_root, f"{file_name}_split")
        split_csv_file(csv_file, out_dir, parts=parts)

    print("\n===== TÜM CSV DOSYALARI BAŞARIYLA BÖLÜNDÜ =====")


# ============================================================
#                   KULLANIM ÖRNEĞİ
# ============================================================

if __name__ == "__main__":
    # Tek bir dosyayı 10'a bölmek için:
    INPUT_CSV = r"C:\Users\ASUS\Desktop\AĞANOMALİ_2\normal_traffic_dataset\all_packets.csv"
    OUTPUT_FOLDER = r"C:\Users\ASUS\Desktop\AĞANOMALİ_2\normal_traffic_dataset\all_packets_split"

    split_csv_file(
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_FOLDER,
        parts=10
    )
