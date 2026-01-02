"""
PCAP Dosyalarını CSV'ye Dönüştürme
Ağ trafiği paketlerinden özellik çıkarma - TORCH/CUDA DESTEKLİ VERSİYON
"""

import pyshark
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
import os

import torch  # <-- Torch eklendi

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Torch / CUDA cihaz seçimi
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"PyTorch CUDA kullanılacak: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.info("PyTorch CPU modunda çalışacak (CUDA bulunamadı).")


class PcapToFeatures:
    """PCAP dosyasından özellik çıkar"""

    def __init__(self, window_size=100):
        """
        Args:
            window_size: Kaç paket gruplanacak
        """
        self.window_size = window_size

        self.feature_names = [
            'total_packets', 'total_bytes', 'avg_packet_size', 'std_packet_size',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'unique_dst_ports', 'port_entropy',
            'avg_packet_interval', 'std_packet_interval', 'packets_per_second',
            'syn_count', 'fin_count', 'rst_count', 'syn_fin_ratio',
            'unique_src_ips', 'unique_dst_ips', 'ip_diversity'
        ]

    def process_pcap_file(self, pcap_path, is_attack=True):
        """
        PCAP dosyasını işle ve özellik çıkar
        """
        logger.info(f"İşleniyor: {pcap_path}")

        try:
            # PCAP'i oku
            packets = rdpcap(str(pcap_path))

            if len(packets) == 0:
                logger.warning(f"Dosya boş: {pcap_path}")
                return None

            # Pencerelere böl ve özellik çıkar
            features_list = []

            # TQDM ile ilerleme çubuğu
            for i in range(0, len(packets), self.window_size // 2):
                window = packets[i:i + self.window_size]

                if len(window) < 10:  # Çok az paket varsa atla
                    continue

                features = self.extract_features_from_window(window)
                if features is not None:
                    features_list.append(features)

            if not features_list:
                return None

            # DataFrame oluştur
            df = pd.DataFrame(features_list, columns=self.feature_names)
            df['label'] = 1 if is_attack else 0

            return df

        except Exception as e:
            logger.error(f"PCAP işleme hatası ({pcap_path}): {e}")
            return None

    def extract_features_from_window(self, packets):
        """Paket penceresinden özellik çıkar"""
        if len(packets) == 0:
            return None

        try:
            features = {}

            # Temel istatistikler
            features['total_packets'] = len(packets)

            packet_sizes = []
            tcp_count = 0
            udp_count = 0
            icmp_count = 0
            dst_ports = []
            src_ips = []
            dst_ips = []
            syn_count = 0
            fin_count = 0
            rst_count = 0
            timestamps = []

            for pkt in packets:
                # Paket boyutu
                packet_sizes.append(len(pkt))

                # Zaman damgası
                if hasattr(pkt, 'time'):
                    timestamps.append(float(pkt.time))

                # IP katmanı
                if IP in pkt:
                    src_ips.append(pkt[IP].src)
                    dst_ips.append(pkt[IP].dst)

                # Protokol sayımı
                if TCP in pkt:
                    tcp_count += 1
                    if hasattr(pkt[TCP], 'dport'):
                        dst_ports.append(pkt[TCP].dport)

                    # TCP bayrakları
                    flags = pkt[TCP].flags
                    if flags & 0x02:  # SYN
                        syn_count += 1
                    if flags & 0x01:  # FIN
                        fin_count += 1
                    if flags & 0x04:  # RST
                        rst_count += 1

                elif UDP in pkt:
                    udp_count += 1
                    if hasattr(pkt[UDP], 'dport'):
                        dst_ports.append(pkt[UDP].dport)

                elif ICMP in pkt:
                    icmp_count += 1

            # Torch tensörlerine çevir (CUDA varsa GPU'ya gider)
            packet_sizes_t = torch.tensor(packet_sizes, dtype=torch.float32, device=device)

            # Paket boyutu istatistikleri (Torch ile)
            total_bytes_t = packet_sizes_t.sum()
            mean_size_t = packet_sizes_t.mean()
            # std'de unbiased=False (numpy std ile daha uyumlu)
            std_size_t = packet_sizes_t.std(unbiased=False)

            features['total_bytes'] = float(total_bytes_t.item())
            features['avg_packet_size'] = float(mean_size_t.item())
            features['std_packet_size'] = float(std_size_t.item())

            # Protokol oranları
            total = len(packets)
            features['tcp_ratio'] = tcp_count / total if total > 0 else 0.0
            features['udp_ratio'] = udp_count / total if total > 0 else 0.0
            features['icmp_ratio'] = icmp_count / total if total > 0 else 0.0

            # Port analizi (CPU tarafında yeterince hızlı)
            features['unique_dst_ports'] = len(set(dst_ports))
            features['port_entropy'] = self._calculate_entropy(dst_ports)

            # Zaman analizi (Torch ile)
            if len(timestamps) > 1:
                ts_t = torch.tensor(timestamps, dtype=torch.float32, device=device)
                intervals_t = ts_t[1:] - ts_t[:-1]

                avg_interval_t = intervals_t.mean()
                std_interval_t = intervals_t.std(unbiased=False)

                features['avg_packet_interval'] = float(avg_interval_t.item())
                features['std_packet_interval'] = float(std_interval_t.item())

                duration = float((ts_t[-1] - ts_t[0]).item())
                features['packets_per_second'] = len(packets) / duration if duration > 0 else 0.0
            else:
                features['avg_packet_interval'] = 0.0
                features['std_packet_interval'] = 0.0
                features['packets_per_second'] = 0.0

            # TCP bayrakları
            features['syn_count'] = syn_count
            features['fin_count'] = fin_count
            features['rst_count'] = rst_count
            features['syn_fin_ratio'] = syn_count / (fin_count + 1)

            # IP çeşitliliği (CPU)
            unique_src = len(set(src_ips))
            unique_dst = len(set(dst_ips))
            features['unique_src_ips'] = unique_src
            features['unique_dst_ips'] = unique_dst
            features['ip_diversity'] = unique_src / (len(packets) + 1)

            # Feature isim sırasına göre liste döndür
            return [features[name] for name in self.feature_names]

        except Exception as e:
            logger.error(f"Feature çıkarma hatası: {e}")
            return None

    def _calculate_entropy(self, data):
        """Shannon entropisi hesapla"""
        if not data:
            return 0.0

        value_counts = defaultdict(int)
        for item in data:
            value_counts[item] += 1

        entropy = 0.0
        total = len(data)
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)


def clean_path_input(path_str):
    """Kullanıcı girdisindeki tırnakları ve boşlukları temizler"""
    if path_str:
        return path_str.strip().strip('"').strip("'")
    return ""


def process_multiple_pcaps(pcap_dir, output_csv, is_attack=True):
    """
    Birden fazla PCAP dosyasını işle ve tek CSV'de birleştir
    """
    # Gelen yoldaki tırnak işaretlerini temizle
    pcap_dir_str = clean_path_input(pcap_dir)
    pcap_dir = Path(pcap_dir_str)

    output_csv = clean_path_input(output_csv)

    if not pcap_dir.exists():
        logger.error(f"Dizin bulunamadı: {pcap_dir}")
        return

    logger.info("Klasörler taranıyor...")

    # Dosyaları bul (Recursive - Alt klasörler dahil)
    extensions = ['*.pcap', '*.pcapng', '*.cap', '*.PCAP', '*.PCAPNG']
    pcap_files = []

    for ext in extensions:
        pcap_files.extend(pcap_dir.rglob(ext))

    # Tekrar edenleri temizle (bazen glob çakışabilir)
    pcap_files = list(set(pcap_files))

    if not pcap_files:
        logger.error(f"HATA: '{pcap_dir}' konumunda veya alt klasörlerinde hiç PCAP dosyası bulunamadı!")
        try:
            # Kullanıcıya yardımcı olmak için klasör içeriğini göster
            contents = [x.name for x in pcap_dir.iterdir()]
            logger.info(f"Bu klasörde görünen dosyalar: {contents[:10]} ...")
        except Exception:
            pass
        return

    logger.info(f"Toplam {len(pcap_files)} adet PCAP dosyası bulundu.")

    processor = PcapToFeatures(window_size=100)
    all_dfs = []

    # Dosyaları tek tek işle
    for pcap_file in tqdm(pcap_files, desc="Dosyalar İşleniyor"):
        df = processor.process_pcap_file(pcap_file, is_attack=is_attack)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("Hiçbir dosyadan veri çıkarılamadı!")
        return

    logger.info("Veriler birleştiriliyor...")
    # Birleştir
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Temizlik
    df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
    df_combined = df_combined.fillna(0)
    df_combined = df_combined.clip(lower=0)

    # Çıktı klasörü yoksa oluştur
    try:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Klasör oluşturma hatası: {e}")

    # Kaydet
    df_combined.to_csv(output_csv, index=False)

    logger.info("\n" + "="*50)
    logger.info("✓ İŞLEM BAŞARIYLA TAMAMLANDI")
    logger.info(f"Toplam Örnek Sayısı: {len(df_combined)}")
    logger.info(f"Dosya Kaydedildi: {output_csv}")
    logger.info("="*50)

    return df_combined


def main():
    """Ana fonksiyon"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║    PCAP'TEN CSV'YE DÖNÜŞTÜRME (TORCH/CUDA DESTEKLİ)     ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    print("\nNE YAPMAK İSTERSİNİZ?")
    print("1. Tek PCAP dosyasını işle")
    print("2. Bir dizindeki tüm PCAP'leri işle (Alt klasörler dahil)")
    print("3. Normal + Saldırı PCAP'lerini ayrı klasörlerden al")

    choice = input("\nSeçiminiz (1-3): ").strip()

    if choice == "1":
        # Tek dosya
        pcap_path = clean_path_input(input("PCAP dosya yolu: "))
        output_csv = clean_path_input(input("Çıktı CSV dosyası: "))
        is_attack = input("Saldırı trafiği mi? (e/h): ").strip().lower() == 'e'

        processor = PcapToFeatures()
        df = processor.process_pcap_file(pcap_path, is_attack=is_attack)

        if df is not None:
            df.to_csv(output_csv, index=False)
            logger.info(f"✓ Kaydedildi: {output_csv}")

    elif choice == "2":
        # Dizin
        pcap_dir = input("PCAP dizini (Örn: C:\\Users\\Veriler): ")
        output_csv = input("Çıktı CSV dosyası (Örn: C:\\Users\\cikti.csv): ")
        is_attack = input("Saldırı trafiği mi? (e/h): ").strip().lower() == 'e'

        process_multiple_pcaps(pcap_dir, output_csv, is_attack=is_attack)

    elif choice == "3":
        # Ayrı ayrı
        print("\n--- NORMAL TRAFİK ---")
        normal_dir = input("Normal trafik PCAP dizini: ")

        print("\n--- SALDIRI TRAFİĞİ ---")
        attack_dir = input("Saldırı trafiği PCAP dizini: ")

        output_dir = input("Çıktıların kaydedileceği klasör: ")
        output_dir = clean_path_input(output_dir)

        # Normal trafiği işle
        if normal_dir:
            logger.info("\nNORMAL TRAFİK İŞLENİYOR...")
            process_multiple_pcaps(normal_dir, f'{output_dir}/normal_traffic.csv', is_attack=False)

        # Saldırı trafiğini işle
        if attack_dir:
            logger.info("\nSALDIRI TRAFİĞİ İŞLENİYOR...")
            process_multiple_pcaps(attack_dir, f'{output_dir}/attack_traffic.csv', is_attack=True)

        print("\n✓ İŞLEMLER TAMAMLANDI!")

    else:
        logger.error("Geçersiz seçim!")


if __name__ == "__main__":
    main()
