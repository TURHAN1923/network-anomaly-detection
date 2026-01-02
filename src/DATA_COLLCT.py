# AMACIMIZ WİRESHARK/SCAPY KULLANARAK AĞ TRAFİĞİNDEN VERİ TOPLAMA İŞLEMİ.

import pyshark
from scapy.all import sniff, IP, TCP, UDP, ICMP, IFACES
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import json
from datetime import datetime
import logging
import sys

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkTrafficCollector:
    def __init__(self, interface='eth0', window_size=100):
        self.interface = interface
        self.window_size = window_size
        self.packets = []
        self.features_list = []

    def extract_features_from_window(self, packets):
        """Paket penceresinden özellik çıkar"""
        if len(packets) == 0:
            return None

        features = {}

        # Temel istatistikler
        features['total_packets'] = len(packets)

        packet_sizes = [len(pkt) for pkt in packets if hasattr(pkt, '__len__')]
        features['total_bytes'] = sum(packet_sizes)
        features['avg_packet_size'] = np.mean(packet_sizes) if packet_sizes else 0
        features['std_packet_size'] = np.std(packet_sizes) if packet_sizes else 0

        # Protokol dağılımı
        tcp_count = sum(1 for pkt in packets if TCP in pkt)
        udp_count = sum(1 for pkt in packets if UDP in pkt)
        icmp_count = sum(1 for pkt in packets if ICMP in pkt)

        total = len(packets)
        features['tcp_ratio'] = tcp_count / total if total > 0 else 0
        features['udp_ratio'] = udp_count / total if total > 0 else 0
        features['icmp_ratio'] = icmp_count / total if total > 0 else 0

        # Port analizi
        dst_ports = []
        syn_count = 0
        fin_count = 0
        rst_count = 0

        for pkt in packets:
            if TCP in pkt:
                if hasattr(pkt[TCP], 'dport'):
                    dst_ports.append(pkt[TCP].dport)
                if pkt[TCP].flags & 0x02:  # SYN
                    syn_count += 1
                if pkt[TCP].flags & 0x01:  # FIN
                    fin_count += 1
                if pkt[TCP].flags & 0x04:  # RST
                    rst_count += 1
            elif UDP in pkt:
                if hasattr(pkt[UDP], 'dport'):
                    dst_ports.append(pkt[UDP].dport)

        features['unique_dst_ports'] = len(set(dst_ports))
        features['port_entropy'] = self._calculate_entropy(dst_ports)

        # Zaman analizi
        timestamps = [float(pkt.time) for pkt in packets if hasattr(pkt, 'time')]
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            features['avg_packet_interval'] = np.mean(intervals)
            features['std_packet_interval'] = np.std(intervals)
            duration = timestamps[-1] - timestamps[0]
            features['packets_per_second'] = len(packets) / duration if duration > 0 else 0
        else:
            features['avg_packet_interval'] = 0
            features['std_packet_interval'] = 0
            features['packets_per_second'] = 0

        # TCP flags
        features['syn_count'] = syn_count
        features['fin_count'] = fin_count
        features['rst_count'] = rst_count
        features['syn_fin_ratio'] = syn_count / (fin_count + 1)

        # IP çeşitliliği
        src_ips = []
        dst_ips = []

        for pkt in packets:
            if IP in pkt:
                src_ips.append(pkt[IP].src)
                dst_ips.append(pkt[IP].dst)

        features['unique_src_ips'] = len(set(src_ips))
        features['unique_dst_ips'] = len(set(dst_ips))
        features['ip_diversity'] = len(set(src_ips)) / (len(packets) + 1)

        return features

    def _calculate_entropy(self, data):
        """Shannon entropi hesapla"""
        if not data:
            return 0

        value_counts = defaultdict(int)
        for item in data:
            value_counts[item] += 1

        entropy = 0
        total = len(data)
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def packet_callback(self, packet):
        self.packets.append(packet)

        if len(self.packets) >= self.window_size:
            features = self.extract_features_from_window(self.packets)
            if features:
                features['timestamp'] = datetime.now().isoformat()
                self.features_list.append(features)
                # Konsolu çok doldurmamak için her 10 örnekte bir yazdıralım
                if len(self.features_list) % 10 == 0:
                    logger.info(f"Özellikler çıkarıldı. Toplam: {len(self.features_list)}")

            # Pencereyi temizle (50% overlap için yarısını tut)
            self.packets = self.packets[self.window_size // 2:]

    def start_capture(self, duration=300, save_file='collected_data.csv'):
        logger.info(f"Trafik dinleme başladı: {self.interface}")
        logger.info(f"Süre: {duration} saniye")
        logger.info(f"Pencere boyutu: {self.window_size} paket")

        try:
            # Scapy ile paket yakalama
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                timeout=duration,
                store=False
            )

            # Verileri kaydet
            if self.features_list:
                df = pd.DataFrame(self.features_list)
                df.to_csv(save_file, index=False)
                logger.info(f"✓ {len(self.features_list)} örnek kaydedildi: {save_file}")
                self.print_statistics(df)
            else:
                logger.warning("Hiç veri toplanamadı! Arayüzü yanlış seçmiş olabilirsiniz.")

        except Exception as e:
            logger.error(f"Kritik Hata: {e}")
            logger.info("Lütfen seçtiğiniz arayüz adının (GUID) tam ve doğru olduğundan emin olun.")

    def print_statistics(self, df):
        print("\n" + "=" * 60)
        print("VERİ TOPLAMA İSTATİSTİKLERİ")
        print("=" * 60)
        print(f"Toplam örnek sayısı: {len(df)}")
        print(f"\nÖzellik değer aralıkları:")
        print(df.describe())


def list_interfaces_and_select():
    print("\n" + "=" * 100)
    print(f"{'NO':<3} | {'AÇIKLAMA (İsim)':<30} | {'IP ADRESİ':<15} | {'GUID (KOPYALANACAK KISIM)'}")
    print("=" * 100)

    interfaces = []

    try:
        # Scapy IFACES verisini al
        for i, iface_name in enumerate(IFACES.data, 1):
            iface = IFACES.data[iface_name]
            ip = iface.ip if iface.ip else "IP Yok"
            friendly_name = str(iface.name)[:30]  # İsmi kısalt

            # Listeye ekle
            interfaces.append((iface_name, friendly_name, ip))

            print(f"{i:<3} | {friendly_name:<30} | {ip:<15} | {iface_name}")
            print("-" * 100)

        print("\nİPUCU: Genellikle '192.168.x.x' IP adresine sahip olanı seçmelisiniz.")

        selection = input("\nLütfen listedeki NUMARAYI girin (veya GUID'i elle yapıştırın): ").strip()

        # Eğer kullanıcı numara girdiyse
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(interfaces):
                selected_guid = interfaces[idx][0]
                print(f"\nSeçilen Arayüz: {interfaces[idx][1]} ({selected_guid})")
                return selected_guid
            else:
                print("Geçersiz numara!")
                return None
        # Kullanıcı GUID yapıştırdıysa
        else:
            return selection

    except Exception as e:
        print(f"Arayüzler listelenirken hata oluştu: {e}")
        return input("Lütfen arayüz adını manuel girin: ")


if __name__ == "__main__":
    print("DATA COLLECTER MODELİNE HOŞGELDİNİZ.")

    # Arayüz seçimi yaptır
    selected_interface = list_interfaces_and_select()

    if selected_interface:
        # Veri toplayıcı oluştur
        collector = NetworkTrafficCollector(
            interface=selected_interface,
            window_size=100
        )

        # Normal trafik toplama
        print("\n1. NORMAL TRAFİK TOPLAMA")
        print("Normal internet kullanımı yapın (web gezin, video izleyin)")
        try:
            input("Hazır olduğunuzda ENTER'a basın...")
        except SyntaxError:
            pass

        collector.start_capture(
            duration=9000,  # kaç sn süreceği
            save_file='normal_traffic3.csv'
        )

        print("\n✓ İşlem tamamlandı.")
        print("Saldırı verisi toplamak için programı yeniden başlatın.")
    else:
        print("Arayüz seçilmedi, program kapatılıyor.")