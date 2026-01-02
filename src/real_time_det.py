from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
import numpy as np
from collections import defaultdict, deque
import time
from datetime import datetime
import logging
import os
import sys
import json

# PyTorch tabanlı modeli import ediyoruz
from model_trainer import AdvancedAnomalyDetectorTorch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimeAnomalyDetector:

    def __init__(self, model, interface=None, window_size=100):
        self.model = model
        self.interface = interface
        self.window_size = window_size
        self.packet_buffer = deque(maxlen=window_size)

        self.stats = {
            'total_packets': 0,          # Program başından beri geçen TOPLAM paket
            'total_anomalies': 0,        # Program başından beri tespit edilen toplam anomali
            'start_time': time.time(),
            'last_alert_time': 0,
            # Son analiz edilen pencere (window) içindeki paket sayısı
            'last_window_packets': 0
        }

        # Eğitimde kullandığın 19 feature ile birebir aynı sıra olmalı
        self.feature_names = [
            'total_packets', 'total_bytes', 'avg_packet_size', 'std_packet_size',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'unique_dst_ports', 'port_entropy',
            'avg_packet_interval', 'std_packet_interval', 'packets_per_second',
            'syn_count', 'fin_count', 'rst_count', 'syn_fin_ratio',
            'unique_src_ips', 'unique_dst_ips', 'ip_diversity'
        ]

        # Son uyarıları tutmak için (ekran/log takibi için)
        self.alert_history = deque(maxlen=20)

    def extract_features(self, packets):
        """Paket penceresinden (window) feature çıkarır."""
        if len(packets) == 0:
            return None

        features = {}
        # BU "total_packets" sadece WINDOW içindeki paket sayısıdır
        features['total_packets'] = len(packets)

        # Paket boyutları
        packet_sizes = [len(pkt) for pkt in packets if hasattr(pkt, '__len__')]
        features['total_bytes'] = sum(packet_sizes)
        features['avg_packet_size'] = float(np.mean(packet_sizes)) if packet_sizes else 0.0
        features['std_packet_size'] = float(np.std(packet_sizes)) if packet_sizes else 0.0

        # Protokol dağılımı
        tcp_count = sum(1 for pkt in packets if TCP in pkt)
        udp_count = sum(1 for pkt in packets if UDP in pkt)
        icmp_count = sum(1 for pkt in packets if ICMP in pkt)
        total = len(packets)

        features['tcp_ratio'] = tcp_count / total if total > 0 else 0.0
        features['udp_ratio'] = udp_count / total if total > 0 else 0.0
        features['icmp_ratio'] = icmp_count / total if total > 0 else 0.0

        # Port & bayraklar
        dst_ports = []
        syn_count = 0
        fin_count = 0
        rst_count = 0

        for pkt in packets:
            if TCP in pkt:
                if hasattr(pkt[TCP], 'dport'):
                    dst_ports.append(pkt[TCP].dport)
                flags = pkt[TCP].flags
                if flags & 0x02:  # SYN
                    syn_count += 1
                if flags & 0x01:  # FIN
                    fin_count += 1
                if flags & 0x04:  # RST
                    rst_count += 1
            elif UDP in pkt:
                if hasattr(pkt[UDP], 'dport'):
                    dst_ports.append(pkt[UDP].dport)

        features['unique_dst_ports'] = len(set(dst_ports))
        features['port_entropy'] = self._calculate_entropy(dst_ports)

        # Zaman analizi
        timestamps = [float(pkt.time) for pkt in packets if hasattr(pkt, 'time')]
        if len(timestamps) > 1:
            timestamps = sorted(timestamps)
            intervals = np.diff(timestamps)
            features['avg_packet_interval'] = float(np.mean(intervals))
            features['std_packet_interval'] = float(np.std(intervals))
            duration = timestamps[-1] - timestamps[0]
            features['packets_per_second'] = len(packets) / duration if duration > 0 else 0.0
        else:
            features['avg_packet_interval'] = 0.0
            features['std_packet_interval'] = 0.0
            features['packets_per_second'] = 0.0

        # TCP bayrak sayıları
        features['syn_count'] = float(syn_count)
        features['fin_count'] = float(fin_count)
        features['rst_count'] = float(rst_count)
        features['syn_fin_ratio'] = syn_count / (fin_count + 1)  # 0'a bölme korumalı

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

        return np.array([features[name] for name in self.feature_names], dtype=np.float32)

    def _calculate_entropy(self, data):
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

    def packet_callback(self, packet):
        self.packet_buffer.append(packet)
        self.stats['total_packets'] += 1  # GLOBAL toplam paket sayısı

        # Her 10 pakette bir anomali analizi yap
        if self.stats['total_packets'] % 10 == 0:
            self.analyze_traffic()

    def analyze_traffic(self):
        try:
            # En az 50 paket olsun ki istatistik anlamlı olsun
            if len(self.packet_buffer) < 50:
                return

            packets_list = list(self.packet_buffer)
            features = self.extract_features(packets_list)
            if features is None:
                return

            # Son window içindeki paket sayısını istatistiklere yaz
            # (features[0] sırası feature_names'e göre 'total_packets' yani WINDOW uzunluğu)
            self.stats['last_window_packets'] = int(features[0])

            # Modelin beklediği şekle getir: (1, feature_sayısı)
            features_reshaped = features.reshape(1, -1)

            # PyTorch modelinden tahmin al
            # AdvancedAnomalyDetectorTorch.detect_anomaly → (bool, float)
            is_anomaly, score = self.model.detect_anomaly(features_reshaped)

            if is_anomaly:
                self.handle_anomaly(features, score)

            # Durum çıktısını belirli aralıklarla göster
            if self.stats['total_packets'] % 100 == 0:
                self.print_status()

        except Exception as e:
            logger.error(f"Analiz hatası: {e}")


    def handle_anomaly(self, features, score):
        """Anomali tespit edildiğinde çağrılır."""
        self.stats['total_anomalies'] += 1
        current_time = time.time()

        # Uyarı spam olmasın diye (0.5 sn throttle)
        if current_time - self.stats['last_alert_time'] < 0.5:
            return

        self.stats['last_alert_time'] = current_time

        # Alert objesi (sadece VAR/YOK mantığı)
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': float(score),
            'features': {name: float(val) for name, val in zip(self.feature_names, features)}
        }

        self.alert_history.append(alert)
        self.print_alert(alert)
        self.log_alert(alert)

    def print_alert(self, alert):
        """Ekrana alert basar."""
        print(f"\n{'='*70}\n   ANOMALİ TESPİT EDİLDİ! \n{'='*70}")
        print(f" Zaman : {alert['timestamp']}")
        print(f" Skor  : {alert['score']:.6f}")
        print(f" PPS   : {alert['features']['packets_per_second']:.2f}")
        print(f" SYN   : {alert['features']['syn_count']}")
        print(f" Toplam Paket (window): {alert['features']['total_packets']}")
        print(f"{'='*70}\n")

    def log_alert(self, alert):
        """Anomali loglarını JSON formatında dosyaya yazar."""
        def convert(o):
            # Numpy tiplerini normal Python tiplerine çevir
            import numpy as np
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        clean_alert = {
            k: ( {fk: convert(fv) for fk, fv in v.items()} if isinstance(v, dict) else convert(v)
            ) for k, v in alert.items()
        }

        with open('anomaly_alerts.log', 'a') as f:
            f.write(json.dumps(clean_alert) + '\n')

    def print_status(self):

        uptime = time.time() - self.stats['start_time']
        avg_pps = self.stats['total_packets'] / uptime if uptime > 0 else 0.0

        window_packets = self.stats.get('last_window_packets', len(self.packet_buffer))

        print(
            f"\rPaket (window): {window_packets} | "
            f"Toplam: {self.stats['total_packets']} | "
            f"Anomali: {self.stats['total_anomalies']} | "
            f"Ortalama PPS: {avg_pps:.1f} | "
            f"Buffer: {len(self.packet_buffer)}",
            end='',
            flush=True
        )

    def start_monitoring(self):
        logger.info("İzleme başlatılıyor...")
        logger.info(f"Interface: {self.interface if self.interface else 'Varsayılan'}")
        logger.info(f"Threshold: {self.model.threshold}")
        logger.info(f"Window Size: {self.window_size}")
        print("\n Paket yakalanıyor, durdurmak için Ctrl + C...\n")

        try:
            if self.interface:
                sniff(iface=self.interface, prn=self.packet_callback, store=False, filter="ip")
            else:
                sniff(prn=self.packet_callback, store=False, filter="ip")
        except KeyboardInterrupt:
            self.print_final_stats()
        except Exception as e:
            logger.error(f"Sniff hatası: {e}")
            logger.error("Lütfen programı yönetici yetkisiyle çalıştırmayı deneyin.")

    def print_final_stats(self):
        print("\n\n" + "="*70)
        print(" İZLEME DURDURULDU - ÖZET")
        print("="*70)
        print(f"Toplam Paket : {self.stats['total_packets']}")
        print(f"Toplam Anomali: {self.stats['total_anomalies']}")
        uptime = time.time() - self.stats['start_time']
        print(f"Çalışma Süresi: {uptime:.1f} sn")
        print(f"Ortalama PPS : {self.stats['total_packets']/uptime:.1f}")
        print("="*70)


def get_windows_interfaces():
    """Windows için ağ arayüzlerini detaylı listele."""
    try:
        from scapy.all import IFACES
        interfaces = []

        priority_keywords = ['wi-fi', 'wireless', 'ethernet', 'local area']

        for iface_name in IFACES.keys():
            iface = IFACES[iface_name]

            if hasattr(iface, 'ip') and iface.ip and iface.ip != '0.0.0.0':
                description = iface.description if hasattr(iface, 'description') else iface_name

                is_virtual = any(x in description.lower() for x in
                                 ['hyper-v', 'virtualbox', 'vmware', 'virtual', 'loopback'])
                is_priority = any(x in description.lower() for x in priority_keywords)

                interfaces.append({
                    'name': iface_name,
                    'description': description,
                    'ip': iface.ip,
                    'is_virtual': is_virtual,
                    'is_priority': is_priority
                })

        interfaces.sort(key=lambda x: (not x['is_priority'], x['is_virtual'], x['description']))
        return interfaces
    except Exception as e:
        logger.warning(f"Interface listesi alınamadı: {e}")
        return [{'name': iface, 'description': iface, 'ip': 'N/A',
                 'is_virtual': False, 'is_priority': False} for iface in get_if_list()]


def main():
    print("TESPİT BASLIYOR.....")

    # Admin kontrolü (Windows)
    if sys.platform == 'win32':
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                logger.warning(" Program yönetici yetkisi ile çalışmıyor!")
                logger.warning("   'Yönetici olarak çalıştır' deneyin.\n")
        except:
            pass

    # Model dosyası kontrolü (PyTorch versiyonu)
    model_folder = r"C:\Users\ASUS\Desktop\AĞANOMALİ_2\model_torch_cpu"
    model_path = os.path.join(model_folder, "lstm_autoencoder_torch.pt")

    if not os.path.exists(model_path):
        logger.error(f"Model dosyası bulunamadı: {model_path}")
        logger.info("Önce 'python model_trainer.py' ile modeli eğitip kaydedin.")
        sys.exit(1)

    # Modeli yükle
    logger.info("Model yükleniyor...")
    detector_model = AdvancedAnomalyDetectorTorch(sequence_length=10, use_pca=True)

    try:
        detector_model.load_model(folder=model_folder)
        logger.info("✓ Model başarıyla yüklendi.\n")
    except Exception as e:
        logger.error(f"Model yüklenemedi: {e}")
        sys.exit(1)

    # Arayüz seçimi
    print(" AĞ ARAYÜZÜ SEÇİMİ")
    print("=" * 80)

    interfaces = get_windows_interfaces()
    if not interfaces:
        logger.error("Hiçbir ağ arayüzü bulunamadı!")
        logger.error("Npcap / WinPcap kurulu mu, kontrol edin.")
        sys.exit(1)

    for i, iface in enumerate(interfaces, 1):
        marker = "🌐" if iface['is_priority'] else ("🔧" if iface['is_virtual'] else "📶")
        virtual_tag = " [SANAL]" if iface['is_virtual'] else ""
        priority_tag = " [ÖNERİLEN]" if iface['is_priority'] else ""

        print(f"{i}. {marker} {iface['description']}{virtual_tag}{priority_tag}")
        print(f"   IP: {iface['ip']}\n")

    print("=" * 80)
    print(" Fiziksel Wi-Fi / Ethernet adaptörünü seçmen en sağlıklısıdır.")
    print("=" * 80)

    try:
        choice = int(input("\n✓ Seçiminiz (Numara): "))
        if 1 <= choice <= len(interfaces):
            interface = interfaces[choice - 1]['name']
            print(f"\n✓ Seçilen: {interfaces[choice - 1]['description']}")
        else:
            logger.error("Geçersiz seçim!")
            sys.exit(1)
    except ValueError:
        logger.error("Geçersiz giriş!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nİşlem iptal edildi.")
        sys.exit(0)

    print("\n" + "=" * 80)
    print("İZLEME BAŞLIYOR... (Durdurmak için Ctrl + C)")
    print("=" * 80 + "\n")

    rt_detector = RealTimeAnomalyDetector(model=detector_model, interface=interface, window_size=100)
    rt_detector.start_monitoring()


if __name__ == "__main__":
    main()
