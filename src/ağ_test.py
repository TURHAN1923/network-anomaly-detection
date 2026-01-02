"""
Ağ Adaptörü Test ve Tespit Aracı
Ethernet bağlantınızı bulur ve test eder
"""

from scapy.all import *
import time
import sys


def get_active_ethernet_interface():
    """Aktif Ethernet adaptörünü bul"""
    print("=" * 70)
    print("AĞ ADAPTÖRÜ ANALİZİ")
    print("=" * 70)

    try:
        from scapy.all import IFACES
        ethernet_interfaces = []

        for iface_name in IFACES.keys():
            iface = IFACES[iface_name]

            if hasattr(iface, 'ip') and iface.ip and iface.ip != '0.0.0.0':
                description = iface.description if hasattr(iface, 'description') else iface_name

                # Ethernet anahtar kelimeleri
                is_ethernet = any(x in description.lower() for x in
                                  ['ethernet', 'realtek', 'intel', 'broadcom', 'local area', 'lan'])

                # Sanal adaptör kontrolü
                is_virtual = any(x in description.lower() for x in
                                 ['hyper-v', 'virtualbox', 'vmware', 'virtual', 'loopback', 'tunnel'])

                info = {
                    'name': iface_name,
                    'description': description,
                    'ip': iface.ip,
                    'is_ethernet': is_ethernet,
                    'is_virtual': is_virtual
                }

                # Fiziksel Ethernet adaptörlerini önceliklendir
                if is_ethernet and not is_virtual:
                    ethernet_interfaces.insert(0, info)
                else:
                    ethernet_interfaces.append(info)

        return ethernet_interfaces

    except Exception as e:
        print(f"Hata: {e}")
        return []


def test_interface_traffic(iface_name, duration=10):
    """Bir interface'te trafik var mı test et"""
    print(f"\n📡 '{iface_name}' üzerinde {duration} saniye trafik dinleniyor...")
    print("   (Bu süre zarfında tarayıcıda bir sayfa açın veya ping atın)")

    packet_count = [0]

    def packet_callback(pkt):
        packet_count[0] += 1
        if packet_count[0] % 10 == 0:
            print(f"\r   Yakalanan paket: {packet_count[0]}", end='', flush=True)

    try:
        sniff(iface=iface_name, prn=packet_callback, timeout=duration,
              store=False, filter="ip")
        print(f"\n✓ Toplam paket: {packet_count[0]}")
        return packet_count[0]
    except Exception as e:
        print(f"\n✗ Hata: {e}")
        return 0


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          AĞ ADAPTÖRÜ TEST ARACI                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Admin kontrolü
    if sys.platform == 'win32':
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("⚠️  UYARI: Program yönetici yetkisi ile çalıştırılmıyor!")
                print("   'Yönetici olarak çalıştır' kullanın.\n")
                input("Devam etmek için Enter'a basın...")
        except:
            pass

    interfaces = get_active_ethernet_interface()

    if not interfaces:
        print("❌ Hiçbir ağ adaptörü bulunamadı!")
        print("   Npcap kurulu olduğundan emin olun.")
        sys.exit(1)

    print("\n🔍 BULUNAN AĞ ADAPTÖRLERI:")
    print("=" * 70)

    for i, iface in enumerate(interfaces, 1):
        marker = "✓" if iface['is_ethernet'] and not iface['is_virtual'] else "○"
        tags = []

        if iface['is_ethernet']:
            tags.append("ETHERNET")
        if iface['is_virtual']:
            tags.append("SANAL")

        tag_str = f" [{', '.join(tags)}]" if tags else ""

        print(f"\n{marker} {i}. {iface['description']}{tag_str}")
        print(f"   IP Adresi: {iface['ip']}")
        print(f"   Scapy Adı: {iface['name']}")

    print("\n" + "=" * 70)
    print("\n💡 Önerilen: Yukarıda ✓ işaretli olan fiziksel Ethernet adaptörü")
    print("=" * 70)

    # Test seçeneği
    print("\n📊 TEST MENÜSÜ:")
    print("1. Adaptörde trafik testi yap (önerilen)")
    print("2. Direkt anomali detector'ı başlat")
    print("0. Çıkış")

    choice = input("\nSeçiminiz: ").strip()

    if choice == '1':
        print("\n" + "=" * 70)
        print("TRAFİK TESTİ")
        print("=" * 70)

        print("\nHangi adaptörü test etmek istiyorsunuz?")
        for i, iface in enumerate(interfaces, 1):
            print(f"{i}. {iface['description']}")

        try:
            idx = int(input("\nSeçim: ")) - 1
            if 0 <= idx < len(interfaces):
                selected = interfaces[idx]
                print(f"\n✓ Test ediliyor: {selected['description']}")

                count = test_interface_traffic(selected['name'], duration=10)

                print("\n" + "=" * 70)
                if count > 0:
                    print(f"✓ BAŞARILI! Bu adaptör çalışıyor ({count} paket)")
                    print(f"✓ Anomali detector için bu adaptörü kullanın:")
                    print(f"   → {selected['description']}")
                else:
                    print("✗ Bu adaptörde trafik tespit edilemedi!")
                    print("  Olası nedenler:")
                    print("  - Yanlış adaptör seçildi")
                    print("  - Ethernet kablosu bağlı değil")
                    print("  - Ağ bağlantısı yok")
                print("=" * 70)
        except:
            print("Geçersiz seçim!")

    elif choice == '2':
        print("\n✓ realtime_detector.py'yi çalıştırın ve uygun adaptörü seçin.")

    print("\n👋 Program sonlandı.")


if __name__ == "__main__":
    main()