"""
PyTorch ile LSTM Autoencoder Anomali Tespiti (CPU) - Bellek Dostu + Geniş Metrikler
- Normal ve saldırı CSV datasetleri
- MinMaxScaler + PCA + LSTM Autoencoder
- Threshold: percentile tabanlı
- Değerlendirme:
    * Sınıflandırma: Confusion Matrix, Precision, Recall, F1
    * Regresyon: MAE, RMSE, MAPE, SMAPE, R^2
    * Olasılık: Log-Likelihood, AIC, BIC (Gaussian hata varsayımı)
    * Zaman serisi: DTW, MDA (örnek bir alt kümeyle)
- Grafikleri PNG olarak kaydeder:
    * recon_error_hist.png
    * roc_curve.png
    * pr_curve.png
"""

import os
import glob
import logging
from collections import deque

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NORMAL_DIR = r"C:\Users\ASUS\Desktop\AĞANOMALİ_2\normal_traffic_dataset"
ATTACK_DIR = r"C:\Users\ASUS\Desktop\AĞANOMALİ_2\attacker_dataset"

TARGET_FEATURE_COUNT = 19
SEQUENCE_LENGTH = 10
BATCH_SIZE = 128
EPOCHS = 20
THRESHOLD_PERCENTILE = 99.5
LEARNING_RATE = 1e-3

METRICS_DIR = "metrics_plots"  # PNG’ler ve metrikler buraya kaydedilecek

# Cihaz seçimi (SADECE CPU)
DEVICE = torch.device("cpu")
logger.info(f"PyTorch device: {DEVICE}")


class LSTMAutoencoder(nn.Module):

    #LSTM Autoencoder:
    #Encoder: İki katmanlı LSTM -> latent vektör
    #Decoder: latent'i tekrar seq_len kadar kopyalayıp LSTM + LSTM


    def __init__(self, seq_len, n_features, latent_dim=32):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder_dropout = nn.Dropout(0.2)
        self.encoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_dropout = nn.Dropout(0.2)
        self.decoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=self.n_features,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        # Encoder
        enc_out, _ = self.encoder_lstm(x)
        enc_out = self.encoder_dropout(enc_out)
        enc_out2, (h_n, c_n) = self.encoder_lstm2(enc_out)

        # Bottleneck: sadece son time-step
        latent = enc_out2[:, -1, :]  # (batch, latent_dim)

        # Decoder için latent vektörü tüm zaman adımlarına kopyala
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decoder
        dec_out, _ = self.decoder_lstm(repeated)
        dec_out = self.decoder_dropout(dec_out)
        dec_out2, _ = self.decoder_lstm2(dec_out)

        return dec_out2

class AdvancedAnomalyDetectorTorch:

    def __init__(self, sequence_length=10, use_pca=True, pca_components=0.95, latent_dim=32):
        self.sequence_length = int(sequence_length)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.latent_dim = int(latent_dim)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = None
        self.model = None
        self.threshold = None
        self.input_dim = None

        self.prediction_buffer = deque(maxlen=self.sequence_length)

    def _clean_data(self, df_or_array):
        if isinstance(df_or_array, pd.DataFrame):
            df_or_array = df_or_array.replace([np.inf, -np.inf], np.nan)
            df_or_array = df_or_array.fillna(0)
        elif isinstance(df_or_array, np.ndarray):
            df_or_array = np.copy(df_or_array)
            df_or_array[np.isinf(df_or_array)] = 0
            df_or_array[np.isnan(df_or_array)] = 0
        return df_or_array

    def fit_preprocessing(self, df_normal):
        df_clean = self._clean_data(df_normal)
        data = df_clean.values if isinstance(df_clean, pd.DataFrame) else df_clean

        data_scaled = self.scaler.fit_transform(data)

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            self.pca.fit(data_scaled)
            self.input_dim = int(self.pca.n_components_)
            logger.info(f"PCA eğitildi. Bileşen sayısı: {self.pca.n_components_}")
            logger.info(f"Açıklanan varyans: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            self.input_dim = int(data_scaled.shape[1])
            logger.info(f"PCA kullanılmıyor. Girdi boyutu: {self.input_dim}")

    def transform_data(self, df):
        df_clean = self._clean_data(df)
        data = df_clean.values if isinstance(df_clean, pd.DataFrame) else df_clean

        data_scaled = self.scaler.transform(data)

        if self.use_pca and self.pca is not None:
            return self.pca.transform(data_scaled)
        return data_scaled

    def create_sequences(self, data):
        if len(data) < self.sequence_length:
            logger.warning(
                f"Veri uzunluğu ({len(data)}) sequence uzunluğundan ({self.sequence_length}) küçük!"
            )
            return np.array([])

        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences, dtype=np.float32)

    def build_model(self):
        self.model = LSTMAutoencoder(
            seq_len=self.sequence_length,
            n_features=self.input_dim,
            latent_dim=self.latent_dim,
        ).to(DEVICE)
        logger.info("PyTorch LSTM Autoencoder modeli oluşturuldu.")
        logger.info(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        df_normal,
        epochs=20,
        batch_size=128,
        threshold_percentile=99.5,
        learning_rate=1e-3,
    ):
        logger.info("=" * 70)
        logger.info("MODEL EĞİTİMİ BAŞLIYOR")
        logger.info("=" * 70)

        # Preprocessing
        logger.info("Scaler + PCA (sadece normal veriyle) eğitiliyor...")
        self.fit_preprocessing(df_normal)

        processed = self.transform_data(df_normal)
        sequences_np = self.create_sequences(processed)

        if len(sequences_np) == 0:
            logger.error("Sequence oluşturulamadı! Veri çok kısa.")
            return

        logger.info(f"Toplam sequence sayısı: {sequences_np.shape[0]}")
        logger.info(f"Sequence shape: {sequences_np.shape}")

        X_train_np, X_val_np = train_test_split(
            sequences_np, test_size=0.2, random_state=42, shuffle=True
        )
        logger.info(f"Train sequences: {X_train_np.shape[0]}, Val sequences: {X_val_np.shape[0]}")

        X_train = torch.from_numpy(X_train_np).to(DEVICE)
        X_val = torch.from_numpy(X_val_np).to(DEVICE)

        train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size, shuffle=False)

        self.build_model()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_loss = np.inf
        best_state = None
        patience = 3
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ---- Training ----
            self.model.train()
            train_losses = []
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ---- Validation ----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    recon = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            logger.info(
                f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = self.model.state_dict().copy()
                logger.info(f"  → Yeni en iyi validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping tetiklendi (patience={patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("En iyi model ağırlıkları geri yüklendi.")

        # Threshold hesaplama
        logger.info("Threshold hesaplanıyor...")
        self.model.eval()
        all_losses = []
        with torch.no_grad():
            for batch_x, _ in train_loader:
                recon = self.model(batch_x)
                mae = torch.mean(torch.abs(recon - batch_x), dim=(1, 2))
                all_losses.extend(mae.cpu().numpy())

        self.threshold = float(np.percentile(all_losses, threshold_percentile))
        logger.info(f"Threshold ({threshold_percentile}. percentile): {self.threshold:.6f}")
        logger.info(f"Min MAE: {np.min(all_losses):.6f}, Max MAE: {np.max(all_losses):.6f}")
        logger.info("=" * 70)
        logger.info("EĞİTİM TAMAMLANDI")
        logger.info("=" * 70)

    # -------------------------- DTW --------------------------

    @staticmethod
    def _dtw_distance(s1, s2):
        n, m = len(s1), len(s2)
        dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return dtw[n, m]

    # --------------------- DEĞERLENDİRME ---------------------

    def evaluate_metrics(self, df_test, y_true, save_dir=METRICS_DIR):

        os.makedirs(save_dir, exist_ok=True)

        logger.info("\n" + "=" * 70)
        logger.info("MODEL DEĞERLENDİRME")
        logger.info("=" * 70)

        processed = self.transform_data(df_test)
        sequences_np = self.create_sequences(processed)

        if len(sequences_np) == 0:
            logger.error("Test verisi için sequence oluşturulamadı!")
            return

        X = torch.from_numpy(sequences_np).to(DEVICE)

        self.model.eval()
        all_mae = []

        # Regresyon metrikleri için toplama değişkenleri
        sse = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        count_elements = 0

        mape_sum = 0.0
        smape_sum = 0.0
        mape_count = 0

        # DTW & MDA için örnek alt küme
        sample_dtw = []
        sample_mda = []
        sample_limit = 100
        sample_used = 0

        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                batch_x = X[i : i + BATCH_SIZE]
                recon = self.model(batch_x)

                # Sequence bazlı MAE (anomali skoru)
                mae_seq = torch.mean(torch.abs(recon - batch_x), dim=(1, 2))
                all_mae.extend(mae_seq.cpu().numpy())

                # Regresyon metrikleri için element bazında
                batch_x_np = batch_x.cpu().numpy()
                recon_np = recon.cpu().numpy()

                diff = recon_np - batch_x_np
                sse += float(np.sum(diff ** 2))
                sum_y += float(np.sum(batch_x_np))
                sum_y2 += float(np.sum(batch_x_np ** 2))
                count_elements += batch_x_np.size

                eps = 1e-8
                abs_true = np.abs(batch_x_np)
                abs_pred = np.abs(recon_np)
                abs_err = np.abs(diff)

                mape_sum += float(np.sum(abs_err / (abs_true + eps)))
                smape_sum += float(
                    np.sum(2 * abs_err / (abs_true + abs_pred + eps))
                )
                mape_count += batch_x_np.size

                # DTW & MDA (sadece sınırlı sayıda sequence)
                if sample_used < sample_limit:
                    k = min(batch_x_np.shape[0], sample_limit - sample_used)
                    for j in range(k):
                        true_seq = batch_x_np[j]  # (seq_len, n_features)
                        pred_seq = recon_np[j]

                        # Özellikler üzerinden ortalama alıp 1D seriye indir
                        true_1d = true_seq.mean(axis=1)
                        pred_1d = pred_seq.mean(axis=1)

                        dtw_dist = self._dtw_distance(true_1d, pred_1d)
                        sample_dtw.append(dtw_dist)

                        true_diff = np.diff(true_1d)
                        pred_diff = np.diff(pred_1d)
                        if len(true_diff) > 0:
                            mda = np.mean(np.sign(true_diff) == np.sign(pred_diff))
                            sample_mda.append(mda)

                    sample_used += k

        all_mae = np.array(all_mae)

        # Sınıflandırma metrikleri
        y_true_clipped = y_true[self.sequence_length - 1 :]
        min_len = min(len(all_mae), len(y_true_clipped))
        all_mae = all_mae[:min_len]
        y_true_clipped = y_true_clipped[:min_len]

        y_pred = (all_mae > self.threshold).astype(int)

        cm = confusion_matrix(y_true_clipped, y_pred)
        cr = classification_report(
            y_true_clipped, y_pred, target_names=["Normal", "Attack"], digits=4
        )

        logger.info(f"\nTest örnekleri: {len(y_true_clipped)}")
        logger.info(
            f"Normal: {np.sum(y_true_clipped == 0)}, Attack: {np.sum(y_true_clipped == 1)}"
        )
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{cr}")

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = (
            2 * (precision_cls * recall_cls) / (precision_cls + recall_cls)
            if (precision_cls + recall_cls) > 0
            else 0
        )

        logger.info("\nDetaylı Sınıflandırma Metrikleri:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (Attack): {precision_cls:.4f}")
        logger.info(f"Recall (Attack): {recall_cls:.4f}")
        logger.info(f"F1-Score (Attack): {f1_cls:.4f}")

        # ---------------- Regresyon / Zaman Serisi Metrikleri ----------------

        # MAE (sequence bazlı)
        mae_seq_mean = float(all_mae.mean())
        rmse_seq = float(np.sqrt((all_mae ** 2).mean()))

        # Element bazlı metrikler
        rmse_element = float(np.sqrt(sse / (count_elements + 1e-8)))
        mae_percent = (mape_sum / (mape_count + 1e-8)) * 100.0
        smape_percent = (smape_sum / (mape_count + 1e-8)) * 100.0

        mean_y = sum_y / (count_elements + 1e-8)
        sst = sum_y2 - 2 * mean_y * sum_y + count_elements * (mean_y ** 2)
        r2 = 1.0 - sse / (sst + 1e-8)

        logger.info("\nRegresyon Bazlı Metrikler:")
        logger.info(f"Sequence MAE (ortalama): {mae_seq_mean:.6f}")
        logger.info(f"Sequence RMSE:          {rmse_seq:.6f}")
        logger.info(f"Element RMSE:          {rmse_element:.6f}")
        logger.info(f"MAPE (%):              {mae_percent:.4f}")
        logger.info(f"SMAPE (%):             {smape_percent:.4f}")
        logger.info(f"R^2:                   {r2:.6f}")

        # Gaussian hata varsayımıyla Log-Likelihood, AIC, BIC
        sigma2 = sse / (count_elements + 1e-8)
        if sigma2 <= 0:
            logger.warning("Sigma^2 <= 0, Log-Likelihood hesaplanamadı.")
            log_likelihood = aic = bic = np.nan
        else:
            log_likelihood = -0.5 * count_elements * (np.log(2 * np.pi * sigma2) + 1.0)
            k_param = 2  # mu ve sigma
            aic = 2 * k_param - 2 * log_likelihood
            bic = k_param * np.log(count_elements + 1e-8) - 2 * log_likelihood

        logger.info("\nOlasılık / Model Seçim Metrikleri (Gaussian hata varsayımı):")
        logger.info(f"Log-Likelihood: {log_likelihood:.2f}")
        logger.info(f"AIC:            {aic:.2f}")
        logger.info(f"BIC:            {bic:.2f}")

        # DTW & MDA
        mean_dtw = float(np.mean(sample_dtw)) if sample_dtw else np.nan
        mean_mda = float(np.mean(sample_mda)) if sample_mda else np.nan

        logger.info("\nZaman Serisi Özgü Metrikler (örnek alt küme):")
        logger.info(f"DTW (ortalama): {mean_dtw:.6f}")
        logger.info(f"MDA:            {mean_mda:.6f}")


        # 1) Reconstruction Error Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(all_mae, bins=100)
        plt.xlabel("Sequence Reconstruction MAE")
        plt.ylabel("Frekans")
        plt.title("Reconstruction Error Dağılımı")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(save_dir, "recon_error_hist.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        logger.info(f"Histogram kaydedildi: {hist_path}")

        # 2) ROC Curve
        fpr, tpr, _ = roc_curve(y_true_clipped, all_mae)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Eğrisi (Anomali Skoru: MAE)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.info(f"ROC eğrisi kaydedildi: {roc_path}")

        # 3) Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true_clipped, all_mae)
        ap = average_precision_score(y_true_clipped, all_mae)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Eğrisi (Anomali Skoru: MAE)")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(save_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        logger.info(f"PR eğrisi kaydedildi: {pr_path}")

        logger.info("=" * 70)


    def save(self, folder="."):
        """Model ve bileşenleri kaydet"""
        os.makedirs(folder, exist_ok=True)

        model_path = os.path.join(folder, "lstm_autoencoder_torch.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "seq_len": self.sequence_length,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
            },
            model_path,
        )

        joblib.dump(self.scaler, os.path.join(folder, "scaler.pkl"))
        joblib.dump(self.pca, os.path.join(folder, "pca.pkl"))
        joblib.dump(self.threshold, os.path.join(folder, "threshold.pkl"))

        metadata = {
            "sequence_length": self.sequence_length,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            "threshold": self.threshold,
        }
        joblib.dump(metadata, os.path.join(folder, "metadata.pkl"))

        logger.info(f"Model ve tüm bileşenler kaydedildi: {folder}")

    def load_model(self, folder="."):
        """Kaydedilmiş modeli yükle"""
        metadata = joblib.load(os.path.join(folder, "metadata.pkl"))
        self.sequence_length = int(metadata["sequence_length"])
        self.input_dim = int(metadata["input_dim"])
        self.latent_dim = int(metadata["latent_dim"])
        self.use_pca = metadata["use_pca"]
        self.pca_components = metadata["pca_components"]
        self.threshold = float(metadata["threshold"])

        model_path = os.path.join(folder, "lstm_autoencoder_torch.pt")
        checkpoint = torch.load(model_path, map_location=DEVICE)

        self.model = LSTMAutoencoder(
            seq_len=self.sequence_length,
            n_features=self.input_dim,
            latent_dim=self.latent_dim,
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
        self.pca = joblib.load(os.path.join(folder, "pca.pkl"))

        self.prediction_buffer = deque(maxlen=self.sequence_length)

        logger.info(f"Model başarıyla yüklendi: {folder}")
        logger.info(f"Threshold: {self.threshold:.6f}")


    def detect_anomaly(self, new_data):

        if isinstance(new_data, pd.Series):
            new_data = new_data.values.reshape(1, -1)
        elif isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        elif isinstance(new_data, np.ndarray):
            if new_data.ndim == 1:
                new_data = new_data.reshape(1, -1)
        else:
            new_data = np.array(new_data, dtype=np.float32).reshape(1, -1)

        new_data = self._clean_data(new_data)
        data_scaled = self.scaler.transform(new_data)

        if self.use_pca and self.pca is not None:
            data_processed = self.pca.transform(data_scaled)
        else:
            data_processed = data_scaled

        self.prediction_buffer.append(data_processed[0])

        if len(self.prediction_buffer) < self.sequence_length:
            return False, 0.0

        seq = np.array(list(self.prediction_buffer), dtype=np.float32).reshape(
            1, self.sequence_length, -1
        )
        seq_t = torch.from_numpy(seq).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(seq_t)
            mae = torch.mean(torch.abs(recon - seq_t), dim=(1, 2)).item()

        is_anomaly = bool(mae > self.threshold)
        return is_anomaly, mae



def load_and_clean_data(
    folder_path,
    label,
    max_features=19,
    max_rows_per_file=200000,
    sample_fraction=None,
):

    if not os.path.exists(folder_path):
        logger.error(f"Klasör bulunamadı: {folder_path}")
        return pd.DataFrame()

    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        logger.warning(f"Klasörde CSV dosyası bulunamadı: {folder_path}")
        return pd.DataFrame()

    logger.info(f"{folder_path} klasöründen {len(all_files)} dosya (bellek dostu) yükleniyor...")

    all_clean_dfs = []

    for f in all_files:
        try:
            logger.info(f"Yükleniyor: {os.path.basename(f)}")

            df = pd.read_csv(
                f,
                on_bad_lines="skip",
                low_memory=False,
                nrows=max_rows_per_file,
            )
            if df.empty:
                continue

            if sample_fraction is not None and 0 < sample_fraction < 1.0:
                df = df.sample(frac=sample_fraction, random_state=42)

            cols_to_drop = [
                "timestamp",
                "src_ip",
                "dst_ip",
                "sample_name",
                "ts",
                "uid",
                "id.orig_h",
                "id.resp_h",
                "id.orig_p",
                "id.resp_p",
            ]
            df.drop(
                columns=[c for c in cols_to_drop if c in df.columns],
                inplace=True,
                errors="ignore",
            )

            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                logger.warning(f"Numeric sütun yok, atlanıyor: {os.path.basename(f)}")
                continue

            df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_numeric.fillna(0, inplace=True)

            if "label" in df_numeric.columns:
                y = df_numeric["label"].values
                df_numeric.drop("label", axis=1, inplace=True)
            else:
                y = np.full(len(df_numeric), label, dtype=int)

            current_cols = df_numeric.shape[1]
            if current_cols > max_features:
                df_numeric = df_numeric.iloc[:, :max_features]
            elif current_cols < max_features:
                for i in range(max_features - current_cols):
                    df_numeric[f"pad_{i}"] = 0

            df_numeric = df_numeric.iloc[:, :max_features]
            df_numeric.columns = [f"feature_{i}" for i in range(max_features)]

            df_numeric["label"] = y

            logger.info(f"  -> Kullanılan satır sayısı: {len(df_numeric)}")
            all_clean_dfs.append(df_numeric)

        except Exception as e:
            logger.warning(f"{os.path.basename(f)} okunamadı, atlanıyor: {e}")

    if not all_clean_dfs:
        logger.warning(f"{folder_path} için geçerli veri yüklenemedi.")
        return pd.DataFrame()

    df_all = pd.concat(all_clean_dfs, ignore_index=True)
    logger.info(f"{folder_path} için toplam satır: {len(df_all)}")
    logger.info(f"Son veri boyutu: {df_all.shape}")
    return df_all

def main():
    print("\n" + "=" * 70)
    print("PyTorch (CPU) LSTM AUTOENCODER ANOMALI TESPİTİ")
    print("=" * 70 + "\n")

    logger.info("NORMAL VERİ YÜKLENİYOR...")
    df_normal = load_and_clean_data(
        NORMAL_DIR,
        label=0,
        max_features=TARGET_FEATURE_COUNT,
        max_rows_per_file=200000,
        sample_fraction=0.3,
    )

    logger.info("\nATTACK VERİ YÜKLENİYOR...")
    df_attack = load_and_clean_data(
        ATTACK_DIR,
        label=1,
        max_features=TARGET_FEATURE_COUNT,
        max_rows_per_file=200000,
        sample_fraction=0.5,
    )

    if df_normal.empty:
        logger.error("Normal veri yüklenemedi! Program sonlandırılıyor.")
        return

    logger.info("\n" + "=" * 70)
    logger.info("VERİ ÖZETİ:")
    logger.info(f"Normal veri: {len(df_normal):,} satır")
    logger.info(f"Attack veri: {len(df_attack):,} satır")
    logger.info(f"Toplam: {len(df_normal) + len(df_attack):,} satır")
    logger.info("=" * 70 + "\n")

    X_train = df_normal.drop("label", axis=1)

    detector = AdvancedAnomalyDetectorTorch(
        sequence_length=SEQUENCE_LENGTH,
        use_pca=True,
        pca_components=0.95,
        latent_dim=32,
    )

    detector.train(
        df_normal=X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        threshold_percentile=THRESHOLD_PERCENTILE,
        learning_rate=LEARNING_RATE,
    )

    if not df_attack.empty:
        logger.info("\nTest seti hazırlanıyor (normal + attack)...")
        df_test = pd.concat([df_normal, df_attack], ignore_index=True)
        X_test = df_test.drop("label", axis=1)
        y_test = df_test["label"].values
        detector.evaluate_metrics(X_test, y_test, save_dir=METRICS_DIR)
    else:
        logger.warning("Attack verisi boş, evaluation atlanıyor.")

    save_folder = "model_torch_cpu"
    detector.save(folder=save_folder)

    print("\n" + "=" * 70)
    print(f"✓ Model başarıyla eğitildi ve '{save_folder}' klasörüne kaydedildi.")
    print("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("ÖRNEK GERÇEK ZAMANLI TESPİT")
    logger.info("=" * 70)

    for i in range(min(20, len(X_train))):
        sample = X_train.iloc[i]
        is_anomaly, score = detector.detect_anomaly(sample)
        logger.info(f"Normal örnek {i+1:02d}: Anomali={is_anomaly}, Score={score:.6f}")

    if not df_attack.empty:
        X_attack = df_attack.drop("label", axis=1)
        detector.prediction_buffer.clear()
        for i in range(min(20, len(X_attack))):
            sample = X_attack.iloc[i]
            is_anomaly, score = detector.detect_anomaly(sample)
            logger.info(f"Attack örnek {i+1:02d}: Anomali={is_anomaly}, Score={score:.6f}")


if __name__ == "__main__":
    main()
