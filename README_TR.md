# 🫀 Makine Öğrenmesi ile Kalp Hastalığı Sınıflandırması (R)

Bu projede, R programlama dili kullanılarak kalp hastalığı verisi üzerinde çeşitli makine öğrenmesi algoritmaları uygulanmıştır. Proje kapsamında veri ön işleme, keşifsel veri analizi (EDA), model oluşturma, değerlendirme ve modellerin karşılaştırılması adımları gerçekleştirilmiştir.

---

## 📁 Veri Kümesi

`kalp.csv` dosyası kalp hastalığı ile ilgili klinik parametreleri içermektedir. Veri kümesinde yaş, cinsiyet, göğüs ağrısı tipi, kolesterol düzeyi gibi özellikler bulunmaktadır.

---

## 📊 Veri Keşfi ve Görselleştirme

İlk adımlar şunlardır:
- Veri yapısının ve özet istatistiklerin incelenmesi
- Eksik verilerin kontrolü
- Özelliklerin dağılımlarının görselleştirilmesi:
  - **Histogramlar** (yaş, kolesterol vs.)
  - **Çubuk grafikler** (cinsiyet, açlık kan şekeri)
  - **Pasta grafikler** (göğüs ağrısı tipi, EKG, eğim, thal)

---

## 🧼 Veri Temizleme

### ✅ Eksik Veriler:
`age` sütununa manuel olarak eklenen eksik veriler en sık tekrar eden değer (mod) ile doldurulmuştur.

### ✅ Aykırı Değerler:
Z-skoru yöntemi ile aykırı değerler belirlenmiş ve özelliklerdeki maksimum normal değer ile değiştirilmiştir.

### ✅ Özellik Ölçekleme:
İki tür ölçekleme uygulanmıştır:
- **Min-Max Normalizasyonu** (0–1 arası)
- **Z-skor Standartlaştırması** (ortalama=0, std=1)

---

## 🤖 Makine Öğrenmesi Modelleri

Projede aşağıdaki sınıflandırma modelleri eğitilip test edilmiştir:

### 1. **K-En Yakın Komşu (KNN)**
- En iyi `k` değeri doğrulama seti ile belirlenmiştir.
- Doğruluk, duyarlılık, özgüllük ve F1-skora göre değerlendirilmiştir.

### 2. **Yapay Sinir Ağı (NN)**
- `neuralnet` kütüphanesi kullanılmıştır.
- Mimari: 5, 3 ve 2 nöronlu 3 gizli katman.
- Korelasyon ve sınıflandırma metrikleri ile test edilmiştir.

### 3. **Rasgele Orman (RF)**
- 500 ağaç kullanılarak eğitilmiştir.
- Yüksek doğruluk ve sağlamlık sunar.

### 4. **Destek Vektör Makineleri (SVM)**
- Lineer ve Radyal (RBF) çekirdek fonksiyonları test edilmiştir.
- `e1071` kütüphanesi kullanılmıştır.

### 5. **Lojistik Regresyon (GLM)**
- `glm()` fonksiyonu ile binomial ailede klasik bir model kurulmuştur.

---

## 📈 Model Değerlendirme

Her model aşağıdaki metriklerle değerlendirilmiştir:

- **Doğruluk (Accuracy)**
- **Duyarlılık (Recall/Sensitivity)**
- **Özgüllük (Specificity)**
- **Kesinlik (Precision)**
- **F1-Skoru**

Tüm modellerin doğrulukları çubuk grafik ile karşılaştırılmıştır.

---

## 🏆 En İyi Modelin Seçimi ve Kaydedilmesi

- Tüm modeller test doğruluğuna göre karşılaştırılmıştır.
- En yüksek doğruluğa sahip model `.rds` dosyası olarak kaydedilmiştir.

```r
saveRDS(model, "en_iyi_model_xxx.rds")
```

---

## 🧪 Yeni Veri Tahmini

Kaydedilen model aşağıdaki gibi yüklenerek yeni hasta verisi üzerinde tahmin yapılır:

```r
yeni_veri <- data.frame(...)
tahmin <- predict(model, yeni_veri, type = "class")
```

---

## 🧰 Gereksinimler

Aşağıdaki R kütüphaneleri kurulmalıdır:

```r
install.packages(c(
  "ggplot2", "ggthemes", "plotly", "dplyr", "psych", 
  "corrplot", "class", "tidyverse", "caret", "zoo", 
  "e1071", "rstatix", "rpart", "neuralnet", "randomForest"
))
```

---

## 👤 Yazar

Bu proje, R dili kullanılarak gerçekleştirilen uygulamalı bir makine öğrenmesi çalışmasıdır.

---
