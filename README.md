# 🛍️ Müşteri Segmentasyonu Projesi

Bu proje, bir alışveriş merkezindeki müşterilerin segmentasyonunu gerçekleştirmek amacıyla yapılmıştır. Müşteri verileri kullanılarak, farklı müşteri grupları belirlenip, bu grupların özellikleri analiz edilmiştir. Amacımız, her bir müşteri segmentinin davranışlarını anlamak ve bu bilgileri pazarlama stratejilerine entegre etmektir.

## 🔗 Kaggle Veri Seti
[Mall Customer Segmentation Data]

## 🔗 Hugging Face Uygulaması
[Müşteri Segmentasyonu - Hugging Face Space]

## 📊 Proje Aşamaları
1. **Veri Yükleme**:
   - `Mall_Customers.csv` dosyası yüklenir.

2. **Veri Analizi ve Görselleştirme**:
   - Yaş ve yıllık gelir dağılımı için histogramlar oluşturulur.
   - Cinsiyete göre harcama puanı analizi yapılır.

3. **Özellik Mühendisliği**:
   - Cinsiyet değişkeni sayısal verilere dönüştürülür.
   - Özellikler olarak yaş, yıllık gelir ve harcama puanı seçilir.

4. **Veri Ölçeklendirme**:
   - Seçilen özellikler için standardizasyon yapılır.

5. **Kümeleme Modeli**:
   - K-Means algoritması ile müşteri segmentleri belirlenir.
   - Optimal küme sayısı Elbow yöntemi ile belirlenir.

6. **Sonuçların Analizi**:
   - Kümeleme sonuçları görselleştirilir.
   - Silhouette skoru ile kümeleme kalitesi değerlendirilir.

7. **Hiyerarşik Kümeleme ve DBSCAN**:
   - Hiyerarşik kümeleme ve DBSCAN algoritmaları uygulanır.

8. **Küme Özellikleri**:
   - Her kümenin özellik ortalamaları hesaplanır ve yorumlanır.

## 📈 Küme Özellikleri
- **Küme 0**: Genç (20-35 yaş), orta gelir, yüksek harcama → Hedef Kitle.
- **Küme 1**: Orta yaş (35-50), yüksek gelir, düşük harcama → Premium Ürünler İçin Potansiyel.
- **Küme 2**: Genç (18-30), düşük gelir, yüksek harcama → İndirim Kampanyalarına Duyarlı.
- **Küme 3**: Yaşlı (50+), orta gelir, düşük harcama → Sadakat Programları.
- **Küme 4**: Genç (25-40), yüksek gelir, orta harcama → Lüks Ürünler.
