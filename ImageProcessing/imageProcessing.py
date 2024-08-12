import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, ttk, Button, Label, Frame, filedialog, messagebox
from PIL import Image, ImageTk
from scipy.signal import convolve2d
import math

# LBP işlemleri
def pixel_degistirme(img, center, x, y):
    new_value = 0 
    try:
        if img[x][y] >= center:  # Eğer (x, y) konumundaki piksel değeri merkez piksel değerine eşit veya büyükse
            new_value = 1  # yeni değer 1 olarak ayarlanır.
    except:
        pass  # Eğer (x, y) konumu resmin dışında kalırsa, bir şey yapmadan geçeriz.
    return new_value  # Hesaplanan yeni değeri döndürür.

""" ****************************************************************************************** """
def lbp_hesapla(img, x, y):
    center = img[x][y]  # Merkezi pikselin değeri alınır.
    val_ar = []  # Komşu piksellerin değerlerini tutacak listeyi başlatıyoruz.
    
    # Sekiz komşu pikselin değerlerini hesaplayıp listeye ekliyoruz.
    val_ar.append(pixel_degistirme(img, center, x-1, y-1))  # Sol üst komşu
    val_ar.append(pixel_degistirme(img, center, x-1, y))    # Üst komşu
    val_ar.append(pixel_degistirme(img, center, x-1, y+1))  # Sağ üst komşu
    val_ar.append(pixel_degistirme(img, center, x, y+1))    # Sağ komşu
    val_ar.append(pixel_degistirme(img, center, x+1, y+1))  # Sağ alt komşu
    val_ar.append(pixel_degistirme(img, center, x+1, y))    # Alt komşu
    val_ar.append(pixel_degistirme(img, center, x+1, y-1))  # Sol alt komşu
    val_ar.append(pixel_degistirme(img, center, x, y-1))    # Sol komşu
    
    # Komşu piksellerin ağırlıklarını belirleyen 2'nin katları
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    
    val = 0  # LBP değeri için başlangıç değeri
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]  # Her komşu pikselin değeri ile onun ağırlığını çarparak toplama ekleriz.
    
    return val  # Hesaplanan LBP değeri döndürülür.
 
""" ****************************************************************************************** """

def create_blank_image(height, width):
    # Verilen yükseklik ve genişlikte sıfırlardan oluşan boş bir görüntü oluşturur.
    img_lbp = np.zeros((height, width), np.uint8)
    return img_lbp

""" ****************************************************************************************** """

def convert_to_grayscale(img_bgr, height, width):
    # Verilen yükseklik ve genişlikte sıfırlardan oluşan gri tonlamalı bir görüntü oluşturur.
    img_gray = np.zeros((height, width), np.uint8)
    
    for y in range(height):  # Yükseklik boyunca tüm satırları dolaşır.
        for x in range(width):  # Genişlik boyunca tüm sütunları dolaşır.
            blue, green, red = img_bgr[y][x]  # BGR görüntüsünden mavi, yeşil ve kırmızı bileşenleri alır.
            # Gri ton değeri hesaplanır. (Yaygın olarak kullanılan bir ağırlıklandırma yöntemi)
            gray_value = int(0.114 * blue + 0.587 * green + 0.299 * red)
            img_gray[y][x] = gray_value  # Hesaplanan gri ton değeri görüntüye atanır.
    
    return img_gray  # Gri tonlamalı görüntü döndürülür.

""" ****************************************************************************************** """

# HOG işlemleri
class HogDescriptor:
    def __init__(self, img, cell_size=8, bin_size=9, block_size=2):
        self.img = img  # İşlenecek görüntü
        self.cell_size = cell_size  # Hücre boyutu
        self.bin_size = bin_size  # Histogram bin sayısı
        self.block_size = block_size  # Blok boyutu
        self.angle_unit = 360 / self.bin_size  # Her bin'in açı birimi
        assert type(self.bin_size) == int, "bin_size should be integer,"  # bin_size'ın tamsayı olup olmadığını kontrol eder
        assert type(self.cell_size) == int, "cell_size should be integer,"  # cell_size'ın tamsayı olup olmadığını kontrol eder
        assert self.angle_unit.is_integer(), "bin_size should be divisible by 360"  # bin_size'ın 360'a bölünebilir olup olmadığını kontrol eder
        self.angle_unit = int(self.angle_unit)  # Açıyı tam sayı yapar

    def extract(self):
        self.apply_gamma_correction()  # Gamma düzeltmesi normalizasyonda parlaklığı ayarlar uygular
        self.local_contrast_normalization()  # Yerel kontrast normalizasyonu uygular

        height, width = self.img.shape  # Görüntünün boyutlarını alır
        gradient_magnitude, gradient_angle = self.global_gradient()  # Küresel gradyan hesaplar
        gradient_magnitude = abs(gradient_magnitude)  # Gradyan büyüklüğünü pozitif yapar

        # Hücre gradyan vektörünü hazırlar
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # Hücre içindeki gradyan büyüklüğünü (şiddet) ve açılarını alır
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # Hücre gradyanını hesaplar ve vektöre ekler
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # HOG görüntüsünü oluşturur
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        # Blok normalizasyonu
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - self.block_size + 1):
            for j in range(cell_gradient_vector.shape[1] - self.block_size + 1):
                block_vector = []
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        # Blok içindeki hücrelerin gradyan vektörlerini blok vektörüne ekler
                        block_vector.extend(cell_gradient_vector[i + x][j + y])
                # Blok vektörünün büyüklüğünü hesaplar
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    # Blok vektörünü normalize eder
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)  # Blok vektörünü HOG vektörüne ekler

        return hog_vector, hog_image  # HOG vektörünü ve HOG görüntüsünü döndürür

    def global_gradient(self):
        height, width = self.img.shape  # Görüntünün boyutlarını alır
        gradient_values_x = np.zeros((height, width), dtype=np.float32)  # X yönündeki gradyan değerlerini tutar
        gradient_values_y = np.zeros((height, width), dtype=np.float32)  # Y yönündeki gradyan değerlerini tutar

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # X ve Y yönlerindeki gradyanları hesaplar
                gradient_values_x[i, j] = self.img[i, j + 1] - self.img[i, j - 1]
                gradient_values_y[i, j] = self.img[i + 1, j] - self.img[i - 1, j]

        # Gradyan büyüklüğünü ve açısını hesaplar
        gradient_magnitude = np.sqrt(gradient_values_x ** 2 + gradient_values_y ** 2)
        gradient_angle = np.arctan2(gradient_values_y, gradient_values_x) * (180 / np.pi) % 360

        return gradient_magnitude, gradient_angle  # Gradyan büyüklüğünü ve açısını döndürür

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size  # Bin merkezlerini başlatır
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]  # Hücre içindeki gradyan büyüklüğü
                gradient_angle = cell_angle[i][j]  # Hücre içindeki gradyan açısı
                # En yakın iki binin açılarını ve modunu hesaplar
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                # Açı ve mod değerlerine göre bin merkezlerini günceller
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers  # Hücre gradyan vektörünü döndürür

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)  # En yakın bin indeksini hesaplar
        mod = gradient_angle % self.angle_unit  # Mod değerini hesaplar
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod  # En yakın iki binin indekslerini ve mod değerini döndürür
        return idx, (idx + 1) % self.bin_size, mod  # En yakın iki binin indekslerini ve mod değerini döndürür

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size // 2  # Hücre genişliğinin yarısını alır
        max_mag = np.array(cell_gradient).max()  # Hücre gradyanlarının maksimum değerini alır
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]  # Hücre gradyan vektörünü alır
                cell_grad /= max_mag  # Hücre gradyanını normalize eder
                angle = 0
                angle_gap = self.angle_unit  # Bin açısı aralığını alır
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)  # Açıyı radyan cinsine çevirir
                    # Gradyan yönünde çizgiler çizer
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * magnitude))  # Çizgiyi çiz
                    angle += angle_gap  # Açıyı güncelle
        return image  # HOG görüntüsünü döndür

    def apply_gamma_correction(self):
        self.img = np.where(self.img >= 0, self.img ** 2.5, 0)  # Gamma düzeltmesi uygular

    def local_contrast_normalization(self):
        mean = np.mean(self.img)  # Görüntünün ortalamasını hesaplar
        std = np.std(self.img)  # Görüntünün standart sapmasını hesaplar
        self.img = (self.img - mean) / std if std != 0 else self.img  # Kontrast normalizasyonu uygular
""" ****************************************************************************************** """
def plot_histogram(hog_vector, histogram_label):
    plt.figure(figsize=(10, 5))  # Yeni bir figür oluşturur ve boyutunu 10x5 inç olarak ayarlar
    plt.title("Histogram of Oriented Gradients")  # Grafiğin başlığını "Histogram of Oriented Gradients" olarak ayarlar
    plt.xlabel("Frequency")  # X eksenini "Bins" olarak etiketler
    plt.ylabel("Bins")  # Y eksenini "Frequency" olarak etiketler
    plt.hist(np.concatenate(hog_vector), bins=50)  # HOG vektörünü birleştirir ve 50 binlik bir histogram çizer
    plt.savefig('histogram.png')  # Histogramı 'histogram.png' dosyasına kaydeder
    plt.close()  # Mevcut figürü kapatır

    histogram_image = Image.open('histogram.png')  # 'histogram.png' dosyasını açar
    histogram_image = histogram_image.resize((300, 300))  # Histogram görüntüsünü 300x300 piksel boyutuna yeniden boyutlandırır
    histogram_image = ImageTk.PhotoImage(histogram_image)  # Tkinter ile uyumlu bir görüntü nesnesi oluşturur
    histogram_label.config(image=histogram_image)  # histogram_label etiketinin görüntü parametresini oluşturulan Tkinter görüntüsü ile günceller
    histogram_label.image = histogram_image  # Histogram görüntüsünü referans olarak tutar (çöp toplayıcıdan korunur)

""" ****************************************************************************************** """
# Arayüz İşlemleri
class ImageProcessingGUI:
    def __init__(self, master):
        self.master = master  # Ana pencereyi saklar
        self.master.title("Görüntü İşleme")  # Pencere başlığını ayarlar
        self.master.geometry("1080x600")  # Pencere boyutlarını ayarlar
        self.master.config(bg="lightblue")
        # Butonların stilini ayarlar
        style = ttk.Style()
        style.configure('TButton', padding=6, relief="flat", borderwidth=1)
        style.map('TButton', background=[('active', '!disabled', 'red')], foreground=[('active', '!disabled', 'white')])
        style.configure('Red.TButton', background='red', foreground='white', borderwidth=1, relief="flat")
        style.configure('Blue.TButton', background='blue', foreground='white', borderwidth=1, relief="flat")

        # Görüntü panelini oluşturur
        self.image_panel = Frame(self.master, width=300, height=300, highlightbackground="light green", highlightthickness=1)
        self.image_panel.grid(row=0, column=1, padx=20, pady=20)

      
        # Görüntüyü göstermek için etiket oluşturur
        self.image_label = Label(self.image_panel)
        self.image_label.pack(fill="both", expand=True)
       
        # Görüntü başlığı etiketi
        self.image_title = Label(self.image_panel, text="Görüntü", font=("Helvetica", 12, "bold"),pady=1,padx=120,background="light green") 
        self.image_title.pack()
        # Histogram panelini oluşturur
        self.histogram_panel = Frame(self.master, width=300, height=300, highlightbackground="light green", highlightthickness=1)
        self.histogram_panel.grid(row=0, column=2,padx=20, pady=20)

        self.histogram_label = Label(self.histogram_panel)  # Histogramı göstermek için etiket oluşturur
        self.histogram_label.pack(fill="both", expand=True)

        self.histogram_title = Label(self.histogram_panel, text="Histogram", font=("Helvetica", 12, "bold"), pady=1,padx=120,background="light green")  # Histogram başlığı etiketi
        self.histogram_title.pack()

        # İşlem sonrası görüntü panelini oluşturur
        self.final_panel = Frame(self.master, width=300, height=300, highlightbackground="light green", highlightthickness=1)
        self.final_panel.grid(row=0, column=3,padx=20, pady=20)

        self.final_label = Label(self.final_panel)  # İşlem sonrası görüntüyü göstermek için etiket oluşturur
        self.final_label.pack(fill="both", expand=True)

        self.final_title = Label(self.final_panel, text="Sonuç", font=("Helvetica", 12, "bold"),  pady=1,padx=126,background="light green")  # İşlem sonrası görüntü başlığı etiketi
        self.final_title.pack()

        # Buton panelini oluşturur
        self.button_panel = Frame(self.master,background="light blue")
        self.button_panel.grid(row=1, column=0, columnspan=5, pady=50)

        # Resim seçme butonu oluşturur
        self.select_button = Button(self.button_panel, text="Resim Seç", bg='blue', fg='white', command=self.select_image,cursor="hand2",overrelief="groove",activebackground="lime", width=15)
        self.select_button.pack(side="left", padx=10)

        # LBP (Local Binary Pattern) işleme butonu oluşturur
        self.lbp_button = Button(self.button_panel, text="LBP İşlemi", bg='red', fg='white', command=self.process_lbp,cursor="hand2",overrelief="groove",activebackground="lime", width=15)
        self.lbp_button.pack(side="left", padx=20)

        # HOG (Histogram of Oriented Gradients) işleme butonu oluşturur
        self.hog_button = Button(self.button_panel, text="HOG İşlemi", bg='red', fg='white', command=self.process_hog, cursor="hand2",overrelief="groove",activebackground="lime",width=15)
        self.hog_button.pack(side="left", padx=10)

        self.selected_image_path = None  # Seçilen görüntünün dosya yolunu saklar
        self.img_bgr = None  # Seçilen görüntüyü saklar

    def select_image(self):
        self.selected_image_path = filedialog.askopenfilename()  # Dosya seçme dialogunu açar ve seçilen dosya yolunu alır
        if self.selected_image_path:  # Eğer bir dosya seçildiyse
            self.show_selected_image()  # Seçilen görüntüyü göster
            self.image_panel.config(highlightbackground="lime")  # Görüntü panelinin çerçeve rengini yeşil yapar

    def show_selected_image(self):
        self.img_bgr = cv2.imread(self.selected_image_path)  # Seçilen görüntüyü okur
        self.img_bgr = cv2.resize(self.img_bgr, (300, 300))  # Görüntüyü sabit boyuta yeniden boyutlandırır
        img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB formatına çevirir
        img_pil = Image.fromarray(img_rgb)  # Görüntüyü PIL formatına çevirir
        img_tk = ImageTk.PhotoImage(image=img_pil)  # Görüntüyü Tkinter formatına çevirir

        self.image_label.config(image=img_tk)  # Görüntü etiketini günceller
        self.control=True
        self.image_label.image = img_tk  # Görüntü referansını saklar (çöp toplayıcıdan korunur)

    def process_lbp(self):
        # Mesaj Bilgisi
        mesaj = messagebox.showinfo(
        title="Bilgi",
        message="LBP İşlemi Uygulanacak"
        )
        print(mesaj)
        if self.img_bgr is not None:  # Eğer bir görüntü seçilmişse
            height, width, _ = self.img_bgr.shape  # Görüntü boyutlarını alır
            img_gray = convert_to_grayscale(self.img_bgr, height, width)  # Görüntüyü gri tonlamaya çevirir
            img_lbp = create_blank_image(height, width)  # Boş bir LBP görüntüsü oluşturur

            for i in range(height):
                for j in range(width):
                    img_lbp[i, j] = lbp_hesapla(img_gray, i, j)  # Her piksel için LBP değeri hesaplar

            histogram = np.zeros(256)  # 256 binlik bir histogram oluşturur
            for i in range(img_lbp.shape[0]):
                for j in range(img_lbp.shape[1]):
                    pixel_value = img_lbp[i, j]  # LBP piksel değerini alır
                    histogram[pixel_value] += 1  # Histogramı günceller

            plt.bar(range(256), histogram, color='blue')  # Histogram çubuğu oluşturur
            plt.xlabel('Piksel Değeri')  # X eksenini etiketler
            plt.ylabel('Toplam Sayı')  # Y eksenini etiketler
            plt.title('Resmin Histogramı')  # Histogram başlığını ayarlar
            plt.savefig('histogram.png')  # Histogramı dosyaya kaydeder
            plt.close()  # Mevcut figürü kapatır

            self.image_panel.config(highlightbackground="lime", highlightthickness=3)  # Görüntü panelinin çerçeve rengini yeşil yapar

            histogram_image = Image.open('histogram.png')  # Histogram görüntüsünü açar
            histogram_image = histogram_image.resize((300, 300))  # Histogram görüntüsünü yeniden boyutlandırır
            histogram_image = ImageTk.PhotoImage(histogram_image)  # Histogram görüntüsünü Tkinter formatına çevirir

            self.histogram_label.config(image=histogram_image)  # Histogram etiketini günceller
            self.histogram_label.image = histogram_image  # Histogram görüntüsünü referans olarak tutar (çöp toplayıcıdan korunur)

            self.histogram_panel.config(highlightbackground="lime", highlightthickness=3)  # Histogram panelinin çerçeve rengini yeşil yapar
            final_image = Image.fromarray(cv2.cvtColor(img_lbp, cv2.COLOR_GRAY2RGB))  # LBP görüntüsünü RGB formatına çevirir
            final_image = final_image.resize((300, 300))  # LBP görüntüsünü yeniden boyutlandırır
            final_image = ImageTk.PhotoImage(final_image)  # LBP görüntüsünü Tkinter formatına çevirir

            self.final_label.config(image=final_image)  # İşlem sonrası görüntü etiketini günceller
            self.final_label.image = final_image  # İşlem sonrası görüntüyü referans olarak tutar (çöp toplayıcıdan korunur)

            self.final_panel.config(highlightbackground="lime", highlightthickness=3)  # İşlem sonrası görüntü panelinin çerçeve rengini yeşil yapar

    def process_hog(self):
        # Mesaj Bilgisi
        mesaj = messagebox.showinfo(
        title="Bilgi",
        message="HOG İşlemi Uygulanacak"
        )
        print(mesaj)
        if self.img_bgr is not None:  # Eğer bir görüntü seçilmişse
            height, width, _ = self.img_bgr.shape  # Görüntü boyutlarını alır
            img_gray = convert_to_grayscale(self.img_bgr, height, width)  # Görüntüyü gri tonlamaya çevirir
            hog = HogDescriptor(img_gray)  # HOG tanımlayıcısını oluşturur
            hog_vector, hog_image = hog.extract()  # HOG vektörü ve görüntüsünü çıkarır
            plot_histogram(hog_vector, self.histogram_label)  # Histogramı panelde göster

            self.image_panel.config(highlightbackground="lime", highlightthickness=3)  # Görüntü panelinin çerçeve rengini yeşil yapar

            histogram_image = Image.open('histogram.png')  # Histogram görüntüsünü açar
            histogram_image = histogram_image.resize((300, 300))  # Histogram görüntüsünü yeniden boyutlandırır
            histogram_image = ImageTk.PhotoImage(histogram_image)  # Histogram görüntüsünü Tkinter formatına çevirir

            self.histogram_label.config(image=histogram_image)  # Histogram etiketini günceller
            self.histogram_label.image = histogram_image  # Histogram görüntüsünü referans olarak tutar (çöp toplayıcıdan korunur)

            self.histogram_panel.config(highlightbackground="lime", highlightthickness=3)  # Histogram panelinin çerçeve rengini yeşil yapar

            hog_img = Image.fromarray(hog_image)  # HOG görüntüsünü PIL formatına çevirir
            hog_img = hog_img.resize((300, 300))  # HOG görüntüsünü yeniden boyutlandırır
            hog_img = ImageTk.PhotoImage(hog_img)  # HOG görüntüsünü Tkinter formatına çevirir

            self.final_label.config(image=hog_img)  # İşlem sonrası görüntü etiketini günceller
            self.final_label.image = hog_img  # İşlem sonrası görüntüyü referans olarak tutar (çöp toplayıcıdan korunur)

            self.final_panel.config(highlightbackground="lime", highlightthickness=3)  # İşlem sonrası görüntü panelinin çerçeve rengini yeşil yapar

# Tkinter ana döngüsünü başlatır
root = Tk()
app = ImageProcessingGUI(root)
root.mainloop()


