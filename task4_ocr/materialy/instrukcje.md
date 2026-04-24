# Instrukcja przygotowania środowiska: Zadanie OCR

Niniejsza instrukcja przeprowadzi Cię przez proces konfiguracji środowiska niezbędnego do wykonania zadania z zakresu optycznego rozpoznawania znaków (OCR). Skrypt został przygotowany tak, aby działał na systemach Windows oraz Linux.

---

## 1. Instalacja bibliotek Python

Otwórz terminal (PowerShell/CMD na Windows lub Bash na Linux) i zainstaluj wymagane pakiety:

```bash
pip install opencv-python pytesseract scikit-learn easyocr matplotlib
```

**Krótki opis bibliotek:**
* `opencv-python`: Przetwarzanie obrazu (skala szarości, progowanie).
* `pytesseract`: Łącznik między Pythonem a silnikiem Tesseract.
* `scikit-learn`: Klasyczne uczenie maszynowe (dla etapu MNIST).
* `easyocr`: Nowoczesny OCR oparty na sieciach neuronowych (Deep Learning).
* `matplotlib`: Wizualizacja i wyświetlanie wyników.

---

## 2. Instalacja silnika Tesseract OCR

Sama biblioteka `pytesseract` to tylko wrapper. Musisz mieć zainstalowany silnik w systemie:

### **System Windows:**
1. Pobierz instalator: [Tesseract OCR for Windows (UB Mannheim)](https://github.com/UB-Mannheim/tesseract/wiki).
2. Zainstaluj (domyślnie: `C:\Program Files\Tesseract-OCR`).
3. **Ważne:** W pliku `ocr.py` odkomentuj linię:
   `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

### **System Linux (Ubuntu/Debian):**
1. Zainstaluj pakiet podstawowy:
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr
   ```
2. (Opcjonalnie) Zainstaluj polski pakiet językowy:
   ```bash
   sudo apt install tesseract-ocr-pol
   ```
*Na Linuxie ścieżka zazwyczaj dodaje się do PATH automatycznie, więc nie musisz nic zmieniać w kodzie.*

---

## 3. Uwagi dotyczące EasyOCR

Przy pierwszym uruchomieniu trybu `--mode easyocr`, biblioteka pobierze modele sieci neuronowych (ok. 100-500 MB). 
* Wymagane połączenie z internetem.
* Modele zostaną zapisane w katalogu domowym (np. `~/.EasyOCR/`).

---

## 4. Uruchamianie skryptu

Przykłady wywołania w zależności od wybranego etapu:

1. **Etap 1: Klasyfikacja ręczna (MNIST):**
   ```bash
   python ocr.py --mode mnist
   ```

2. **Etap 2: Tradycyjny OCR (Tesseract):**
   ```bash
   python ocr.py --mode tesseract --image "materialy/foto.png"
   ```

3. **Etap 3: Deep Learning OCR (EasyOCR):**
   ```bash
   python ocr.py --mode easyocr --image "materialy/foto.png"
   ```