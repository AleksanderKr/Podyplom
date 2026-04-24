import argparse
import cv2
import pytesseract
import easyocr
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wymagane dla Windowsa, jeśli Tesseract nie jest w PATH (Zmień ścieżkę wedle potrzeb)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def stage1_mnist_basics():
    """Etap 1: Klasyfikacja cyfr z użyciem klasycznego ML na zbiorze MNIST."""
    print("[INFO] Pobieranie zbioru MNIST (może chwilę potrwać)...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Trenowanie modelu Random Forest...")
    # ZADANIE DLA STUDENTA: Zaimplementuj klasyfikator. My używamy prostego Lasu Losowego.
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"[WYNIK] Dokładność klasyfikacji na surowych pikselach: {acc * 100:.2f}%")


def stage2_tesseract_preprocessing(image_path, lang):
    """Etap 2: Tesseract + OpenCV preprocessing."""
    print(f"[INFO] Przetwarzanie obrazu: {image_path} (Język: {lang})")
    image = cv2.imread(image_path)

    if image is None:
        print("[BŁĄD] Nie można wczytać obrazu. Sprawdź ścieżkę.")
        return

    # ZADANIE DLA STUDENTA: Zaimplementuj preprocessing w OpenCV
    # 1. Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Binaryzacja (np. adaptacyjna lub Otsu)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    print("[INFO] Rozpoznawanie tekstu za pomocą Tesseract...")
    # Parametr psm 6 oznacza założenie pojedynczego bloku tekstu
    text = pytesseract.image_to_string(thresh, config='--psm 6', lang=lang)

    print("\n--- Rozpoznany tekst (Tesseract) ---")
    print(text.strip())
    print("------------------------------------\n")

    # Wyświetlenie oryginalnego i przetworzonego obrazu (do debugowania)
    cv2.imshow("Oryginal", image)
    cv2.imshow("Preprocessed", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stage3_easyocr_deep_learning(image_path, langs_str, use_gpu):
    """Etap 3: Nowoczesny OCR oparty na sieciach neuronowych (EasyOCR)."""
    # Zamiana stringa (np. "pl,en") na listę (['pl', 'en']) wymaganą przez EasyOCR
    lang_list = langs_str.split(',')

    device_name = "GPU" if use_gpu else "CPU"
    print(f"[INFO] Analiza obrazu {image_path} przy użyciu EasyOCR (Deep Learning)...")
    print(f"[INFO] Wybrane języki: {lang_list} | Urządzenie: {device_name}")

    # Inicjalizacja czytnika
    reader = easyocr.Reader(lang_list, gpu=use_gpu)

    results = reader.readtext(image_path)

    print("\n--- Rozpoznany tekst (EasyOCR) ---")
    image = cv2.imread(image_path)

    for (bbox, text, prob) in results:
        print(f"Tekst: '{text}' | Pewność: {prob:.4f}")
        # Opcjonalnie: ZADANIE DLA STUDENTA - narysować ramki na obrazie (bounding boxes)
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print("----------------------------------\n")
    cv2.imshow("EasyOCR Wynik", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zadanie: Implementacja i porównanie systemów OCR.")
    parser.add_argument("--mode", type=str, required=True, choices=["mnist", "tesseract", "easyocr"],
                        help="Wybierz tryb działania algorytmu.")
    parser.add_argument("--image", type=str,
                        help="Ścieżka do obrazu (wymagane dla trybów tesseract i easyocr).")

    # Nowe parametry
    parser.add_argument("--lang", type=str, default="pol",
                        help="Tesseract: np. 'pol', 'eng', 'pol+eng'. EasyOCR: np. 'pl', 'en', 'pl,en' (oddzielone przecinkiem).")
    parser.add_argument("--gpu", action="store_true",
                        help="Użyj karty graficznej (GPU). Działa tylko z modelem EasyOCR i wymaga zainstalowanej biblioteki CUDA.")

    args = parser.parse_args()

    if args.mode == "mnist":
        stage1_mnist_basics()
    elif args.mode == "tesseract":
        if not args.image:
            print("[BŁĄD] Tryb 'tesseract' wymaga podania ścieżki do obrazu poprzez --image.")
        else:
            stage2_tesseract_preprocessing(args.image, args.lang)
    elif args.mode == "easyocr":
        if not args.image:
            print("[BŁĄD] Tryb 'easyocr' wymaga podania ścieżki do obrazu poprzez --image.")
        else:
            # Domyślnie args.lang ma wartość "pol", co zepsuje EasyOCR.
            # Zróbmy małą sprytną podmiankę, jeśli użytkownik nie nadpisze tego parametru:
            lang_easyocr = "pl,en" if args.lang == "pol" else args.lang
            stage3_easyocr_deep_learning(args.image, lang_easyocr, args.gpu)