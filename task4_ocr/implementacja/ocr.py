import argparse
import cv2
import pytesseract
import easyocr
import torch
import numpy as np
import json
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), median_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def stage1_mnist():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[RES] MNIST Accuracy: {acc * 100:.2f}%")


def stage2_tesseract(image_path, lang):
    img = cv2.imread(image_path)
    if img is None: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 6', lang=lang)
    print("\n--- Tesseract Output ---\n", text.strip())
    cv2.namedWindow("Tesseract", cv2.WINDOW_NORMAL)
    cv2.imshow("Tesseract", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stage3_easyocr(image_path, langs_str, use_gpu, llm_model):
    import ollama
    lang_list = langs_str.split(',')
    img = cv2.imread(image_path)
    if img is None: return

    deskewed_img = deskew_image(img)
    reader = easyocr.Reader(lang_list, gpu=use_gpu)
    raw_results = reader.readtext(deskewed_img, paragraph=True)

    llm_input_data = []
    for (bbox, text) in raw_results:
        y_center = int((bbox[0][1] + bbox[2][1]) / 2)
        x_center = int((bbox[0][0] + bbox[1][0]) / 2)
        llm_input_data.append({"text": text, "x": x_center, "y": y_center})
        cv2.rectangle(deskewed_img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)

    if llm_model:
        llm_input_data.sort(key=lambda i: (i['y'], i['x']))
        json_payload = json.dumps(llm_input_data, ensure_ascii=False)

        # PASS 1: Spatial Reconstruction & Initial Cleaning
        prompt1 = (
                "SYSTEM: You are a professional OCR layout restorer. Use the provided JSON coordinates (x, y) "
                "to reconstruct the original text flow and document structure. "
                "Fix obvious character-level OCR errors. Respond in the SAME LANGUAGE as the source text. "
                "STRICT: Output ONLY the reconstructed text. No introductions, no explanations, no markdown notes. "
                "USER: " + json_payload
        )

        res1 = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt1}])
        pass1_text = res1['message']['content']
        print("\n--- LLM PASS 1 (Reconstruction) ---\n", pass1_text)

        # PASS 2: Linguistic Refinement & Logic Check
        prompt2 = (
                "SYSTEM: Jesteś ekspertem lingwistycznym i edytorem tekstów. Twoim zadaniem jest wygładzenie "
                "tekstu otrzymanego z systemu OCR. Popraw błędy ortograficzne, interpunkcyjne i gramatyczne. "
                "Zadbaj o logiczną spójność i czytelność tekstu, usuwając pozostałości szumu z OCR (przypadkowe znaki). "
                "Zachowaj oryginalny ton i formatowanie dokumentu. Zwróć TYLKO czysty, poprawiony tekst w języku źródłowym. "
                "USER: " + pass1_text
        )

        res2 = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt2}])
        final_text = res2['message']['content']
        print("\n--- LLM PASS 2 (Final Refinement) ---\n", final_text)

    cv2.namedWindow("OCR Result", cv2.WINDOW_NORMAL)
    cv2.imshow("OCR Result", deskewed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["mnist", "tesseract", "easyocr"])
    parser.add_argument("--image", type=str)
    parser.add_argument("--lang", type=str, default="pol")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--llm", type=str)
    args = parser.parse_args()

    if args.mode == "mnist":
        stage1_mnist()
    elif args.mode == "tesseract":
        stage2_tesseract(args.image, args.lang)
    elif args.mode == "easyocr":
        stage3_easyocr(args.image, "pl,en" if args.lang == "pol" else args.lang, args.gpu, args.llm)

r"""
python ocr.py --mode easyocr --image recept_skew.jpg --gpu --llm llama3
"""