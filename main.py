import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def predict_text(model, class_names, text):

    class_id = int(model.predict([text])[0])
    return class_names[class_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", help="Английский текст для одного прогноза")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        parser.error("Модель не найдена. Выполните python train.py, затем python train.py --final.")
    if args.predict is not None and not args.predict.strip():
        parser.error("Введите непустой текст.")


    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    class_names = saved["class_names"]

    if args.predict is not None:
        print(predict_text(model, class_names, args.predict))
        return

    print("Модель загружена. Введите политический текст на английском.")
    print("Категории: оружие, Ближний Восток, прочая политика.")
    print("Для завершения введите /exit.")
    while True:
        try:
            text = input("\nТекст: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nРабота завершена.")
            break
        if text == "/exit":
            break
        if not text:
            print("Пустой текст нельзя классифицировать.")
            continue
        print("Категория:", predict_text(model, class_names, text))



if __name__ == "__main__":
    main()
