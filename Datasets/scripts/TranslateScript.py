import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from dataset_paths import DATASET_DIR, OUTPUT_DIR

input_file = DATASET_DIR / 'inputs' / 'clickbait_data.csv'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / 'clickbait_data_ru.csv'
temp_file = OUTPUT_DIR / 'clickbait_data_ru_temp.csv'

print("1. Загрузка датасета...")
# Читаем CSV. Если в вашем файле есть заголовки, замените header=None на header=0
df = pd.read_csv(input_file, header=None, names=['text', 'label'])
texts = df['text'].astype(str).fillna('').tolist()

print("2. Загрузка модели и токенизатора (около 300 МБ, только при первом запуске)...")
model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Автоматически определяем, есть ли видеокарта (GPU), иначе используем процессор (CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Модель успешно загружена на устройство: {device.upper()}")

print("3. Начинаем локальный перевод. Лимитов и блокировок нет!")
all_translations = []
batch_size = 32  # Оптимальный размер батча для баланса скорости и потребления памяти

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    
    # 1. Токенизация текста (превращаем слова в числа для модели)
    inputs = tokenizer(
        batch_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)
    
    # 2. Генерация перевода
    with torch.no_grad(): # Отключаем подсчет градиентов для ускорения и экономии памяти
        outputs = model.generate(
            **inputs, 
            max_length=512, 
            num_beams=4,        # Поиск по 4 вариантам для более качественного и связного перевода
            early_stopping=True
        )
    
    # 3. Декодирование чисел обратно в русский текст
    translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_translations.extend(translated_batch)
    
    print(f"➡️ Переведено {min(i+batch_size, len(texts))} из {len(texts)} строк...")
    
    # Сохраняем промежуточный результат, чтобы не потерять прогресс при внезапном сбое
    temp_df = pd.DataFrame({
        'text_ru': all_translations, 
        'label': df['label'].iloc[:len(all_translations)]
    })
    temp_df.to_csv(temp_file, index=False, encoding='utf-8')

# Финальное сохранение полного файла с исходным текстом, переводом и меткой
df['text_ru'] = all_translations
df[['text', 'text_ru', 'label']].to_csv(output_file, index=False, encoding='utf-8')

# Удаляем временный файл после успешного завершения
if os.path.exists(temp_file):
    os.remove(temp_file)

print(f"\n🎉 Успешно! Полностью переведенный файл сохранен как: {output_file}")
