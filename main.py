import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Инициализация моделей InsightFace
app_insight = FaceAnalysis(name="buffalo_l")
app_insight.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 для CPU

swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)

# Храним фото пользователя в памяти
user_photos = {}

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photos = update.message.photo

    if not photos:
        await update.message.reply_text("Пожалуйста, пришли фото.")
        return

    photo = photos[-1]
    file = await photo.get_file()
    img_bytes = await file.download_as_bytearray()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if user_id not in user_photos:
        user_photos[user_id] = {'source': img_np, 'target': None}
        await update.message.reply_text("Фото получено. Теперь пришли второе фото.")
    else:
        user_photos[user_id]['target'] = img_np
        await update.message.reply_text("Выполняю подмену лица...")

        result_img = do_faceswap(user_photos[user_id]['target'], user_photos[user_id]['source'])

        if result_img is not None:
            _, buffer = cv2.imencode(".jpg", result_img)
            await update.message.reply_photo(photo=BytesIO(buffer.tobytes()))
        else:
            await update.message.reply_text("Не удалось обнаружить лица.")

        user_photos.pop(user_id, None)

def do_faceswap(target_img, source_img):
    faces_target = app_insight.get(target_img)
    faces_source = app_insight.get(source_img)

    if not faces_target or not faces_source:
        return None

    return swapper.get(target_img, faces_target[0], faces_source[0], paste_back=True)

# Запуск бота
def main():
    TOKEN = "8034098006:AAEtSO15opeQsHR4OwU9ndCM5tAR2AxK0ig"
    app = ApplicationBuilder().token(TOKEN).build()

    photo_handler = MessageHandler(filters.PHOTO, handle_photo)
    app.add_handler(photo_handler)

    print("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
