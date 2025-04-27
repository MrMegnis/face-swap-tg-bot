import os, sys

# путь к CUDA Runtime DLL
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"

# 1) Добавляем в системный PATH, чтобы shutil.which работал и Windows искал cudart64_12.dll
os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

# 2) Регистрируем как директорию для поиска DLL (Python 3.8+)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_bin)

import cv2
import time
import subprocess
import shlex
import aiohttp
import ffmpeg
import tempfile
import asyncio
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageSequence, UnidentifiedImageError
from io import BytesIO
from dotenv import load_dotenv, dotenv_values
from telegram import Update, InputFile
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters, CommandHandler
from DeepFace_Img2Img.src.main import FaceSwapper
import shutil
from pathlib import Path

import onnxruntime, os

# путь до __init__.py
base = os.path.dirname(onnxruntime.__file__)
# возможные поддиректории
candidates = [base, os.path.join(base, "capi"), os.path.join(base, "providers")]
for d in candidates:
    dll = os.path.join(d, "onnxruntime_providers_cuda.dll")
    if os.path.isfile(dll):
        print("CUDA-DLL найден в:", d)

import shutil

print("ffmpeg找CUDA DLL:", shutil.which("cudart64_12.dll"))  # должен вернуть путь
print("Доступные провайдеры:", onnxruntime.get_available_providers())
print("cudart находится здесь:", shutil.which("cudart64_12.dll"))
print("Провайдеры ONNX Runtime:", onnxruntime.get_available_providers())

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
print("Содержимое папки CUDA bin:")
for f in os.listdir(cuda_bin):
    if f.lower().startswith("cudart64"):
        print("  ", f)

swapper = FaceSwapper()

user_photos = {}
user_tasks = {}  # user_id -> {'cancel': False}

SPINNERS = ["|", "/", "-", "\\"]
SAVE_OUTPUTS = False  # Сохранять сырое и финальное видео в ./saved

# Папка, куда сохраняются временные файлы
CUSTOM_TEMP_DIR = Path("./tmp")
CUSTOM_TEMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(CUSTOM_TEMP_DIR)

USE_PARALEL_VIDEO = True

FFMPEG_BIN = "C:/Рабочий стол/_____Projects/face-swap-tg-bot/ffmpeg-2025-04-17-git-7684243fbe-essentials_build/bin/ffmpeg.exe"


async def download_telegram_file(file_path, dest_path, bot_token, progress_message):
    """
    Скачиваем файл потоково с обновлением progress_message:
     - показываем %, МБ скачано / всего.
    """
    # формируем URL
    fp = file_path
    if fp.startswith("http"):
        url = fp
    else:
        url = f"https://api.telegram.org/file/bot{bot_token}/{fp}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = 0

            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded / total * 100)
                        # обновляем каждые +2%
                        if pct - last_pct >= 2 or pct == 100:
                            last_pct = pct
                            mbs = downloaded / (1024 ** 2)
                            total_mbs = total / (1024 ** 2)
                            text = f"📥 Скачиваю: {pct}% — {mbs:.1f}/{total_mbs:.1f} МБ"
                            try:
                                await progress_message.edit_text(text)
                            except:
                                pass


def image_to_bytesio(img: Image.Image) -> BytesIO:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def swap_frame_bgr(frame_bgr, source_img):
    """
    Меняем лицо на одном BGR‑кадре — возвращает тоже BGR.
    source_img — исходный PIL.Image, а не BytesIO.
    """
    # 1) Подготовка буферов для PIL.open — для каждого кадра свой буфер!
    #    Сначала исходное лицо:
    src_buf = image_to_bytesio(source_img)
    #    Потом этот кадр:
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tgt_buf = image_to_bytesio(pil)

    # 2) Сам свап
    swapped_pil = swapper.swap_face(src_buf, tgt_buf, full_generate=False)

    # 3) Обратно в BGR
    return cv2.cvtColor(np.array(swapped_pil), cv2.COLOR_RGB2BGR)


async def process_video_parallel(source_img, video_path, output_path, progress_message, user_id):
    # 1) Считываем все кадры
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    # 2) Готовим VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 3) Параллельно свапим кадры
    swapped_dict = {}
    start_time = time.time()
    spinner = 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(swap_frame_bgr, frame, source_img): idx
            for idx, frame in enumerate(frames)
        }
        for done in as_completed(futures):
            idx = futures[done]
            try:
                swapped = done.result()
            except Exception as e:
                print(f"[ERROR] swap_frame_bgr idx={idx}", e)
                swapped = frames[idx]
            swapped_dict[idx] = swapped

            # апдейт прогресса
            await update_progress_message(
                progress_message,
                len(swapped_dict),
                total_frames,
                start_time,
                spinner
            )
            spinner += 1

    # 4) Запись в порядке
    for i in range(total_frames):
        out.write(swapped_dict[i])
    out.release()

    await edit_text_safely(progress_message, "Готово!")


def video_to_gif(input_video: str, output_gif: str, fps: int = 8, width: int = 320):
    """
    MP4 → анимированный GIF:
      1) palettegen с -frames:v 1
      2) paletteuse + loop=0
      3) удаляем палитру
    """
    palette = output_gif.replace(".gif", "_palette.png")

    # 1) Генерируем палитру — ровно один кадр!
    cmd1 = [
        FFMPEG_BIN,
        "-y",
        "-loglevel", "error",
        "-i", input_video,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
        "-frames:v", "1",  # <- ключевой момент
        palette
    ]
    subprocess.run(cmd1, check=True)

    # 2) Собираем GIF по палитре
    cmd2 = [
        FFMPEG_BIN,
        "-y",
        "-loglevel", "error",
        "-i", input_video,
        "-i", palette,
        "-filter_complex",
        f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        "-loop", "0",
        output_gif
    ]
    subprocess.run(cmd2, check=True)

    # 3) Удаляем палитру сразу по завершении
    try:
        os.remove(palette)
    except OSError:
        pass


# Безопасное обновление сообщения (без дубликатов)
async def edit_text_safely(message, new_text):
    try:
        if message.text != new_text:
            await message.edit_text(new_text)
            # await asyncio.sleep(1)
    except Exception as e:
        print("Edit message error:", e)


# Прогресс-бар с анимацией и ETA
async def update_progress_message(message, current, total, start_time, spinner_index):
    if current % 20 == 0:
        percent = int(current / total * 100)
        blocks = int(percent / 10)
        bar = '▓' * blocks + '░' * (10 - blocks)

        elapsed = time.time() - start_time
        remaining = (elapsed / current) * (total - current) if current else 0
        eta = f"{int(remaining)}s" if remaining < 120 else f"{int(remaining // 60)}m"
        spinner = SPINNERS[spinner_index % len(SPINNERS)]

        text = f"Обработка: [{bar}] {percent}% — осталось ~{eta} ⏳ {spinner}"
        await edit_text_safely(message, text)


# Обработка GIF
async def process_gif(source_img, gif_source, output_path, progress_message, user_id):
    # gif_source — либо путь, либо файлоподобный объект с методом read()/seek()
    if isinstance(gif_source, (str, Path)):
        gif = Image.open(gif_source)
    else:
        gif_source.seek(0)
        gif = Image.open(gif_source)

    frames = list(ImageSequence.Iterator(gif))
    duration = gif.info.get("duration", 50)
    total = len(frames)
    processed = []
    source_buf = image_to_bytesio(source_img)
    start_time = time.time()

    for i, frame in enumerate(frames):
        if user_tasks.get(user_id, {}).get("cancel"):
            print(f"[CANCEL] Генерация отменена для пользователя {user_id}")
            await edit_text_safely(progress_message, "⛔ Генерация отменена.")
            return
        frame_buf = BytesIO()
        frame.convert("RGB").save(frame_buf, format="PNG")
        frame_buf.seek(0)

        swapped = swapper.swap_face(source_buf, frame_buf, full_generate=False)
        processed.append(swapped.convert("P"))

        if i % 2 == 0:
            await update_progress_message(progress_message, i + 1, total, start_time, i)

    processed[0].save(output_path, save_all=True, append_images=processed[1:], loop=0, duration=duration)
    await update_progress_message(progress_message, total, total, start_time, total)


# Обработка видео
async def process_video(source_img, video_path, output_path, progress_message, user_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_num = 0
    start_time = time.time()

    try:

        while cap.isOpened():
            print(user_id, user_tasks)
            if user_tasks.get(user_id, {}).get("cancel"):
                print(f"[CANCEL] Генерация отменена для пользователя {user_id}")
                await edit_text_safely(progress_message, "⛔ Генерация отменена.")
                cap.release()
                out.release()
                return
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            swapped = swapper.swap_face(
                input_source=image_to_bytesio(source_img),
                target_source=image_to_bytesio(pil_frame),
                full_generate=False
            )

            frame_bgr = cv2.cvtColor(np.array(swapped), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            frame_num += 1
            if frame_num % 5 == 0:
                await asyncio.sleep(0)
                await update_progress_message(progress_message, frame_num, total_frames, start_time, frame_num)

    finally:
        cap.release()
        out.release()

    await edit_text_safely(progress_message, "Готово!")


# Обработка статичного изображения
def process_image(source_img, target_path):
    with open(target_path, "rb") as f:
        return swapper.swap_face(
            input_source=image_to_bytesio(source_img),
            target_source=f,
            full_generate=True
        )


def fix_video_with_ffmpeg(input_path: str, output_path: str):
    try:
        ffmpeg.input(input_path).output(
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            profile='baseline',
            preset='fast',
            movflags='faststart',
            crf=23
        ).overwrite_output().run(
            cmd=FFMPEG_BIN,
            quiet=True)
        print(f"[FFMPEG] Видео перекодировано: {output_path}")
        return True
    except ffmpeg.Error as e:
        print("[FFMPEG] Ошибка:", e.stderr.decode())
        return False


# Обработка полученного файла
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_tasks:
        user_tasks[user_id] = {"cancel": False}

    # 1) Получаем объект File (но ещё не скачиваем его целиком в память)
    try:
        if update.message.photo:
            file_ext = "jpg"
            file = await update.message.photo[-1].get_file()
        elif update.message.video:
            file_ext = "mp4"
            file = await update.message.video.get_file()
        elif update.message.animation:
            file_ext = "gif"
            file = await update.message.animation.get_file()
        elif update.message.document:
            file_ext = update.message.document.file_name.split('.')[-1].lower()
            file = await update.message.document.get_file()
        else:
            return await update.message.reply_text("Пожалуйста, пришли фото, видео или GIF.")
        fp = file.file_path
    except BadRequest as e:
        # Telegram API отказывает на файлы >20 МБ, но мы попробуем загрузить напрямую
        if "File is too big" in str(e):
            print(f"[WARN] get_file() возвратил BadRequest ({e}), пробуем прямую загрузку")
            await update.message.reply_text(
                "⚠️ Файл слишком большой для get_file(), загружаю потоково по URL…"
            )
            # Теперь инициализируем file вручную для скачивания файла по URL
            # Для видео или документа
            if update.message.video:
                fp = update.message.video.file_path
            elif update.message.document:
                fp = update.message.document.file_path
            # else:
            #     fp = update.message.photo[-1].file_id  # Для фото (хотя маловероятно)

        else:
            raise

    # 2) Создаём временный файл на диске и скачиваем в него потоково
    fd, input_path = tempfile.mkstemp(suffix=f".{file_ext}")
    os.close(fd)

    progress_msg = await update.message.reply_text("📥 Готов к скачиванию…")
    await download_telegram_file(fp, input_path, context.bot.token, progress_msg)

    await progress_msg.edit_text("✅ Скачивание завершено, начинаю обработку…")

    print(f"[DEBUG] Повторно в handle_file, file_ext={file_ext}, input_path={input_path}")

    # 3) Теперь, как раньше, работаем с input_path
    source_pil = user_photos.get(user_id, {}).get("source")
    if not source_pil:
        if file_ext in ["jpg", "jpeg", "png"]:
            img = Image.open(input_path).convert("RGB")
            user_photos[user_id] = {"source": img}
            os.remove(input_path)
            return await update.message.reply_text("Лицо получено! Теперь пришли видео или GIF.")
        else:
            os.remove(input_path)
            return await update.message.reply_text(
                "Сначала нужно прислать *фото*, а потом — видео или GIF для свопа."
            )

    await update.message.reply_text("Начинаю обработку…")
    progress_message = await update.message.reply_text("Обработка: [░░░░░░░░░░] 0%")

    # fd, input_path = tempfile.mkstemp(suffix=f".{file_ext}")
    # with os.fdopen(fd, "wb") as f:
    #     f.write(input_path)

    # Имя выходного файла в temp
    output_path = input_path.replace(f".{file_ext}",
                                     "_swapped.mp4" if file_ext in ["mp4", "avi", "mov"] else f"_swapped.{file_ext}")

    try:
        if file_ext in ["jpg", "jpeg", "png"]:
            result = process_image(source_pil, input_path)
            result.save(output_path)
            await progress_message.delete()
            with open(output_path, "rb") as f:
                img_bytes = BytesIO(f.read())
                img_bytes.name = "result.jpg"  # Telegram требует имя
                await update.message.reply_photo(photo=img_bytes)

        elif file_ext == "gif":
            # 1) всегда конвертим как видео + GIF, чтобы получить корректную анимацию
            await edit_text_safely(progress_message, "Обрабатываю GIF как видео…")
            mp4_input = input_path.replace(".gif", ".mp4")
            os.replace(input_path, mp4_input)

            # 2) Генерируем свап-видео
            video_out = mp4_input.replace(".mp4", "_swapped.mp4")
            await process_video(source_pil, mp4_input, video_out, progress_message, user_id)

            # 3) Конвертируем получившийся MP4 → GIF
            await edit_text_safely(progress_message, "Видео готово, генерирую GIF…")
            gif_out = mp4_input.replace(".mp4", "_swapped.gif")
            video_to_gif(video_out, gif_out)

            # 4) Отправляем и чистим
            with open(gif_out, "rb") as f:
                buf = BytesIO(f.read())
                buf.name = "result.gif"
                await update.message.reply_animation(animation=buf)

            for p in (mp4_input, video_out, gif_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass

        elif file_ext in ["mp4", "avi", "mov"]:
            if USE_PARALEL_VIDEO:
                print(f"[DEBUG] Запускаю video swap для {input_path} -> {output_path}")
                await process_video_parallel(source_pil, input_path, output_path, progress_message, user_id)
            else:
                await process_video(source_pil, input_path, output_path, progress_message, user_id)

            final_path = None
            success = False

            if SAVE_OUTPUTS:
                save_dir = Path("./saved")
                save_dir.mkdir(exist_ok=True)

                timestamp = int(time.time())
                raw_name = f"user{user_id}_{timestamp}_raw.mp4"
                final_name = f"user{user_id}_{timestamp}_final.mp4"
                raw_path = save_dir / raw_name
                final_path = save_dir / final_name

                shutil.copyfile(output_path, raw_path)
                print(f"[DEBUG] Сырое видео сохранено: {raw_path}")

                success = fix_video_with_ffmpeg(str(raw_path), str(final_path))
            else:
                print(f"[DEBUG] Сохранение отключено")
                # просто временный файл для финального mp4
                final_path = output_path.replace(".mp4", "_final.mp4")
                print("Пути:", output_path, final_path)
                success = fix_video_with_ffmpeg(output_path, final_path)

            if success and os.path.getsize(final_path) > 100_000:
                print(f"[DEBUG] Финальное видео сохранено: {final_path}")

                video_size = os.path.getsize(final_path)
                if video_size >= 49_000_000:
                    await update.message.reply_text("⚠️ Видео слишком большое для Telegram-плеера, отправляю как файл.")
                    await update.message.reply_document(document=InputFile(str(final_path)))
                else:
                    with open(final_path, "rb") as f:
                        video_bytes = BytesIO(f.read())
                        video_bytes.name = "video.mp4"
                        await update.message.reply_video(video=video_bytes)
            else:
                await edit_text_safely(progress_message, "⚠️ Не удалось сгенерировать корректное видео.")

        else:
            await edit_text_safely(progress_message, "Формат не поддерживается.")

    except Exception as e:
        print("Ошибка:", e)
        await edit_text_safely(progress_message, "Произошла ошибка при обработке.")

    user_photos.pop(user_id, None)
    user_tasks.pop(user_id, None)
    try:
        os.remove(input_path)
    except OSError:
        pass
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass


async def cancel_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_tasks:
        user_tasks[user_id]["cancel"] = True
        await update.message.reply_text("⛔ Генерация отменяется…")
    else:
        await update.message.reply_text("Нет активной задачи для отмены.")


def main():
    load_dotenv()
    TOKEN = os.getenv("TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(
        filters.PHOTO | filters.VIDEO | filters.ANIMATION | filters.Document.IMAGE | filters.Document.VIDEO,
        handle_file
    ))
    app.add_handler(CommandHandler("cancel", cancel_generation))
    print("device = ", swapper.device)
    print("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
