# عقده ای (Oghdeii)

**میانبرهای لپ‌تاپ با ضربه به بدنه و فرمان صوتی**

Oghdeii (عقده ای) یک برنامه‌ی دسکتاپ برای اجرای میانبرها با یک، دو یا سه ضربه به بدنه‌ی لپ‌تاپ است. در ویندوز، فرمان‌های صوتی انگلیسی و بررسی اختیاری intent با V1M هم پشتیبانی می‌شوند.

## الهام

ایده‌ی کنترل لپ‌تاپ با ضربه از [MacTap](https://github.com/jaskirat1616/mactap-app) الهام گرفته شده است. عقده ای پروژه‌ای مستقل با پیاده‌سازی و امکانات متناسب با Windows و لپ‌تاپ‌های دیگر است؛ وابسته به MacTap نیست.

جزئیات ارجاع در [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) آمده است.

## قابلیت‌ها

- نگاشت یک، دو و سه ضربه و ضربه‌ی سمت چپ/راست به میانبرهای دلخواه.
- فیلتر صوت محیط و تایپ برای کاهش ضربه‌های اشتباه.
- فرمان صوتی آفلاین با VAD کوتاه و `faster-whisper` (`distil-small.en`)؛ متن، تطبیق فرمان و گزینه‌های جایگزین به V1M داده می‌شوند.
- طبقه‌بندی اختیاری V1M برای intent، انتخاب action و ارزیابی ریسک. با فعال‌بودن V1M، فرمان بدون پاسخ معتبر سرویس اجرا نمی‌شود.
- تنظیم میکروفن، حساسیت، میانبرها و رفتار برنامه از رابط گرافیکی.

## نیازمندی‌ها

- Python 3.11 تا 3.13
- ویندوز برای Voice Mode و دسترسی میکروفن
- میکروفن و دستگاه صوتی سازگار با `sounddevice`

## اجرای برنامه از سورس

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

در اولین اجرای Voice Mode، مدل `distil-small.en` از Hugging Face دریافت و در cache محلی ذخیره می‌شود. این مدل با `faster-whisper` و `int8` روی CPU اجرا می‌شود؛ فایل صوتی از دستگاه خارج نمی‌شود.

برای اجرای CLI:

```powershell
.\.venv\Scripts\python.exe main.py --cli
```

برای ساخت نسخه‌ی پوشه‌ای ویندوز با PyInstaller، پس از نصب نیازمندی‌ها اجرا کن:

```powershell
.\build_exe.bat
```

این دستور خروجی را در `dist/` می‌سازد؛ خروجی build در Git قرار نمی‌گیرد.

برای استفاده از طبقه‌بندی V1M، آن را از تنظیمات برنامه فعال و کلید خودت را در UI وارد کن یا متغیر محیطی `V1M_API_KEY` را تنظیم کن. کلید API را داخل سورس یا Git قرار نده. وقتی V1M روشن است، transcript و گزینه‌های جایگزین برای سرویس فرستاده می‌شوند.

## آزمون‌ها

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

تست‌های سخت‌افزاری جدا هستند و برای اجرا میانبر واقعی به کار نمی‌برند:

```powershell
.\.venv\Scripts\python.exe tap_validate.py
.\.venv\Scripts\python.exe voice_validate.py
```

## حریم خصوصی

حالت ضربه روی دستگاه پردازش می‌شود. اگر V1M فعال باشد، متن تشخیص‌داده‌شده، گزینه‌های جایگزین و زمینه‌ی لازم برای طبقه‌بندی به API ارسال می‌شوند. کلید API در فایل تنظیمات محلی کاربر یا محیط سیستم نگه‌داری شود.

## ساختار مخزن

مخزن شامل سورس، تست‌ها و آیکون‌هاست. پوشه‌ی محیط مجازی، خروجی‌های build، مدل‌های cache‌شده، گزارش‌های محلی، cacheها و اسکریپت‌های نصب/لانچر در Git قرار نمی‌گیرند. برای نگه‌داشتن helper قدیمی Windows Speech می‌توان `voice_backend` را در تنظیمات محلی روی `windows` گذاشت؛ مقدار پیش‌فرض `whisper` است.

## Development and package installation

Install the project and its development checks in a virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\oghdeii.exe --help
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy main.py gui.py action_executor.py audio_engine.py config_manager.py tap_detector.py voice_detector.py whisper_voice_detector.py oghdeii
```

The `oghdeii` command supports `--cli` and `--minimized`. The package includes
the application modules, icons, and Windows speech helper. The `cloud` extra is
a compatibility alias and still installs the desktop dependencies, including Qt.

Automated tests mock microphones, cloud requests, and operating-system shortcuts.
The GUI regression tests use Qt's offscreen platform. Real microphone accuracy,
desktop shortcuts, and tray behavior require an interactive hardware check.
Voice calibration currently uses the legacy Windows Speech helper; it does not
calibrate the Whisper model. Calibration launchers require a source or Python
package installation on Windows and are unavailable in the standalone EXE.

See [PROJECT_REVIEW.md](PROJECT_REVIEW.md) for the review, fixes, and remaining improvements.
