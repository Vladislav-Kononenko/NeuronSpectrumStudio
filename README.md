# NeuronSpectrum

Настольное приложение для работы с EEG через LSL: поиск потоков, просмотр metadata, запись в `npy`, offline-анализ корреляций, live-корреляции и сравнение матриц.

## Что есть в проекте

- CLI-скрипты в `scripts/` сохранены и продолжают работать.
- GUI-приложение на `PySide6` запускается через `main.py`.
- Основные вычисления переиспользуют существующую логику проекта, а не переписаны с нуля.

## Запуск из исходников

1. Создайте и активируйте виртуальное окружение.
2. Установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

3. Запустите GUI:

```powershell
python main.py
```

## Полезные CLI-команды

```powershell
python scripts/lsl_browser.py
python scripts/lsl_inspect.py --type EEG --out data/sessions/stream_meta
python scripts/lsl_record.py --type EEG --duration 60 --out data/sessions/session_001
python scripts/check_recording.py --samples data/sessions/session_001_samples.npy --timestamps data/sessions/session_001_timestamps.npy --sfreq 500 --window-seconds 2 --start-seconds 0 --out data/sessions/session_001_check --save-csv --save-png
python scripts/live_correlations.py --type EEG --sfreq 500 --window-seconds 2 --step-seconds 0.25 --out data/sessions/live_001 --save-csv --save-png
python scripts/compare_matrices.py --reference-prefix data/sessions/session_001_check --candidate-prefix data/sessions/live_001 --out data/sessions/offline_vs_live --save-csv --save-png
```

## Сборка Windows exe

Рекомендуемый вариант: `onefolder` через `PyInstaller`, потому что для `PySide6`, `matplotlib` и `pylsl` это стабильнее, чем `onefile`.

### PowerShell

```powershell
.\tools\build\build.ps1
```

### CMD

```cmd
tools\build\build.bat
```

### Ручная сборка

```powershell
python -m PyInstaller --clean packaging\pyinstaller\NeuronSpectrumGUI.spec
```

## Сборка установщика

После сборки папки `dist/NeuronSpectrumGUI` можно собрать обычный Windows-установщик на `Inno Setup`.

### PowerShell

```powershell
.\tools\build\build-installer.ps1
```

### CMD

```cmd
tools\build\build-installer.bat
```

### Ручная сборка через Inno Setup

```powershell
ISCC.exe packaging\inno\installer.iss
```

## Где лежит GUI и сборка

- Основной GUI-скрипт: `main.py`
- GUI-модули: `neurospectrum_gui/`
- PyInstaller spec: `packaging/pyinstaller/NeuronSpectrumGUI.spec`
- Готовая сборка: `dist/NeuronSpectrumGUI/NeuronSpectrumGUI.exe`
- Скрипт установщика: `packaging/inno/installer.iss`
- Готовый установщик: `installer_dist/NeuronSpectrumGUI_Setup.exe`

## Замечания

- Для записи и live-анализа нужен реальный LSL-поток EEG.
- В текущей среде offline-анализ и сравнение проверены на данных из `data/sessions`.
- Если `onefile` понадобится отдельно, можно добавить второй spec, но по умолчанию оставлен более надёжный `onefolder`.
