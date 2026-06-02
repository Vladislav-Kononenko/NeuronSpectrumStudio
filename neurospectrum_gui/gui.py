from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import services
from .workers import CompareWorker, LiveCorrelationWorker, OfflineAnalysisWorker, RecordingWorker


class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(5, 4))
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def draw_matrix(self, matrix: np.ndarray, labels: list[str], title: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        self.figure.tight_layout()
        self.draw()


class MatrixViewer(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labels: list[str] = []
        self.matrices: dict[str, np.ndarray] = {}

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel(title))
        self.matrix_selector = QComboBox()
        self.matrix_selector.currentIndexChanged.connect(self.refresh_view)
        header.addWidget(self.matrix_selector, 1)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas = HeatmapCanvas(self)
        splitter.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        splitter.addWidget(self.table)
        splitter.setSizes([360, 240])
        layout.addWidget(splitter)

    def set_matrices(self, matrices: dict[str, np.ndarray], labels: list[str]) -> None:
        self.matrices = matrices
        self.labels = labels
        self.matrix_selector.blockSignals(True)
        self.matrix_selector.clear()
        self.matrix_selector.addItems(list(matrices.keys()))
        self.matrix_selector.blockSignals(False)
        self.refresh_view()

    def refresh_view(self) -> None:
        key = self.matrix_selector.currentText()
        if not key or key not in self.matrices:
            self.table.clear()
            return

        matrix = self.matrices[key]
        labels = self.labels if len(self.labels) == matrix.shape[0] else [f"ch{i}" for i in range(matrix.shape[0])]
        self.canvas.draw_matrix(matrix, labels, key)

        self.table.setRowCount(matrix.shape[0])
        self.table.setColumnCount(matrix.shape[1])
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(f"{matrix[i, j]:.4f}"))
        self.table.resizeColumnsToContents()


class MetadataDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Метаданные LSL-потока")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        layout.addWidget(summary)

        tabs = QTabWidget()
        json_view = QPlainTextEdit()
        json_view.setReadOnly(True)
        json_view.setPlainText(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
        xml_view = QPlainTextEdit()
        xml_view.setReadOnly(True)
        xml_view.setPlainText(payload["xml_text"])
        tabs.addTab(json_view, "JSON")
        tabs.addTab(xml_view, "XML")
        layout.addWidget(tabs)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("NeuronSpectrum", "NeuronSpectrumGUI")
        self.available_streams: list[services.StreamDescriptor] = []
        self.record_worker: RecordingWorker | None = None
        self.offline_worker: OfflineAnalysisWorker | None = None
        self.compare_worker: CompareWorker | None = None
        self.live_worker: LiveCorrelationWorker | None = None
        self.last_offline_result: dict[str, Any] | None = None
        self.last_compare_result: dict[str, Any] | None = None
        self.last_live_result: dict[str, Any] | None = None

        self.setWindowTitle("NeuronSpectrum EEG/LSL Studio")
        self.resize(1400, 980)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.lsl_tab = self._build_lsl_tab()
        self.record_tab = self._build_record_tab()
        self.offline_tab = self._build_offline_tab()
        self.live_tab = self._build_live_tab()
        self.compare_tab = self._build_compare_tab()
        self.log_tab = self._build_log_tab()
        self.settings_tab = self._build_settings_tab()
        self.about_tab = self._build_about_tab()

        self.tabs.addTab(self.lsl_tab, "Потоки LSL")
        self.tabs.addTab(self.record_tab, "Запись")
        self.tabs.addTab(self.offline_tab, "Проверка записи")
        self.tabs.addTab(self.live_tab, "Live-корреляции")
        self.tabs.addTab(self.compare_tab, "Сравнение матриц")
        self.tabs.addTab(self.log_tab, "Журнал")
        self.tabs.addTab(self.settings_tab, "Настройки")
        self.tabs.addTab(self.about_tab, "О проекте")

        self.apply_saved_settings()
        self.refresh_streams()

    def _build_lsl_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self.lsl_timeout = QDoubleSpinBox()
        self.lsl_timeout.setRange(0.1, 20.0)
        self.lsl_timeout.setSingleStep(0.5)
        self.lsl_timeout.setValue(2.0)
        top.addWidget(QLabel("Таймаут поиска, сек"))
        top.addWidget(self.lsl_timeout)
        self.lsl_refresh_button = QPushButton("Найти потоки")
        self.lsl_refresh_button.clicked.connect(self.refresh_streams)
        top.addWidget(self.lsl_refresh_button)
        self.lsl_metadata_button = QPushButton("Показать metadata")
        self.lsl_metadata_button.clicked.connect(self.show_selected_metadata)
        top.addWidget(self.lsl_metadata_button)
        self.lsl_save_meta_button = QPushButton("Сохранить metadata")
        self.lsl_save_meta_button.clicked.connect(self.save_selected_metadata)
        top.addWidget(self.lsl_save_meta_button)
        top.addStretch(1)
        layout.addLayout(top)

        self.lsl_table = QTableWidget(0, 6)
        self.lsl_table.setHorizontalHeaderLabels(["Имя", "Type", "Source ID", "Частота", "Каналы", "Format"])
        self.lsl_table.itemSelectionChanged.connect(self.update_lsl_details)
        layout.addWidget(self.lsl_table)

        self.lsl_details = QTextEdit()
        self.lsl_details.setReadOnly(True)
        layout.addWidget(self.lsl_details)
        return widget

    def _build_record_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QGridLayout()
        self.record_stream_combo = QComboBox()
        self.record_prefix = QLineEdit()
        self.record_browse = QPushButton("Выбрать путь")
        self.record_browse.clicked.connect(lambda: self.pick_prefix(self.record_prefix))
        self.record_timeout = QDoubleSpinBox()
        self.record_timeout.setRange(0.1, 20.0)
        self.record_timeout.setValue(2.0)
        self.record_pull_timeout = QDoubleSpinBox()
        self.record_pull_timeout.setRange(0.1, 5.0)
        self.record_pull_timeout.setValue(0.5)
        self.record_max_duration = QDoubleSpinBox()
        self.record_max_duration.setRange(0.0, 86400.0)
        self.record_max_duration.setValue(0.0)
        self.record_max_duration.setSuffix(" сек")
        form.addWidget(QLabel("Поток"), 0, 0)
        form.addWidget(self.record_stream_combo, 0, 1, 1, 2)
        form.addWidget(QLabel("Префикс сохранения"), 1, 0)
        form.addWidget(self.record_prefix, 1, 1)
        form.addWidget(self.record_browse, 1, 2)
        form.addWidget(QLabel("Таймаут поиска"), 2, 0)
        form.addWidget(self.record_timeout, 2, 1)
        form.addWidget(QLabel("Таймаут pull_sample"), 2, 2)
        form.addWidget(self.record_pull_timeout, 2, 3)
        form.addWidget(QLabel("Макс. длительность"), 3, 0)
        form.addWidget(self.record_max_duration, 3, 1)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.record_start_button = QPushButton("Запустить запись")
        self.record_start_button.clicked.connect(self.start_recording)
        self.record_stop_button = QPushButton("Остановить запись")
        self.record_stop_button.clicked.connect(self.stop_recording)
        self.record_stop_button.setEnabled(False)
        self.record_open_folder_button = QPushButton("Открыть папку результатов")
        self.record_open_folder_button.clicked.connect(lambda: self.open_result_folder(self.record_prefix.text()))
        buttons.addWidget(self.record_start_button)
        buttons.addWidget(self.record_stop_button)
        buttons.addWidget(self.record_open_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 1)
        self.record_progress.setValue(0)
        layout.addWidget(self.record_progress)

        self.record_stats = QTextEdit()
        self.record_stats.setReadOnly(True)
        layout.addWidget(self.record_stats)
        return widget

    def _build_offline_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QGridLayout()
        self.offline_samples = QLineEdit()
        self.offline_samples_btn = QPushButton("Файл samples")
        self.offline_samples_btn.clicked.connect(lambda: self.pick_file(self.offline_samples))
        self.offline_timestamps = QLineEdit()
        self.offline_timestamps_btn = QPushButton("Файл timestamps")
        self.offline_timestamps_btn.clicked.connect(lambda: self.pick_file(self.offline_timestamps))
        self.offline_meta = QLineEdit()
        self.offline_meta_btn = QPushButton("Meta JSON")
        self.offline_meta_btn.clicked.connect(lambda: self.pick_file(self.offline_meta))
        self.offline_out = QLineEdit()
        self.offline_out_btn = QPushButton("Префикс вывода")
        self.offline_out_btn.clicked.connect(lambda: self.pick_prefix(self.offline_out))
        self.offline_sfreq = QDoubleSpinBox()
        self.offline_sfreq.setRange(1.0, 100000.0)
        self.offline_sfreq.setValue(500.0)
        self.offline_window = QDoubleSpinBox()
        self.offline_window.setRange(0.1, 120.0)
        self.offline_window.setValue(2.0)
        self.offline_start = QDoubleSpinBox()
        self.offline_start.setRange(0.0, 86400.0)
        self.offline_start.setValue(0.0)
        self.offline_save_csv = QCheckBox("Сохранять CSV")
        self.offline_save_png = QCheckBox("Сохранять PNG")
        self.offline_save_rel = QCheckBox("Сохранять диагностические матрицы")
        self.offline_save_rel.setChecked(True)

        form.addWidget(QLabel("samples.npy"), 0, 0)
        form.addWidget(self.offline_samples, 0, 1)
        form.addWidget(self.offline_samples_btn, 0, 2)
        form.addWidget(QLabel("timestamps.npy"), 1, 0)
        form.addWidget(self.offline_timestamps, 1, 1)
        form.addWidget(self.offline_timestamps_btn, 1, 2)
        form.addWidget(QLabel("meta.json"), 2, 0)
        form.addWidget(self.offline_meta, 2, 1)
        form.addWidget(self.offline_meta_btn, 2, 2)
        form.addWidget(QLabel("Префикс вывода"), 3, 0)
        form.addWidget(self.offline_out, 3, 1)
        form.addWidget(self.offline_out_btn, 3, 2)
        form.addWidget(QLabel("Частота, Гц"), 4, 0)
        form.addWidget(self.offline_sfreq, 4, 1)
        form.addWidget(QLabel("Окно, сек"), 4, 2)
        form.addWidget(self.offline_window, 4, 3)
        form.addWidget(QLabel("Смещение окна, сек"), 5, 0)
        form.addWidget(self.offline_start, 5, 1)
        form.addWidget(self.offline_save_csv, 5, 2)
        form.addWidget(self.offline_save_png, 5, 3)
        form.addWidget(self.offline_save_rel, 6, 2, 1, 2)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.offline_run_button = QPushButton("Запустить offline-анализ")
        self.offline_run_button.clicked.connect(self.run_offline_analysis)
        self.offline_open_folder_button = QPushButton("Открыть папку результатов")
        self.offline_open_folder_button.clicked.connect(lambda: self.open_result_folder(self.offline_out.text()))
        buttons.addWidget(self.offline_run_button)
        buttons.addWidget(self.offline_open_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.offline_progress = QProgressBar()
        self.offline_progress.setRange(0, 1)
        self.offline_progress.setValue(0)
        layout.addWidget(self.offline_progress)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.offline_summary = QTextEdit()
        self.offline_summary.setReadOnly(True)
        splitter.addWidget(self.offline_summary)
        self.offline_viewer = MatrixViewer("Матрицы offline-анализа")
        splitter.addWidget(self.offline_viewer)
        splitter.setSizes([220, 560])
        layout.addWidget(splitter)
        return widget

    def _build_live_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QGridLayout()
        self.live_stream_combo = QComboBox()
        self.live_meta = QLineEdit()
        self.live_meta_btn = QPushButton("Meta JSON")
        self.live_meta_btn.clicked.connect(lambda: self.pick_file(self.live_meta))
        self.live_out = QLineEdit()
        self.live_out_btn = QPushButton("Префикс сохранения")
        self.live_out_btn.clicked.connect(lambda: self.pick_prefix(self.live_out))
        self.live_timeout = QDoubleSpinBox()
        self.live_timeout.setRange(0.1, 20.0)
        self.live_timeout.setValue(2.0)
        self.live_pull_timeout = QDoubleSpinBox()
        self.live_pull_timeout.setRange(0.1, 5.0)
        self.live_pull_timeout.setValue(0.5)
        self.live_sfreq = QDoubleSpinBox()
        self.live_sfreq.setRange(0.0, 100000.0)
        self.live_sfreq.setValue(500.0)
        self.live_window = QDoubleSpinBox()
        self.live_window.setRange(0.1, 120.0)
        self.live_window.setValue(2.0)
        self.live_step = QDoubleSpinBox()
        self.live_step.setRange(0.01, 60.0)
        self.live_step.setValue(0.25)
        self.live_save_csv = QCheckBox("Сохранять CSV")
        self.live_save_png = QCheckBox("Сохранять PNG")

        form.addWidget(QLabel("Поток"), 0, 0)
        form.addWidget(self.live_stream_combo, 0, 1, 1, 2)
        form.addWidget(QLabel("Meta JSON"), 1, 0)
        form.addWidget(self.live_meta, 1, 1)
        form.addWidget(self.live_meta_btn, 1, 2)
        form.addWidget(QLabel("Префикс сохранения"), 2, 0)
        form.addWidget(self.live_out, 2, 1)
        form.addWidget(self.live_out_btn, 2, 2)
        form.addWidget(QLabel("Частота, Гц"), 3, 0)
        form.addWidget(self.live_sfreq, 3, 1)
        form.addWidget(QLabel("Окно, сек"), 3, 2)
        form.addWidget(self.live_window, 3, 3)
        form.addWidget(QLabel("Шаг обновления, сек"), 4, 0)
        form.addWidget(self.live_step, 4, 1)
        form.addWidget(QLabel("Таймаут поиска"), 4, 2)
        form.addWidget(self.live_timeout, 4, 3)
        form.addWidget(QLabel("Таймаут pull_sample"), 5, 0)
        form.addWidget(self.live_pull_timeout, 5, 1)
        form.addWidget(self.live_save_csv, 5, 2)
        form.addWidget(self.live_save_png, 5, 3)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.live_start_button = QPushButton("Запустить realtime-анализ")
        self.live_start_button.clicked.connect(self.start_live_analysis)
        self.live_stop_button = QPushButton("Остановить realtime-анализ")
        self.live_stop_button.clicked.connect(self.stop_live_analysis)
        self.live_stop_button.setEnabled(False)
        self.live_save_button = QPushButton("Сохранить последние матрицы")
        self.live_save_button.clicked.connect(self.save_live_snapshot)
        self.live_open_folder_button = QPushButton("Открыть папку результатов")
        self.live_open_folder_button.clicked.connect(lambda: self.open_result_folder(self.live_out.text()))
        buttons.addWidget(self.live_start_button)
        buttons.addWidget(self.live_stop_button)
        buttons.addWidget(self.live_save_button)
        buttons.addWidget(self.live_open_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.live_progress = QProgressBar()
        self.live_progress.setRange(0, 1)
        self.live_progress.setValue(0)
        layout.addWidget(self.live_progress)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.live_summary = QTextEdit()
        self.live_summary.setReadOnly(True)
        splitter.addWidget(self.live_summary)
        self.live_viewer = MatrixViewer("Матрицы live-анализа")
        splitter.addWidget(self.live_viewer)
        splitter.setSizes([220, 560])
        layout.addWidget(splitter)
        return widget

    def _build_compare_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QGridLayout()
        self.compare_ref = QLineEdit()
        self.compare_ref_btn = QPushButton("Эталонный префикс")
        self.compare_ref_btn.clicked.connect(lambda: self.pick_prefix(self.compare_ref))
        self.compare_cand = QLineEdit()
        self.compare_cand_btn = QPushButton("Сравниваемый префикс")
        self.compare_cand_btn.clicked.connect(lambda: self.pick_prefix(self.compare_cand))
        self.compare_meta = QLineEdit()
        self.compare_meta_btn = QPushButton("Meta JSON")
        self.compare_meta_btn.clicked.connect(lambda: self.pick_file(self.compare_meta))
        self.compare_out = QLineEdit()
        self.compare_out_btn = QPushButton("Префикс вывода")
        self.compare_out_btn.clicked.connect(lambda: self.pick_prefix(self.compare_out))
        self.compare_save_csv = QCheckBox("Сохранять CSV")
        self.compare_save_png = QCheckBox("Сохранять PNG")

        form.addWidget(QLabel("Эталонный префикс"), 0, 0)
        form.addWidget(self.compare_ref, 0, 1)
        form.addWidget(self.compare_ref_btn, 0, 2)
        form.addWidget(QLabel("Сравниваемый префикс"), 1, 0)
        form.addWidget(self.compare_cand, 1, 1)
        form.addWidget(self.compare_cand_btn, 1, 2)
        form.addWidget(QLabel("Meta JSON"), 2, 0)
        form.addWidget(self.compare_meta, 2, 1)
        form.addWidget(self.compare_meta_btn, 2, 2)
        form.addWidget(QLabel("Префикс вывода"), 3, 0)
        form.addWidget(self.compare_out, 3, 1)
        form.addWidget(self.compare_out_btn, 3, 2)
        form.addWidget(self.compare_save_csv, 4, 1)
        form.addWidget(self.compare_save_png, 4, 2)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.compare_run_button = QPushButton("Сравнить матрицы")
        self.compare_run_button.clicked.connect(self.run_compare)
        self.compare_open_folder_button = QPushButton("Открыть папку результатов")
        self.compare_open_folder_button.clicked.connect(lambda: self.open_result_folder(self.compare_out.text()))
        buttons.addWidget(self.compare_run_button)
        buttons.addWidget(self.compare_open_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.compare_progress = QProgressBar()
        self.compare_progress.setRange(0, 1)
        self.compare_progress.setValue(0)
        layout.addWidget(self.compare_progress)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.compare_summary = QTextEdit()
        self.compare_summary.setReadOnly(True)
        splitter.addWidget(self.compare_summary)
        self.compare_viewer = MatrixViewer("Матрицы различий")
        splitter.addWidget(self.compare_viewer)
        splitter.setSizes([220, 560])
        layout.addWidget(splitter)
        return widget

    def _build_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return widget

    def _build_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.settings_data_dir = QLineEdit("data/sessions")
        self.settings_timeout = QDoubleSpinBox()
        self.settings_timeout.setRange(0.1, 20.0)
        self.settings_timeout.setValue(2.0)
        self.settings_sfreq = QDoubleSpinBox()
        self.settings_sfreq.setRange(1.0, 100000.0)
        self.settings_sfreq.setValue(500.0)
        self.settings_window = QDoubleSpinBox()
        self.settings_window.setRange(0.1, 120.0)
        self.settings_window.setValue(2.0)
        self.settings_step = QDoubleSpinBox()
        self.settings_step.setRange(0.01, 60.0)
        self.settings_step.setValue(0.25)
        self.settings_pull_timeout = QDoubleSpinBox()
        self.settings_pull_timeout.setRange(0.1, 5.0)
        self.settings_pull_timeout.setValue(0.5)

        form.addRow("Папка результатов по умолчанию", self.settings_data_dir)
        form.addRow("Таймаут поиска LSL, сек", self.settings_timeout)
        form.addRow("Частота по умолчанию, Гц", self.settings_sfreq)
        form.addRow("Окно по умолчанию, сек", self.settings_window)
        form.addRow("Шаг обновления по умолчанию, сек", self.settings_step)
        form.addRow("Таймаут pull_sample, сек", self.settings_pull_timeout)
        layout.addLayout(form)

        save_button = QPushButton("Сохранить настройки")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        layout.addStretch(1)
        return widget

    def _build_about_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "NeuronSpectrum EEG/LSL Studio\n\n"
            "GUI объединяет существующие функции проекта:\n"
            "- поиск и просмотр LSL-потоков;\n"
            "- запись EEG в samples/timestamps/meta;\n"
            "- offline-анализ Pearson/Spearman;\n"
            "- live-корреляции по скользящему окну;\n"
            "- сравнение сохранённых матриц;\n"
            "- экспорт NPY/CSV/PNG/JSON.\n"
        )
        layout.addWidget(text)
        return widget

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def handle_error(self, title: str, message: str) -> None:
        self.log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def pick_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", target.text() or self.settings_data_dir.text(), "NumPy/JSON (*.npy *.json);;Все файлы (*.*)")
        if path:
            target.setText(path)

    def pick_prefix(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Выбор префикса", target.text() or self.settings_data_dir.text(), "Все файлы (*.*)")
        if path:
            target.setText(path)

    def open_result_folder(self, raw_path: str) -> None:
        if not raw_path:
            self.handle_error("Открытие папки", "Путь не задан.")
            return
        path = Path(raw_path)
        folder = path if path.is_dir() else path.parent
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # type: ignore[attr-defined]

    def apply_saved_settings(self) -> None:
        self.settings_data_dir.setText(self.settings.value("paths/data_dir", "data/sessions"))
        self.settings_timeout.setValue(float(self.settings.value("defaults/timeout", 2.0)))
        self.settings_sfreq.setValue(float(self.settings.value("defaults/sfreq", 500.0)))
        self.settings_window.setValue(float(self.settings.value("defaults/window", 2.0)))
        self.settings_step.setValue(float(self.settings.value("defaults/step", 0.25)))
        self.settings_pull_timeout.setValue(float(self.settings.value("defaults/pull_timeout", 0.5)))

        self.lsl_timeout.setValue(self.settings_timeout.value())
        self.record_timeout.setValue(self.settings_timeout.value())
        self.record_pull_timeout.setValue(self.settings_pull_timeout.value())
        self.live_timeout.setValue(self.settings_timeout.value())
        self.live_pull_timeout.setValue(self.settings_pull_timeout.value())
        self.offline_sfreq.setValue(self.settings_sfreq.value())
        self.live_sfreq.setValue(self.settings_sfreq.value())
        self.offline_window.setValue(self.settings_window.value())
        self.live_window.setValue(self.settings_window.value())
        self.live_step.setValue(self.settings_step.value())

        default_dir = self.settings_data_dir.text()
        self.record_prefix.setText(str(Path(default_dir) / "session_gui"))
        self.offline_out.setText(str(Path(default_dir) / "offline_gui"))
        self.live_out.setText(str(Path(default_dir) / "live_gui"))
        self.compare_out.setText(str(Path(default_dir) / "compare_gui"))

    def save_settings(self) -> None:
        self.settings.setValue("paths/data_dir", self.settings_data_dir.text())
        self.settings.setValue("defaults/timeout", self.settings_timeout.value())
        self.settings.setValue("defaults/sfreq", self.settings_sfreq.value())
        self.settings.setValue("defaults/window", self.settings_window.value())
        self.settings.setValue("defaults/step", self.settings_step.value())
        self.settings.setValue("defaults/pull_timeout", self.settings_pull_timeout.value())
        self.apply_saved_settings()
        self.log("Настройки сохранены.")

    def refresh_streams(self) -> None:
        try:
            timeout = self.lsl_timeout.value()
            self.available_streams = services.discover_lsl_streams(timeout)
            self.lsl_table.setRowCount(len(self.available_streams))
            for row, stream in enumerate(self.available_streams):
                values = [
                    stream.name,
                    stream.type,
                    stream.source_id,
                    f"{stream.nominal_srate:g}",
                    str(stream.channel_count),
                    stream.channel_format,
                ]
                for col, value in enumerate(values):
                    self.lsl_table.setItem(row, col, QTableWidgetItem(value))
            self.lsl_table.resizeColumnsToContents()

            self.record_stream_combo.clear()
            self.live_stream_combo.clear()
            for stream in self.available_streams:
                label = f"{stream.name} | {stream.type} | {stream.channel_count} каналов | {stream.nominal_srate:g} Гц"
                self.record_stream_combo.addItem(label)
                self.live_stream_combo.addItem(label)

            self.update_lsl_details()
            self.log(f"Найдено LSL-потоков: {len(self.available_streams)}")
        except Exception as exc:
            self.handle_error("Поиск LSL-потоков", str(exc))

    def selected_lsl_stream(self) -> services.StreamDescriptor | None:
        row = self.lsl_table.currentRow()
        if 0 <= row < len(self.available_streams):
            return self.available_streams[row]
        return self.available_streams[0] if self.available_streams else None

    def selected_record_stream(self) -> services.StreamDescriptor | None:
        index = self.record_stream_combo.currentIndex()
        if 0 <= index < len(self.available_streams):
            return self.available_streams[index]
        return None

    def selected_live_stream(self) -> services.StreamDescriptor | None:
        index = self.live_stream_combo.currentIndex()
        if 0 <= index < len(self.available_streams):
            return self.available_streams[index]
        return None

    def update_lsl_details(self) -> None:
        stream = self.selected_lsl_stream()
        if not stream:
            self.lsl_details.setPlainText("LSL-потоки не найдены.")
            return
        self.lsl_details.setPlainText(
            json.dumps(
                {
                    "name": stream.name,
                    "type": stream.type,
                    "source_id": stream.source_id,
                    "nominal_srate": stream.nominal_srate,
                    "channel_count": stream.channel_count,
                    "channel_format": stream.channel_format,
                    "uid": stream.uid,
                    "hostname": stream.hostname,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def show_selected_metadata(self) -> None:
        stream = self.selected_lsl_stream()
        if not stream:
            self.handle_error("Метаданные LSL", "Сначала найдите и выберите поток.")
            return
        try:
            payload = services.inspect_lsl_stream(stream, self.lsl_timeout.value())
            dialog = MetadataDialog(payload, self)
            dialog.exec()
        except Exception as exc:
            self.handle_error("Метаданные LSL", str(exc))

    def save_selected_metadata(self) -> None:
        stream = self.selected_lsl_stream()
        if not stream:
            self.handle_error("Сохранение metadata", "Сначала выберите поток.")
            return
        prefix, _ = QFileDialog.getSaveFileName(self, "Куда сохранить metadata", str(Path(self.settings_data_dir.text()) / "stream_meta"), "Все файлы (*.*)")
        if not prefix:
            return
        try:
            paths = services.save_stream_metadata(stream, prefix, self.lsl_timeout.value())
            self.log(f"Сохранены metadata: {paths['xml']}, {paths['json']}")
        except Exception as exc:
            self.handle_error("Сохранение metadata", str(exc))

    def start_recording(self) -> None:
        descriptor = self.selected_record_stream()
        if not descriptor:
            self.handle_error("Запись", "LSL-поток не выбран.")
            return
        if not self.record_prefix.text().strip():
            self.handle_error("Запись", "Не задан префикс сохранения.")
            return

        self.record_worker = RecordingWorker(
            descriptor=descriptor,
            out_prefix=self.record_prefix.text().strip(),
            discover_timeout=self.record_timeout.value(),
            pull_timeout=self.record_pull_timeout.value(),
            max_duration=self.record_max_duration.value(),
        )
        self.record_worker.log.connect(self.log)
        self.record_worker.stats.connect(self.update_record_stats)
        self.record_worker.finished_ok.connect(self.recording_finished)
        self.record_worker.failed.connect(lambda msg: self.worker_failed("Запись", msg))
        self.record_worker.start()

        self.record_progress.setRange(0, 0)
        self.record_start_button.setEnabled(False)
        self.record_stop_button.setEnabled(True)
        self.log("Запуск записи EEG.")

    def stop_recording(self) -> None:
        if self.record_worker:
            self.record_worker.stop()
            self.log("Остановка записи запрошена.")

    def update_record_stats(self, stats: dict[str, Any]) -> None:
        self.record_stats.setPlainText(json.dumps(stats, ensure_ascii=False, indent=2))

    def recording_finished(self, payload: dict[str, Any]) -> None:
        self.record_progress.setRange(0, 1)
        self.record_progress.setValue(1)
        self.record_start_button.setEnabled(True)
        self.record_stop_button.setEnabled(False)
        self.record_stats.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.log(f"Запись завершена. Данные сохранены с префиксом {payload['out_prefix']}")

    def run_offline_analysis(self) -> None:
        if not self.offline_samples.text().strip() or not self.offline_timestamps.text().strip():
            self.handle_error("Offline-анализ", "Укажите файлы samples и timestamps.")
            return
        self.offline_worker = OfflineAnalysisWorker(
            samples_path=self.offline_samples.text().strip(),
            timestamps_path=self.offline_timestamps.text().strip(),
            sfreq=self.offline_sfreq.value(),
            window_seconds=self.offline_window.value(),
            start_seconds=self.offline_start.value(),
            out_prefix=self.offline_out.text().strip() or None,
            meta_json=self.offline_meta.text().strip() or None,
            save_csv=self.offline_save_csv.isChecked(),
            save_png=self.offline_save_png.isChecked(),
            save_relations=self.offline_save_rel.isChecked(),
        )
        self.offline_worker.log.connect(self.log)
        self.offline_worker.finished_ok.connect(self.offline_finished)
        self.offline_worker.failed.connect(lambda msg: self.worker_failed("Offline-анализ", msg))
        self.offline_worker.start()

        self.offline_progress.setRange(0, 0)
        self.offline_run_button.setEnabled(False)

    def offline_finished(self, result: dict[str, Any]) -> None:
        self.last_offline_result = result
        matrices = {
            "Pearson": result["pearson"],
            "Spearman": result["spearman"],
            "|Pearson - Spearman|": result["diff_abs"],
        }
        for name, matrix in result["relations"].items():
            matrices[f"diag::{name}"] = matrix
        self.offline_viewer.set_matrices(matrices, result["labels"])
        self.offline_summary.setPlainText(json.dumps(result["report"], ensure_ascii=False, indent=2))
        self.offline_progress.setRange(0, 1)
        self.offline_progress.setValue(1)
        self.offline_run_button.setEnabled(True)
        self.log("Offline-анализ завершён.")

    def start_live_analysis(self) -> None:
        descriptor = self.selected_live_stream()
        if not descriptor:
            self.handle_error("Live-анализ", "LSL-поток не выбран.")
            return

        self.live_worker = LiveCorrelationWorker(
            descriptor=descriptor,
            discover_timeout=self.live_timeout.value(),
            sfreq_override=self.live_sfreq.value(),
            window_seconds=self.live_window.value(),
            step_seconds=self.live_step.value(),
            pull_timeout=self.live_pull_timeout.value(),
            meta_json=self.live_meta.text().strip() or None,
        )
        self.live_worker.log.connect(self.log)
        self.live_worker.update_ready.connect(self.live_update)
        self.live_worker.finished_ok.connect(self.live_finished)
        self.live_worker.failed.connect(lambda msg: self.worker_failed("Live-анализ", msg))
        self.live_worker.start()

        self.live_progress.setRange(0, 0)
        self.live_start_button.setEnabled(False)
        self.live_stop_button.setEnabled(True)

    def stop_live_analysis(self) -> None:
        if self.live_worker:
            self.live_worker.stop()
            self.log("Остановка live-анализа запрошена.")

    def live_update(self, result: dict[str, Any]) -> None:
        self.last_live_result = result
        matrices = {
            "Pearson": result["pearson"],
            "Spearman": result["spearman"],
            "|Pearson - Spearman|": result["diff_abs"],
        }
        self.live_viewer.set_matrices(matrices, result["labels"])
        self.live_summary.setPlainText(json.dumps(result["report"], ensure_ascii=False, indent=2))

    def live_finished(self) -> None:
        self.live_progress.setRange(0, 1)
        self.live_progress.setValue(1)
        self.live_start_button.setEnabled(True)
        self.live_stop_button.setEnabled(False)
        self.log("Live-анализ завершён.")

    def save_live_snapshot(self) -> None:
        if not self.last_live_result:
            self.handle_error("Сохранение live-матриц", "Пока нет рассчитанных матриц.")
            return
        if not self.live_out.text().strip():
            self.handle_error("Сохранение live-матриц", "Не задан префикс сохранения.")
            return
        try:
            prefix = services.save_live_snapshot(
                out_prefix=self.live_out.text().strip(),
                labels=self.last_live_result["labels"],
                pearson=self.last_live_result["pearson"],
                spearman=self.last_live_result["spearman"],
                diff_abs=self.last_live_result["diff_abs"],
                report=self.last_live_result["report"],
                save_csv=self.live_save_csv.isChecked(),
                save_png=self.live_save_png.isChecked(),
            )
            self.log(f"Сохранены последние live-матрицы: {prefix}")
        except Exception as exc:
            self.handle_error("Сохранение live-матриц", str(exc))

    def run_compare(self) -> None:
        if not self.compare_ref.text().strip() or not self.compare_cand.text().strip():
            self.handle_error("Сравнение матриц", "Укажите оба префикса результатов.")
            return
        self.compare_worker = CompareWorker(
            reference_prefix=self.compare_ref.text().strip(),
            candidate_prefix=self.compare_cand.text().strip(),
            meta_json=self.compare_meta.text().strip() or None,
            out_prefix=self.compare_out.text().strip() or None,
            save_csv=self.compare_save_csv.isChecked(),
            save_png=self.compare_save_png.isChecked(),
        )
        self.compare_worker.log.connect(self.log)
        self.compare_worker.finished_ok.connect(self.compare_finished)
        self.compare_worker.failed.connect(lambda msg: self.worker_failed("Сравнение матриц", msg))
        self.compare_worker.start()

        self.compare_progress.setRange(0, 0)
        self.compare_run_button.setEnabled(False)

    def compare_finished(self, result: dict[str, Any]) -> None:
        self.last_compare_result = result
        self.compare_viewer.set_matrices(
            {
                "|Pearson_ref - Pearson_cand|": result["pearson_absdiff"],
                "|Spearman_ref - Spearman_cand|": result["spearman_absdiff"],
            },
            result["labels"],
        )
        self.compare_summary.setPlainText(json.dumps(result["report"], ensure_ascii=False, indent=2))
        self.compare_progress.setRange(0, 1)
        self.compare_progress.setValue(1)
        self.compare_run_button.setEnabled(True)
        self.log("Сравнение матриц завершено.")

    def worker_failed(self, title: str, message: str) -> None:
        self.record_progress.setRange(0, 1)
        self.offline_progress.setRange(0, 1)
        self.live_progress.setRange(0, 1)
        self.compare_progress.setRange(0, 1)
        self.record_start_button.setEnabled(True)
        self.record_stop_button.setEnabled(False)
        self.live_start_button.setEnabled(True)
        self.live_stop_button.setEnabled(False)
        self.offline_run_button.setEnabled(True)
        self.compare_run_button.setEnabled(True)
        self.handle_error(title, message)


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
