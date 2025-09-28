from itertools import count
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox, font as tkfont, ttk
import serial
import serial.tools.list_ports
import time
import os
import csv
from datetime import datetime
import re
import random

import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
BAUD_RATE               = 9600
FONT_FAMILY             = 'Consolas'
FONT_SIZE_TEXT          = 11
FONT_SIZE_ENTRY         = 12

OUTPUT_DIR              = 'out_data'
VIDEO_FPS               = 20.0
SERIAL_TIMEOUT          = 1
BOARD_RESET_DELAY_MS    = 2000
REDO_COOLDOWN_S         = 10 # Cooldown time in seconds after flagging an error

# --- OpenCV Text & Font Configuration ---
THAI_FONT_PATH          = "tahoma.ttf"
OPENCV_TEXT_FONT_SIZE_SMALL = 18
OPENCV_TEXT_FONT_SIZE_LARGE = 22
OPENCV_TEXT_FONT_SIZE_COOLDOWN = 60

# --- Labeling Configuration ---
DEFAULT_UNLABELED_VALUE = "nothing"
REDO_FLAG_LABEL         = "error_redo"
REDO_COOLDOWN_LABEL     = "cooldown"
BREAK_TIME_LABEL        = "break_time"
SELECTABLE_LABELS_LIST = [
    "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ"
]
GROUPED_LABEL_SEPARATOR = "_"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class SerialGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Sensor Data Collection & Labeling Tool")

        # --- State Variables ---
        self.ser: serial.Serial | None = None
        self.alive: bool = False
        self.reader_thread: threading.Thread | None = None
        self.sensor_paused: bool = False
        self.record: bool = False
        self.sensor_data: list[list] = []
        self.output_filename_base: str | None = None
        self.current_session_date: str | None = None
        self.video_writer = None
        self.last_timestamp_ms: str | None = None
        self.count_sample: int = 0
        # --- Labeling & UI State ---
        self.original_labels_list = SELECTABLE_LABELS_LIST[:]
        self.current_grouped_labels_list: list[str] = []
        self.label_group_size_var = tk.IntVar(value=1)
        self.break_time_var = tk.IntVar(value=3)
        self.current_selected_grouped_label_var = tk.StringVar()
        self.active_individual_components_list: list[str] = []
        self.current_normal_mode_component_idx: int = 0
        self.normal_mode_completed_cycles: int = 0
        self.labeling_is_active: bool = False
        self.current_continuous_label: str = DEFAULT_UNLABELED_VALUE
        self.last_label_start_index: int = 0

        # --- Cooldown States ---
        self.is_in_redo_cooldown: bool = False
        self.is_in_break_cooldown: bool = False
        self.cooldown_end_time: float = 0.0

        # --- Random Mode State ---
        self.random_labeling_mode_active = tk.BooleanVar(value=False)
        self.random_sequence_running = False
        self.random_label_sequence: list[str] = []
        self.current_random_label_index: int = 0
        self.total_random_labels_in_sequence: int = 0
        
        # --- Auto-Advance State ---
        self.auto_label_mode_active = tk.BooleanVar(value=False)
        self.auto_label_duration_var = tk.DoubleVar(value=3.0)
        self.is_in_auto_label_period: bool = False
        self.auto_label_end_time: float = 0.0


        # --- Video & Graphics State ---
        self.cap: cv2.VideoCapture = cv2.VideoCapture(0)
        self.image_label: tk.Label | None = None
        self.cv_thai_font_small = None
        self.cv_thai_font_large = None
        self.cv_thai_font_cooldown = None
        self._load_fonts()

        self._create_gui_elements()

        self.update_label_grouping()
        self.update_image()
        self._update_ui_element_states()
        self.master.protocol("WM_DELETE_WINDOW", self.close)

    def _load_fonts(self):
        if THAI_FONT_PATH and THAI_FONT_PATH.strip():
            try:
                self.cv_thai_font_small = ImageFont.truetype(THAI_FONT_PATH, OPENCV_TEXT_FONT_SIZE_SMALL)
                self.cv_thai_font_large = ImageFont.truetype(THAI_FONT_PATH, OPENCV_TEXT_FONT_SIZE_LARGE)
                self.cv_thai_font_cooldown = ImageFont.truetype(THAI_FONT_PATH, OPENCV_TEXT_FONT_SIZE_COOLDOWN)
                print(f"Successfully loaded Thai font: {THAI_FONT_PATH}")
            except IOError:
                print(f"Error: Could not load font '{THAI_FONT_PATH}'.")
        else:
            print("Warning: THAI_FONT_PATH is not set.")

    def _create_gui_elements(self):
        self.master.rowconfigure(4, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.text_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_TEXT)
        self.entry_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_ENTRY)

        port_frame = ttk.Labelframe(self.master, text="Connection")
        port_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        port_frame.columnconfigure(1, weight=1)
        tk.Label(port_frame, text="Port:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, state='readonly')
        self.port_combo.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        self.refresh_button = ttk.Button(port_frame, text="Refresh", command=self.refresh_ports)
        self.refresh_button.grid(row=0, column=2, padx=5, pady=5)
        self.connect_button = ttk.Button(port_frame, text="Connect", command=self.connect_serial)
        self.connect_button.grid(row=0, column=3, padx=5, pady=5)
        self.disconnect_button = ttk.Button(port_frame, text="Disconnect", command=self.disconnect_serial, state=tk.DISABLED)
        self.disconnect_button.grid(row=0, column=4, padx=5, pady=5)
        self.refresh_ports()

        label_frame = ttk.Labelframe(self.master, text="Labeling Controls")
        label_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        label_frame.columnconfigure(2, weight=1)
        grouping_frame = ttk.Frame(label_frame)
        grouping_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Label(grouping_frame, text="Group Size:").pack(side=tk.LEFT, padx=(0, 2))
        self.label_group_size_entry = ttk.Entry(grouping_frame, textvariable=self.label_group_size_var, width=4)
        self.label_group_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.apply_grouping_button = ttk.Button(grouping_frame, text="Apply", command=self.update_label_grouping)
        self.apply_grouping_button.pack(side=tk.LEFT, padx=(0, 15))
        tk.Label(grouping_frame, text="Break Time (s):").pack(side=tk.LEFT, padx=(0, 2))
        self.break_time_entry = ttk.Entry(grouping_frame, textvariable=self.break_time_var, width=4)
        self.break_time_entry.pack(side=tk.LEFT)
        tk.Label(label_frame, text="Select Set ('s' key cycles):").grid(row=0, column=1, sticky="e", padx=(10, 2))
        self.label_combo = ttk.Combobox(label_frame, textvariable=self.current_selected_grouped_label_var, values=[], state='readonly')
        self.label_combo.grid(row=0, column=2, sticky="we", padx=5, pady=5)
        self.label_combo.bind("<<ComboboxSelected>>", self._on_grouped_label_selected)

        custom_label_frame = ttk.Labelframe(self.master, text="Custom Label Set")
        custom_label_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        custom_label_frame.columnconfigure(0, weight=1)

        self.custom_labels_entry_var = tk.StringVar()
        self.custom_labels_entry = ttk.Entry(custom_label_frame, textvariable=self.custom_labels_entry_var)
        self.custom_labels_entry.grid(row=0, column=0, sticky="we", padx=5, pady=5)
        self.custom_labels_entry.insert(0, ", ".join(SELECTABLE_LABELS_LIST))

        self.update_labels_button = ttk.Button(custom_label_frame, text="Update Labels", command=self._update_custom_labels)
        self.update_labels_button.grid(row=0, column=1, padx=5, pady=5)

        random_mode_frame = ttk.Labelframe(self.master, text="Setting")
        random_mode_frame.grid(row=3, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        random_mode_frame.columnconfigure(3, weight=1)
        
        # --- Settings Row 1 ---
        tk.Label(random_mode_frame, text="Count per Label:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.random_label_count_var = tk.IntVar(value=5)
        self.random_label_count_entry = ttk.Entry(random_mode_frame, textvariable=self.random_label_count_var, width=5)
        self.random_label_count_entry.grid(row=0, column=1, padx=5, pady=5)
        self.random_mode_checkbox = ttk.Checkbutton(random_mode_frame, text="Enable Random Mode", variable=self.random_labeling_mode_active, command=self.on_random_mode_toggled)
        self.random_mode_checkbox.grid(row=0, column=2, padx=10, pady=5)
        self.random_mode_status_var = tk.StringVar()
        ttk.Label(random_mode_frame, textvariable=self.random_mode_status_var, anchor="w").grid(row=0, column=3, sticky="we", padx=5, pady=5)

        # --- Settings Row 2 (Auto-Advance Controls) ---
        tk.Label(random_mode_frame, text="Label Duration (s):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.auto_label_duration_entry = ttk.Entry(random_mode_frame, textvariable=self.auto_label_duration_var, width=5)
        self.auto_label_duration_entry.grid(row=1, column=1, padx=5, pady=5)
        self.auto_mode_checkbox = ttk.Checkbutton(random_mode_frame, text="Enable Auto-Advance", variable=self.auto_label_mode_active)
        self.auto_mode_checkbox.grid(row=1, column=2, padx=10, pady=5)


        outer_pane = tk.PanedWindow(self.master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        outer_pane.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        left_pane = tk.PanedWindow(outer_pane, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        top_frame = ttk.Labelframe(left_pane, text="Commands / Status")
        top_frame.rowconfigure(0, weight=1); top_frame.columnconfigure(0, weight=1)
        self.top_text = scrolledtext.ScrolledText(top_frame, wrap=tk.WORD, font=self.text_font, state=tk.DISABLED, bg='black', fg='lightyellow')
        self.top_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_pane.add(top_frame, minsize=180)
        bottom_frame = ttk.Labelframe(left_pane, text="Session Control & Sensor Data")
        bottom_frame.rowconfigure(2, weight=1); bottom_frame.columnconfigure(1, weight=1)
        tk.Label(bottom_frame, text="Filename Base:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.filename_entry_var = tk.StringVar()
        self.filename_entry = ttk.Entry(bottom_frame, textvariable=self.filename_entry_var)
        self.filename_entry.grid(row=0, column=1, sticky="we", padx=5, pady=3)
        button_bar_frame = ttk.Frame(bottom_frame)
        button_bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        button_bar_frame.columnconfigure((0, 1, 2), weight=1)
        self.pause_button = ttk.Button(button_bar_frame, text="Pause Sensor", command=self.toggle_pause_sensor)
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.record_button = ttk.Button(button_bar_frame, text="Record (a)", command=self.toggle_record)
        self.record_button.grid(row=0, column=1, sticky="ew", padx=2)
        self.error_button = ttk.Button(button_bar_frame, text="Flag & Redo (d)", command=self.on_error_button_press, state=tk.DISABLED)
        self.error_button.grid(row=0, column=2, sticky="ew", padx=(2, 0))
        self.bottom_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, font=self.text_font, state=tk.DISABLED, bg='black', fg='cyan')
        self.bottom_text.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        left_pane.add(bottom_frame, minsize=150)
        outer_pane.add(left_pane, stretch="always", minsize=300)
        image_frame = ttk.LabelFrame(outer_pane, text="Camera Feed")
        image_frame.rowconfigure(0, weight=1); image_frame.columnconfigure(0, weight=1)
        self.image_label = tk.Label(image_frame, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        outer_pane.add(image_frame, stretch="always", minsize=300)

        send_frame = ttk.Labelframe(self.master, text="Serial Command")
        send_frame.grid(row=5, column=0, columnspan=2, sticky="we", padx=10, pady=(5, 10))
        send_frame.columnconfigure(0, weight=1)
        self.send_entry = ttk.Entry(send_frame, font=self.entry_font, state=tk.DISABLED)
        self.send_entry.grid(row=0, column=0, sticky="we", padx=5, pady=5)
        self.send_entry.bind("<Return>", self.send_data)
        self.send_button = ttk.Button(send_frame, text="Send", state=tk.DISABLED, command=self.send_data)
        self.send_button.grid(row=0, column=1, padx=(0, 5), pady=5)

        self.master.bind("<s>", self.on_s_key_press)
        self.master.bind("<a>", self.on_a_key_press)
        self.master.bind("<d>", self.on_d_key_press)

    def _get_current_effective_label(self) -> str:
        """Enhanced label determination with better state checking."""
        if not self.record:
            return "not_recording"
        
        if self.is_in_redo_cooldown:
            return REDO_COOLDOWN_LABEL
        
        if self.is_in_break_cooldown:
            return BREAK_TIME_LABEL
        
        if self.labeling_is_active:
            # Ensure we have a valid label
            if self.current_continuous_label and self.current_continuous_label.strip():
                return self.current_continuous_label
        
        return DEFAULT_UNLABELED_VALUE

    def _update_custom_labels(self):
        custom_labels_str = self.custom_labels_entry_var.get().strip()
        if not custom_labels_str:
            messagebox.showwarning("Input Error", "Custom labels field is empty. Please enter labels separated by commas.")
            return

        new_labels = [label.strip() for label in custom_labels_str.split(',') if label.strip()]

        if not new_labels:
            messagebox.showwarning("Input Error", "No valid labels found. Please enter labels separated by commas.")
            return

        self.original_labels_list = new_labels
        self._append_to_top(f"[Info] Label set updated. New labels: {self.original_labels_list}\n")
        self.update_label_grouping()

    def on_a_key_press(self, event=None):
        focused_widget = self.master.focus_get()
        if isinstance(focused_widget, ttk.Entry):
            return
        self.toggle_record()

    def on_d_key_press(self, event=None):
        focused_widget = self.master.focus_get()
        if isinstance(focused_widget, ttk.Entry):
            return
        self.on_error_button_press()

    def _update_ui_element_states(self):
        """Enhanced UI state management with better error button control."""
        # More precise error button state control
        can_flag_error = (
            self.record and 
            self.labeling_is_active and 
            not self.is_in_redo_cooldown and 
            not self.is_in_break_cooldown and
            len(self.sensor_data) > self.last_label_start_index  # Ensure we have data to flag
        )
        
        self.error_button.config(state=tk.NORMAL if can_flag_error else tk.DISABLED)

        # Update status display
        if self.is_in_redo_cooldown:
            remaining_time = max(0, self.cooldown_end_time - time.time())
            self.random_mode_status_var.set(f"COOLDOWN ACTIVE... ({remaining_time:.1f}s remaining)")
            return
            
        if self.is_in_break_cooldown:
            remaining_time = max(0, self.cooldown_end_time - time.time())
            self.random_mode_status_var.set(f"BREAK TIME... ({remaining_time:.1f}s remaining)")
            return

        set_info = f" (Set: '{self.current_selected_grouped_label_var.get() or 'None'}')"
        if not self.random_labeling_mode_active.get():
            self.random_mode_status_var.set("Random mode: INACTIVE")
            return

        if not self.record:
            self.random_mode_status_var.set(f"Random: ENABLED (Start Recording to activate){set_info}")
            return
        if not self.active_individual_components_list:
            self.random_mode_status_var.set(f"Random: ENABLED (Select a Label Set to activate){set_info}")
            return

        if not self.random_sequence_running:
             self.random_mode_status_var.set(f"Random: READY{set_info} (Press 's' to start sequence)")
             return

        if self.current_random_label_index >= self.total_random_labels_in_sequence:
            progress = f"({self.total_random_labels_in_sequence}/{self.total_random_labels_in_sequence})"
            self.random_mode_status_var.set(f"Random: COMPLETE!{set_info} {progress}. 's' to restart.")
            return

        current_target_label = self.random_label_sequence[self.current_random_label_index]
        progress = f"({self.current_random_label_index + 1}/{self.total_random_labels_in_sequence})"
        action_text = "LABELING" if self.labeling_is_active else "NEXT"
        instruction_text = "'s' to STOP." if self.labeling_is_active else "'s' to START."
        self.random_mode_status_var.set(f"Random: {action_text} '{current_target_label}'{set_info} {progress}. {instruction_text}")

    def on_s_key_press(self, event=None):
        if self.is_in_redo_cooldown or self.is_in_break_cooldown:
            self._append_to_top("[Warning] Cannot label. A cooldown is active.\n")
            return "break"

        focused_widget = self.master.focus_get()
        if isinstance(focused_widget, (ttk.Entry, ttk.Button)):
            return

        if not self.record:
            self._append_to_top("[Warning] Press 'Record' to enable labeling.\n")
            return "break"
        
        if self.random_labeling_mode_active.get():
            self._handle_random_mode_s_key()
        else:
            self._handle_normal_mode_s_key()

        self._update_ui_element_states()
        if focused_widget == self.label_combo: return "break"
        return None

    def on_error_button_press(self):
        """Handle error flagging and redo functionality with proper validation and labeling."""
        
        # Enhanced validation checks
        if not self.record:
            self._append_to_top("[Warning] Cannot flag error - not currently recording.\n")
            return
            
        if not self.labeling_is_active:
            self._append_to_top("[Warning] Cannot flag error - no active labeling session.\n")
            return
            
        if self.is_in_redo_cooldown or self.is_in_break_cooldown:
            self._append_to_top("[Warning] Cannot flag error - cooldown is currently active.\n")
            return
        
        if not self.sensor_data:
            self._append_to_top("[Warning] Cannot flag error - no sensor data recorded yet.\n")
            return
        
        if self.last_label_start_index >= len(self.sensor_data):
            self._append_to_top("[Warning] Cannot flag error - invalid label start index.\n")
            return

        # Store the failed label for logging
        failed_label = self.current_continuous_label
        
        # Relabel all data points from the current labeling session
        num_relabeled = 0
        for i in range(self.last_label_start_index, len(self.sensor_data)):
            # Parse the existing data line to extract sensor values
            data_line = self.sensor_data[i]
            
            # Find the label portion and replace it
            if ",[Label]" in data_line:
                # Split at the label marker and reconstruct with new label
                sensor_part = data_line.split(",[Label]")[0]
                self.sensor_data[i] = f"{sensor_part},[Label]['{REDO_FLAG_LABEL}']"
                num_relabeled += 1
            else:
                # Fallback: if format is unexpected, append the redo flag
                self.sensor_data[i] = f"{data_line},[Label]['{REDO_FLAG_LABEL}']"
                num_relabeled += 1
        
        # Log the redo action
        self._append_to_top(
            f"[REDO] Attempt for '{failed_label}' flagged as error. "
            f"{num_relabeled} data points relabeled to '{REDO_FLAG_LABEL}'.\n"
        )
        
        # Stop current labeling session
        self.labeling_is_active = False
        self.is_in_auto_label_period = False # Also stop auto-label timer
        self.current_continuous_label = DEFAULT_UNLABELED_VALUE
        
        # Start cooldown period
        self.is_in_redo_cooldown = True
        self.cooldown_end_time = time.time() + REDO_COOLDOWN_S
        self._append_to_top(f"[Info] Starting {REDO_COOLDOWN_S}s error cooldown period.\n")
        
        # Update UI to reflect new state
        self._update_ui_element_states()

    def _start_break_cooldown(self):
        try:
            break_duration = self.break_time_var.get()
        except tk.TclError:
            break_duration = 0

        if break_duration > 0:
            self.is_in_break_cooldown = True
            self.cooldown_end_time = time.time() + break_duration
            self._append_to_top(f"[Info] Starting {break_duration}s break.\n")
        else:
            # If no break, and auto mode is on, immediately trigger next label
            if self.auto_label_mode_active.get() and self.record:
                 self._start_next_label_automatically()
            elif self.active_individual_components_list: # If not auto mode, just show prompt
                next_label = self.active_individual_components_list[self.current_normal_mode_component_idx]
                self._append_to_top(f"[Info] Next label is '{next_label}'.\n")
        self._update_ui_element_states()

    def on_random_mode_toggled(self):
        is_now_enabled = self.random_labeling_mode_active.get()
        if is_now_enabled:
            if self.record and self.active_individual_components_list:
                self._try_start_random_sequence_run()
        else:
            if self.random_sequence_running:
                self._execute_stop_random_mode()
        self._update_ui_element_states()

    def _draw_text_with_bg(self, draw_ctx, text, pos, font, color, bg_color=(0,0,0,128)):
        if not font or not draw_ctx: return
        try:
            text_bbox = draw_ctx.textbbox(pos, text, font=font)
            bg_bbox = (text_bbox[0] - 2, text_bbox[1], text_bbox[2] + 2, text_bbox[3] + 2)
            draw_ctx.rectangle(bg_bbox, fill=bg_color)
            draw_ctx.text(pos, text, font=font, fill=color)
        except Exception:
            pass

    def _draw_centered_text(self, draw_ctx, text, font, color, y_pos, bg_color=(0,0,0,128)):
        if not font or not draw_ctx: return
        w, _ = pil_image.size
        text_bbox = draw_ctx.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        pos = ((w - text_w) / 2, y_pos)
        self._draw_text_with_bg(draw_ctx, text, pos, font, color, bg_color)

    def update_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.master.after(30, self.update_image)
            return

        frame = cv2.flip(frame, 1)
        global pil_image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw_ctx = ImageDraw.Draw(pil_image)

        y_offset = 10
        self._draw_text_with_bg(draw_ctx, f"RECORDING: {self.record}", (10, y_offset), self.cv_thai_font_small, (0, 255, 0) if self.record else (255, 255, 0))
        y_offset += 30

        if self.last_timestamp_ms is not None:
            self._draw_text_with_bg(draw_ctx, f"Timestamp: {self.last_timestamp_ms} ms", (10, y_offset), self.cv_thai_font_small, (255, 255, 255))
            y_offset += 30

            self._draw_text_with_bg(draw_ctx, f"sample: {self.count_sample}", (10, y_offset), self.cv_thai_font_small, (255, 255, 255))
            y_offset += 30

        # --- HYBRID DISPLAY LOGIC ---
        effective_label = self._get_current_effective_label()
        label_color = (255, 255, 255)
        if self.labeling_is_active:
            label_color = (0, 255, 255)
        elif self.is_in_break_cooldown or self.is_in_redo_cooldown:
            label_color = (255, 255, 0)
        self._draw_text_with_bg(draw_ctx, f"CURRENT LABEL: {effective_label}", (10, y_offset), self.cv_thai_font_large, label_color)
        y_offset += 35

        # --- Auto-Advance Status Display ---
        if self.auto_label_mode_active.get() and self.record:
            auto_status_text = "AUTO-ADVANCE ACTIVE"
            auto_status_color = (50, 255, 150) # Light green
            if self.is_in_auto_label_period:
                remaining_auto_time = max(0, self.auto_label_end_time - time.time())
                auto_status_text += f" ({remaining_auto_time:.1f}s)"
            self._draw_text_with_bg(draw_ctx, auto_status_text, (10, y_offset), self.cv_thai_font_large, auto_status_color)
            y_offset += 35


        prompt_text = ""
        if self.record and effective_label == DEFAULT_UNLABELED_VALUE:
            if self.random_labeling_mode_active.get():
                if self.random_sequence_running:
                     if self.current_random_label_index < self.total_random_labels_in_sequence:
                          next_label = self.random_label_sequence[self.current_random_label_index]
                          prompt_text = f"PROMPT: Press 's' to START '{next_label}'"
                     else:
                          prompt_text = "PROMPT: Sequence COMPLETE! Press 's' to restart."
            else: # Normal mode
                if self.active_individual_components_list:
                    next_label = self.active_individual_components_list[self.current_normal_mode_component_idx]
                    prompt_text = f"PROMPT: Press 's' to START '{next_label}'"

        if prompt_text:
            self._draw_text_with_bg(draw_ctx, prompt_text, (10, y_offset), self.cv_thai_font_large, (255, 255, 0))
        # --- END HYBRID DISPLAY LOGIC ---

        # --- Auto-advance timer logic ---
        if self.is_in_auto_label_period and time.time() >= self.auto_label_end_time:
            # Call stop function, indicating it was NOT a manual stop
            self._stop_current_label_and_advance(manual_stop=False)

        if self.is_in_redo_cooldown:
            remaining = self.cooldown_end_time - time.time()
            if remaining > 0:
                h = pil_image.size[1]
                self._draw_centered_text(draw_ctx, "COOLDOWN", self.cv_thai_font_cooldown, (255,255,0), h / 2 - 50, bg_color=(255,0,0,150))
                self._draw_centered_text(draw_ctx, f"{remaining:.1f}s", self.cv_thai_font_cooldown, (255,255,0), h / 2 + 10, bg_color=(255,0,0,150))
            else:
                self.is_in_redo_cooldown = False
                self._append_to_top("[Info] Cooldown finished. Ready to retry.\n")
                self._update_ui_element_states()

        elif self.is_in_break_cooldown:
            remaining = self.cooldown_end_time - time.time()
            if remaining > 0:
                h = pil_image.size[1]
                self._draw_centered_text(draw_ctx, "NEXT IN...", self.cv_thai_font_cooldown, (255,255,255), h / 2 - 50, bg_color=(0,150,0,150))
                self._draw_centered_text(draw_ctx, f"{remaining:.1f}s", self.cv_thai_font_cooldown, (255,255,255), h / 2 + 10, bg_color=(0,150,0,150))
            else:
                self.is_in_break_cooldown = False
                self._append_to_top("[Info] Break finished.\n")
                # Trigger next label if auto mode is on
                if self.auto_label_mode_active.get() and self.record:
                    self._start_next_label_automatically()
                self._update_ui_element_states()

        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
        if self.record and self.video_writer: self.video_writer.write(frame)

        target_w, target_h = self.image_label.winfo_width(), self.image_label.winfo_height()
        if target_w > 1 and target_h > 1:
            h, w, _ = frame.shape
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset, y_offset = (target_w - nw) // 2, (target_h - nh) // 2
            canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized_frame
            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

        self.master.after(30, self.update_image)

    def _process_serial_data(self, line: str):
        """Process incoming serial data with consistent label formatting."""
        if line.startswith("[Sensor]"):
            self._append_to_bottom(line + '\n')
            sensor_values_str = line.replace("[Sensor]", "").strip()
            readings = sensor_values_str.split(',')

            if readings and readings[0]:
                self.last_timestamp_ms = readings[0]

            if self.record:
                if self.labeling_is_active:
                    self.count_sample += 1
                else:
                    self.count_sample = 0

                # Get current effective label and format consistently
                current_label = self._get_current_effective_label()
                # Use consistent label format: [Label]['label_name']
                formatted_data = f"{sensor_values_str},[Label]['{current_label}']"
                self.sensor_data.append(formatted_data)
        else:
            self._append_to_top(line + '\n')

    def toggle_record(self):
        if not self.record:
            if not self.ser or not self.ser.is_open:
                messagebox.showwarning("Not Connected", "Connect to a serial device before recording.")
                return

            self.current_session_date = datetime.now().strftime("%Y-%m-%d")
            data_dir = os.path.join(OUTPUT_DIR, self.current_session_date, "data")
            video_dir = os.path.join(OUTPUT_DIR, self.current_session_date, "video")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)

            user_filename = self.filename_entry_var.get().strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-.]+', '', user_filename.replace(" ", "_")) if user_filename else "session"
            self.output_filename_base = f"{timestamp}_{safe_filename}"

            self.sensor_data.clear()
            self.labeling_is_active = False
            self.is_in_auto_label_period = False # Reset auto-label state
            self.current_continuous_label = DEFAULT_UNLABELED_VALUE
            self.normal_mode_completed_cycles = 0 # Reset cycle count for new recording

            # --- FIX: Immediately start random sequence if mode is active ---
            if self.random_labeling_mode_active.get():
                if not self._try_start_random_sequence_run():
                    self.output_filename_base = None
                    self._append_to_top("[Error] Could not start random sequence. Aborting record.\n")
                    return

            video_path = os.path.join(video_dir, f"{self.output_filename_base}.mp4")
            self.master.after(BOARD_RESET_DELAY_MS, lambda: self._finalize_recording_start(video_path))

        else: # Stopping recording
            self.record = False
            self.is_in_auto_label_period = False # Stop auto-timer
            if self.is_in_redo_cooldown or self.is_in_break_cooldown:
                self.is_in_redo_cooldown = False
                self.is_in_break_cooldown = False
                self._append_to_top("[Info] Cooldown cancelled due to stopping recording.\n")
            if self.random_sequence_running: self._execute_stop_random_mode()
            if self.labeling_is_active:
                self.labeling_is_active = False
                self.current_continuous_label = DEFAULT_UNLABELED_VALUE
                self._append_to_top(f"[Warning] Labeling stopped.\n")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self._append_to_top(f"[Info] Video stopped: {self.output_filename_base}.mp4\n")
            self._save_sensor_data()
            self.output_filename_base = None

        self.record_button.config(text="Stop Recording (a)" if self.record else "Record (a)")
        self.filename_entry.config(state=tk.DISABLED if self.record else tk.NORMAL)
        
        # Disable auto-advance settings during recording
        auto_advance_widget_state = tk.DISABLED if self.record else tk.NORMAL
        self.auto_mode_checkbox.config(state=auto_advance_widget_state)
        self.auto_label_duration_entry.config(state=auto_advance_widget_state)

        self._update_ui_element_states()

    def _set_grouping_and_random_count_entry_state(self, new_state: str):
        combo_state = 'readonly' if new_state == tk.NORMAL else tk.DISABLED
        self.label_group_size_entry.config(state=new_state)
        self.apply_grouping_button.config(state=new_state)
        self.label_combo.config(state=combo_state)
        self.random_label_count_entry.config(state=new_state)
        self.break_time_entry.config(state=new_state)

    def _handle_normal_mode_s_key(self):
        if not self.active_individual_components_list:
            self._append_to_top("[Warning] No labels in the selected set. Choose a set first.\n")
            return

        try:
            total_cycles_to_run = self.random_label_count_var.get()
            if total_cycles_to_run <= 0:
                self._append_to_top("[Error] 'Count' value in settings must be a positive number.\n")
                return
        except tk.TclError:
            self._append_to_top("[Error] Invalid 'Count' value in settings. Please enter a number.\n")
            return

        if self.normal_mode_completed_cycles >= total_cycles_to_run:
            self._append_to_top(f"[Info] All {total_cycles_to_run} cycles are complete. To start over, change the label set or restart recording.\n")
            return

        if not self.labeling_is_active:
            target_label = self.active_individual_components_list[self.current_normal_mode_component_idx]
            self._start_labeling_session(target_label)
            
            progress_in_cycle = self.current_normal_mode_component_idx + 1
            total_in_cycle = len(self.active_individual_components_list)
            current_cycle = self.normal_mode_completed_cycles + 1
            self._append_to_top(
                f"[Labeling STARTED] Applying '{self.current_continuous_label}' ({progress_in_cycle}/{total_in_cycle}) | Cycle {current_cycle}/{total_cycles_to_run}\n"
            )
        else:
            # User is pressing 's' to manually stop a label
            self._stop_current_label_and_advance(manual_stop=True)

    def _handle_random_mode_s_key(self):
        is_finished = self.current_random_label_index >= self.total_random_labels_in_sequence
        if not self.random_sequence_running or is_finished:
            if not self._try_start_random_sequence_run():
                return
        
        if not self.labeling_is_active:
            target_label = self.random_label_sequence[self.current_random_label_index]
            self._start_labeling_session(target_label)
            self._append_to_top(f"[Labeling STARTED] Random: Applying '{target_label}' ({self.current_random_label_index + 1}/{self.total_random_labels_in_sequence}).\n")
        else:
            # User is pressing 's' to manually stop a label
            self._stop_current_label_and_advance(manual_stop=True)
    
    def _start_labeling_session(self, label_to_apply: str):
        """Centralized function to begin any labeling session."""
        self.current_continuous_label = label_to_apply
        self.labeling_is_active = True
        self.last_label_start_index = len(self.sensor_data)

        # Start auto-advance timer if enabled
        if self.auto_label_mode_active.get():
            try:
                duration = self.auto_label_duration_var.get()
                if duration > 0:
                    self.is_in_auto_label_period = True
                    self.auto_label_end_time = time.time() + duration
                    self._append_to_top(f"[AUTO] Timer started for {duration}s.\n")
            except tk.TclError:
                self._append_to_top("[Warning] Invalid auto-label duration. Timer not started.\n")

    def _stop_current_label_and_advance(self, manual_stop: bool = False):
        """
        Centralized function to stop a label and move to the next state.
        The 'manual_stop' flag determines if the auto-advance sequence should continue.
        """
        if not self.labeling_is_active:
            return

        stopped_label = self.current_continuous_label
        self.labeling_is_active = False
        self.is_in_auto_label_period = False # Always stop the timer
        self.current_continuous_label = DEFAULT_UNLABELED_VALUE
        
        log_msg = "[Labeling STOPPED]"
        if manual_stop:
            log_msg += f" Manually stopped '{stopped_label}'."
        else:
            log_msg += f" [AUTO] Finished '{stopped_label}'."
        self._append_to_top(log_msg + '\n')

        # --- Advance the label index regardless of stop reason ---
        sequence_is_complete = self._advance_label_index()

        # --- Decide what to do next ---
        if not sequence_is_complete:
            # If the stop was AUTOMATIC (timer) AND auto-mode is on, continue the sequence.
            if manual_stop or self.auto_label_mode_active.get():
                self._start_break_cooldown()
            else:
                # If stop was MANUAL, or if auto-mode is off, PAUSE the sequence.
                # The system will now wait for the next 's' press.
                self._append_to_top("[Info] Sequence paused. Press 's' to start next label.\n")
        
        self._update_ui_element_states()

    def _advance_label_index(self) -> bool:
        """
        Increments the appropriate label index (random or normal).
        Returns True if the entire sequence is complete, False otherwise.
        """
        if self.random_labeling_mode_active.get():
            self.current_random_label_index += 1
            if self.current_random_label_index >= self.total_random_labels_in_sequence:
                self._append_to_top("[Info] Random labeling sequence complete!\n")
                return True
        else: # Normal Mode
            self.current_normal_mode_component_idx += 1
            total_cycles_to_run = self.random_label_count_var.get()
            if self.current_normal_mode_component_idx >= len(self.active_individual_components_list):
                self.normal_mode_completed_cycles += 1
                self.current_normal_mode_component_idx = 0
                self._append_to_top(f"[Info] Cycle {self.normal_mode_completed_cycles}/{total_cycles_to_run} complete!\n")
                if self.normal_mode_completed_cycles >= total_cycles_to_run:
                    self._append_to_top("[Info] All normal mode labeling cycles finished.\n")
                    return True
        return False

    def _start_next_label_automatically(self):
        """Called after a break cooldown when auto-advance is on."""
        if not self.record or not self.auto_label_mode_active.get() or self.labeling_is_active:
            return # Safety checks

        self._append_to_top("[AUTO] Starting next label automatically.\n")

        if self.random_labeling_mode_active.get():
            # Check if random sequence is finished
            if self.current_random_label_index < self.total_random_labels_in_sequence:
                self._handle_random_mode_s_key()
        else:
            # Check if normal mode cycles are finished
            total_cycles_to_run = self.random_label_count_var.get()
            if self.normal_mode_completed_cycles < total_cycles_to_run:
                self._handle_normal_mode_s_key()
        
        self._update_ui_element_states()


    def _try_start_random_sequence_run(self) -> bool:
        if not self.generate_random_label_sequence():
            self._append_to_top("[Error] Failed to generate random label sequence.\n")
            return False
        self._set_grouping_and_random_count_entry_state(tk.DISABLED)
        self.random_sequence_running = True
        self.labeling_is_active = False
        self.current_continuous_label = DEFAULT_UNLABELED_VALUE
        self._append_to_top(f"[Info] Random labeling sequence STARTED. {self.total_random_labels_in_sequence} items in sequence.\n")
        if self.total_random_labels_in_sequence > 0:
             self._append_to_top(f"[Info] Random: First is '{self.random_label_sequence[0]}'.\n")
             # Auto-start the very first label if auto mode is on
             if self.auto_label_mode_active.get():
                 self.master.after(500, self._start_next_label_automatically) # slight delay
        return True

    def _execute_stop_random_mode(self):
        if self.labeling_is_active:
            self.labeling_is_active = False
            self.is_in_auto_label_period = False
            self.current_continuous_label = DEFAULT_UNLABELED_VALUE
            self._append_to_top(f"[Labeling STOPPED] Random mode was manually ended.\n")
        self.random_label_sequence.clear()
        self.current_random_label_index = 0
        self.total_random_labels_in_sequence = 0
        self.random_sequence_running = False
        self._set_grouping_and_random_count_entry_state(tk.NORMAL)
        self._append_to_top("[Info] Random labeling session STOPPED/RESET.\n")
        self._update_ui_element_states()

    def generate_random_label_sequence(self) -> bool:
        if not self.active_individual_components_list:
            messagebox.showerror("Error", "No active labels in the selected set for random mode.")
            return False
        try:
            count_per_label = self.random_label_count_var.get()
            if count_per_label <= 0:
                messagebox.showerror("Error", "Count per label must be a positive number.")
                return False
        except tk.TclError:
            messagebox.showerror("Error", "Invalid count per label. Please enter a number.")
            return False
        self.random_label_sequence = self.active_individual_components_list * count_per_label
        random.shuffle(self.random_label_sequence)
        self.current_random_label_index = 0
        self.total_random_labels_in_sequence = len(self.random_label_sequence)
        return self.total_random_labels_in_sequence > 0

    def update_label_grouping(self, event=None):
        if self.labeling_is_active:
            self._append_to_top(f"[Warning] Labeling stopped due to regrouping.\n")
            self.labeling_is_active = False
            self.is_in_auto_label_period = False
            self.current_continuous_label = DEFAULT_UNLABELED_VALUE
        if self.random_sequence_running:
            self._append_to_top("[Info] Label grouping changed. Stopping active random session.\n")
            self._execute_stop_random_mode()
        try:
            group_size = max(1, self.label_group_size_var.get())
        except tk.TclError:
            group_size = 1
        self.label_group_size_var.set(group_size)
        if group_size == 1:
            self.current_grouped_labels_list = self.original_labels_list[:]
        else:
            self.current_grouped_labels_list = [GROUPED_LABEL_SEPARATOR.join(self.original_labels_list[i:i + group_size])
                                                for i in range(0, len(self.original_labels_list), group_size)]
        self.label_combo['values'] = self.current_grouped_labels_list
        self.current_selected_grouped_label_var.set(self.current_grouped_labels_list[0] if self.current_grouped_labels_list else "")
        self._update_active_individual_components()

    def _on_grouped_label_selected(self, event=None):
        self._update_active_individual_components()

    def _update_active_individual_components(self):
        if self.labeling_is_active:
            self._append_to_top(f"[Warning] Labeling stopped due to set change.\n")
            self.labeling_is_active = False
            self.is_in_auto_label_period = False
            self.current_continuous_label = DEFAULT_UNLABELED_VALUE
        if self.random_sequence_running:
            self._append_to_top("[Warning] Active label set changed. Stopping random mode.\n")
            self._execute_stop_random_mode()
        selected_group = self.current_selected_grouped_label_var.get()
        self.active_individual_components_list = [
            comp.strip() for comp in selected_group.split(GROUPED_LABEL_SEPARATOR) if comp.strip()
        ]
        # Reset normal mode progress whenever the label set is changed.
        self.current_normal_mode_component_idx = 0
        self.normal_mode_completed_cycles = 0
        self._append_to_top(f"[Info] Active set: '{selected_group or 'None'}'. Components: {self.active_individual_components_list}. Normal mode progress reset.\n")
        self._update_ui_element_states()

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports: self.port_var.set(ports[0])
        else: self.port_var.set('')

    def connect_serial(self):
        if self.ser and self.ser.is_open: return
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Connection Error", "No port selected.")
            return
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
            self.alive = True
            self.reader_thread = threading.Thread(target=self.read_from_serial, daemon=True)
            self.reader_thread.start()
            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self.port_combo.config(state=tk.DISABLED)
            self.send_entry.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self._append_to_top(f"[Info] Connected to {port} @ {BAUD_RATE} baud.\n")
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {e}")

    def disconnect_serial(self):
        self.alive = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.5)
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self.port_combo.config(state='readonly')
        self.send_entry.delete(0, tk.END)
        self.send_entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self._append_to_top("[Info] Disconnected.\n")
        if self.record: self.toggle_record()

    def send_data(self, event=None):
        if not (self.ser and self.ser.is_open):
            self._append_to_top("[Warning] Not connected. Cannot send data.\n")
            return
        command = self.send_entry.get()
        if not command: return
        try:
            self.ser.write((command + '\n').encode('utf-8'))
            self._append_to_top(f"[Sent] > {command}\n")
            self.send_entry.delete(0, tk.END)
        except serial.SerialException as e:
            messagebox.showerror("Serial Write Error", f"Failed to send command: {e}")
            self._append_to_top(f"[Error] Failed to send command: {e}\n")

    def read_from_serial(self):
        while self.alive:
            try:
                if not (self.ser and self.ser.is_open):
                    time.sleep(0.1)
                    continue
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line and not self.sensor_paused:
                    self.master.after(0, self._process_serial_data, line)
            except serial.SerialException:
                if self.alive: self.master.after(0, self.disconnect_serial)
                break
            except Exception as e:
                self._append_to_top(f"[Critical] Unexpected error in serial reader: {e}\n")
                if self.alive: self.master.after(0, self.disconnect_serial)
                break

    def _finalize_recording_start(self, video_path: str):
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Camera Error", "Cannot start recording, camera is not available.")
            self.toggle_record()
            return
        self.record = True
        self._append_to_top(f"[Info] Recording started. Base: {self.output_filename_base}\n")
        h, w = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self.video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
            if not self.video_writer.isOpened():
                raise IOError(f"VideoWriter could not be opened for {video_path}")
        except Exception as e:
            messagebox.showerror("Video Error", f"Failed to start video writer: {e}")
            self.toggle_record()
            return
        self._append_to_top(f"[Info] Video recording to: {video_path}\n")
        
        # Auto-start first label if applicable
        if self.auto_label_mode_active.get():
            # In normal mode, we can kick it off right away.
            # For random mode, this is handled in _try_start_random_sequence_run
            if not self.random_labeling_mode_active.get():
                 self.master.after(500, self._start_next_label_automatically)

        self._update_ui_element_states()

    def _save_sensor_data(self):
        """Enhanced data saving with better error handling."""
        if not self.sensor_data or not self.output_filename_base or not self.current_session_date:
            self._append_to_top("[Warning] No data to save or missing filename/session info.\n")
            return
            
        data_dir = os.path.join(OUTPUT_DIR, self.current_session_date, "data")
        filepath = os.path.join(data_dir, f"{self.output_filename_base}_sensor.txt")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                # Write header if needed
                file.write("# Sensor Data with Labels\n")
                file.write("# Format: sensor_values,[Label]['label_name']\n")
                
                for item in self.sensor_data:
                    file.write(item + '\n')
                    
            self._append_to_top(f"[Info] Sensor data saved successfully: {filepath}\n")
            self._append_to_top(f"[Info] Total records saved: {len(self.sensor_data)}\n")
            
        except Exception as e:
            self._append_to_top(f"[Error] Failed to save sensor data: {e}\n")
            # Don't clear data if save failed
            return
            
        finally:
            # Only clear data after successful save
            if os.path.exists(filepath):
                self.sensor_data.clear()

    def toggle_pause_sensor(self):
        self.sensor_paused = not self.sensor_paused
        self.pause_button.config(text="Resume Sensor" if self.sensor_paused else "Pause Sensor")
        self._append_to_top(f"[Info] Sensor data processing {'paused' if self.sensor_paused else 'resumed'}.\n")

    def close(self):
        if messagebox.askokcancel("Quit", "Do you want to stop and save any active recording before quitting?"):
            self.alive = False
            if self.record: self.toggle_record()
            self.disconnect_serial()
            if self.cap.isOpened(): self.cap.release()
            self.master.destroy()

    def _append_to_top(self, text: str):
        if self.master.winfo_exists():
            self.top_text.config(state=tk.NORMAL)
            self.top_text.insert(tk.END, text)
            self.top_text.see(tk.END)
            self.top_text.config(state=tk.DISABLED)

    def _append_to_bottom(self, text: str):
        if self.master.winfo_exists():
            self.bottom_text.config(state=tk.NORMAL)
            self.bottom_text.insert(tk.END, text)
            self.bottom_text.see(tk.END)
            self.bottom_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x850")
    gui = SerialGUI(root)
    root.mainloop()