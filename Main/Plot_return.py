import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox

MODEL_DIR = "../Model"
MODEL_EXTENSIONS = (".pth", ".pt")


def load_checkpoint(file_path: str):
    """Load checkpoint safely and return dict or None."""
    resolved_path = file_path
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join('.', file_path)
    try:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.load(resolved_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"找不到Model檔案: {file_path}")
    except Exception as err:
        print(f"載入Model失敗: {err}")
    return None


def extract_iterations(checkpoint):
    """Return sorted list of (iteration_number, data_dict)."""
    iterations = []
    for key, data in checkpoint.items():
        if not isinstance(key, str) or not key.startswith('iteration:'):
            continue
        try:
            iteration = int(key.split(':', 1)[1])
        except ValueError:
            continue
        iterations.append((iteration, data))
    return sorted(iterations, key=lambda item: item[0])


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Prompt user via GUI dialog; fallback to CLI if needed."""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno("繪圖模式", message)
        return default if result is None else result
    except Exception as err:
        print(f"GUI 選單開啟失敗 ({err})，改用命令列輸入。")
    finally:
        if root is not None:
            root.destroy()

    hint = "Y/n" if default else "y/N"
    while True:
        user = input(f"{message} ({hint}): ").strip().lower()
        if not user:
            return default
        if user in ("y", "yes"):
            return True
        if user in ("n", "no"):
            return False
        print("請輸入 Y 或 N。")


def list_model_files():
    """Return list of (display_name, relative_path) for available model files."""
    base_dir = os.path.join('.', MODEL_DIR)
    if not os.path.isdir(base_dir):
        return []

    model_files = []
    for name in sorted(os.listdir(base_dir)):
        if name.lower().endswith(MODEL_EXTENSIONS):
            rel_path = os.path.join(MODEL_DIR, name)
            model_files.append((name, rel_path))
    return model_files


def select_model_file():
    """Open a file dialog for picking a model file."""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        initial_dir = os.path.join('.', MODEL_DIR)
        if not os.path.isdir(initial_dir):
            initial_dir = '.'
        file_path = filedialog.askopenfilename(
            title="選擇模型檔案",
            initialdir=initial_dir,
            filetypes=[("Torch Model", "*.pth *.pt"), ("所有檔案", "*.*")]
        )
    except Exception as err:
        print(f"開啟檔案選擇視窗失敗: {err}")
        file_path = None
    finally:
        if root is not None:
            root.destroy()

    if not file_path:
        return None, None
    return os.path.basename(file_path), file_path


def combine_rewards(iterations):
    segments = []
    for iteration, data in iterations:
        rewards = data.get('reward') if isinstance(data, dict) else None
        if rewards is None or len(rewards) == 0:
            continue
        rewards_array = np.asarray(rewards, dtype=float)
        segments.append(rewards_array)

    if not segments:
        return None

    return np.concatenate(segments)


def collect_model_rewards(display_name, rel_path):
    checkpoint = load_checkpoint(rel_path)
    if not checkpoint:
        return None

    iterations = extract_iterations(checkpoint)
    if not iterations:
        print("此檔案沒有 Return 資料")
        return None

    combined = combine_rewards(iterations)
    if combined is None or combined.size == 0:
        print("沒有可用的 reward 資料")
        return None

    return display_name, combined


def plot_models(model_reward_list):
    if not model_reward_list:
        print("沒有可繪製的模型資料")
        return

    num_models = len(model_reward_list)
    fig_height = max(3, 2.5 * num_models)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, fig_height), sharex=False)
    if num_models == 1:
        axes = [axes]

    for ax, (display_name, rewards) in zip(axes, model_reward_list):
        steps = np.arange(len(rewards))
        ax.plot(steps, rewards, linewidth=1.2)
        ax.set_ylim(bottom=-100)
        ax.set_title(display_name, fontsize=12, loc='left')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step')
    fig.suptitle('Actor Return', fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_models_from_entries(model_entries):
    collected = []
    for idx, (display_name, rel_path) in enumerate(model_entries, start=1):
        print(f"[{idx}/{len(model_entries)}] {display_name}")
        result = collect_model_rewards(display_name, rel_path)
        if result is not None:
            collected.append(result)
    if collected:
        plot_models(collected)
        names = ', '.join(name for name, _ in collected)
        print(f"繪製完成：{names}")


def main():
    model_files = list_model_files()
    message = "是否繪製 Model 資料夾內所有模型？\n\n是（選擇所有 Model）\n否（選擇單一 Model）"

    if prompt_yes_no(message, default=True):
        if not model_files:
            print("Model 資料夾內找不到可用的模型檔案")
            return
        plot_models_from_entries(model_files)
    else:
        display, rel_path = select_model_file()
        if not rel_path:
            print("未選擇模型")
            return
        result = collect_model_rewards(display, rel_path)
        if result is None:
            return
        plot_models([result])


if __name__ == "__main__":
    main()