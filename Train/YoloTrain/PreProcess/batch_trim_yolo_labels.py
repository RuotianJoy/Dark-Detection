#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_trim_yolo_labels.py

功能
------
批量裁剪 YOLO 六列标签文件，只保留前 5 列：
class_id  x_center  y_center  width  height

- 支持递归遍历子目录
- 自动跳过格式异常的行，并记录日志
- 可选“就地覆盖”或“输出到新目录”

用法
------
python batch_trim_yolo_labels.py \
       --src  path/to/labels_with_temp \
       --dst  path/to/trimmed_labels

若希望直接覆盖原文件，可省略 --dst 或令其等于 --src。
"""

import argparse
import logging
from pathlib import Path

def process_label_file(src_path: Path, dst_path: Path) -> None:
    """
    读取单个标签文件，仅保留前 5 列写入目标文件。
    """
    with src_path.open("r", encoding="utf-8") as fin, \
         dst_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                logging.warning(f"{src_path}:{lineno} 少于 5 列，已跳过")
                continue
            # 若包含第 6 列（如温度），截断之
            fout.write(" ".join(parts[:5]) + "\n")

def batch_trim(src_dir: Path, dst_dir: Path) -> None:
    """
    遍历 src_dir 下所有 .txt 标签文件，输出到 dst_dir。
    """
    txt_files = list(src_dir.rglob("*.txt"))
    if not txt_files:
        logging.warning("未找到任何 .txt 标签文件")
        return

    for txt in txt_files:
        rel_path = txt.relative_to(src_dir)
        dst_file = dst_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        process_label_file(txt, dst_file)
        logging.info(f"已处理 → {dst_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch trim YOLO 6-column labels to 5-column format."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="含 6 列标签的根目录")
    parser.add_argument("--dst", type=Path, default=None,
                        help="输出目录；若省略则覆盖原文件")
    args = parser.parse_args()

    # 若未指定 dst，则在 src 基础上原地覆盖
    dst_root = args.dst or args.src

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    logging.info(f"源目录: {args.src.resolve()}")
    logging.info(f"目标目录: {dst_root.resolve()}")
    batch_trim(args.src, dst_root)
    logging.info("全部处理完毕。")
