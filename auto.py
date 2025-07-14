
#!/usr/bin/env python
# compare_ocr.py  ——  在 PyCharm 右键 ▶ Run 即可
from __future__ import annotations

import re, logging, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2, fitz                       # PyMuPDF ≤ 1.22.3
import pandas as pd
from rapidfuzz import fuzz
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# ───────────── 全局常量 ──────────────
ID_RE      = re.compile(r"\d{17}[\dXx]")
SCHOOL_KWS = ["大学", "学院", "学校"]          # 初筛关键词
FONT_PATH  = "/System/Library/Fonts/STHeiti Medium.ttc"  # mac；Win 可改 simfang.ttf

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")

ocr = PaddleOCR(lang="ch", show_log=False)      # 单例，后面一直复用

# ───────────── 工具函数 ──────────────
def pdf_to_images(pdf: Path, dpi: int = 300) -> List[np.ndarray]:
    """PyMuPDF 把 PDF 每页渲染为 RGB ndarray"""
    doc = fitz.open(pdf)
    for p in doc:
        pix = p.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_vis(img_src, ocr_res, save_path: Path):
    """
    把 PaddleOCR 检测框 + 文本画到图片并保存
    img_src 可以是 ndarray 或 图片路径
    """
    if isinstance(img_src, (str, Path)):
        img = cv2.cvtColor(cv2.imread(str(img_src)), cv2.COLOR_BGR2RGB)
    else:
        img = img_src.copy()

    boxes  = [r[0]     for r in ocr_res]
    texts  = [r[1][0]  for r in ocr_res]
    scores = [r[1][1]  for r in ocr_res]

    vis = draw_ocr(img, boxes, texts, scores, font_path=FONT_PATH)
    Image.fromarray(vis).save(save_path, quality=95)

# ───────────── 字段抽取逻辑 ──────────────
def pick_school(texts: list[str]) -> str:
    """基于标签行 / 关键词提取学校名"""
    for i, t in enumerate(texts):
        if any(k in t for k in SCHOOL_KWS):
            if "名称" not in t:             # 直接就是值
                return t.strip()
            if i + 1 < len(texts):         # 标签行 → 下一行
                nxt = texts[i + 1].strip()
                if any(k in nxt for k in SCHOOL_KWS):
                    return nxt
    return ""

def pick_following(label_kw: str, texts: list[str]) -> str:
    """先找标签行返回下一行；否则 fallback 去标签"""
    for i, t in enumerate(texts):
        if t.strip().startswith(label_kw):
            if i + 1 < len(texts):
                return texts[i + 1].strip()
    for t in texts:                         # fallback
        if label_kw in t and len(t.strip()) > 4:
            return t.replace(label_kw, "").replace("：", "").strip()
    return ""

def extract(file: Path, vis_dir: Path | None = None) -> Dict[str, str]:
    """
    OCR 抽取字段；若 vis_dir 传入，则在该目录保存可视化 jpg。
    返回 dict: 身份证号 / 学校 / 专业
    """
    if not file or not file.exists():
        return {"身份证号": "", "学校": "", "专业": ""}

    is_pdf = file.suffix.lower() == ".pdf"
    pages  = pdf_to_images(file) if is_pdf else [str(file)]
    texts: List[str] = []

    for idx, pg in enumerate(pages, 1):
        res = ocr.ocr(pg, cls=True)[0]          # 单页 OCR
        texts.extend(r[1][0] for r in res)

        # —— 可视化保存 ——
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
            if is_pdf:
                save_vis(pg, res, vis_dir / f"{file.stem}_page{idx}.jpg")
            else:
                save_vis(pg, res, vis_dir / f"{file.stem}_vis.jpg")

    full_text = "\n".join(texts)

    return {
        "身份证号": (m := ID_RE.search(full_text)) and m.group(0) or "",
        "学校":     pick_school(texts),
        "专业":     pick_following("专业", texts),
    }

# ───────────── 比对函数 ──────────────
def eq(exp, act, fuzzy=False, th=90) -> bool:
    """精确 / 模糊比对"""
    exp, act = str(exp or "").strip(), str(act or "").strip()
    if not exp or not act:
        return False
    return fuzz.ratio(exp, act) >= th if fuzzy else exp == act

# ───────────── 主流程 ──────────────
def main():
    root  = Path(__file__).resolve().parent
    excel = root / "data.xlsx"
    docs  = root / "docs"
    if not excel.is_file():
        logging.error(f"未找到 {excel}")
        return
    if not docs.is_dir():
        logging.error(f"未找到 {docs} 文件夹")
        return

    vis_root = root / "vis_results"           # 可视化结果输出目录
    df = pd.read_excel(excel, dtype=str).fillna("")
    rows = []

    for _, r in df.iterrows():
        name = r["姓名"].strip()
        pdir = docs / name
        logging.info(f"[{name}] 处理中 …")

        idf = next(pdir.glob("id.*"),        None)
        baf = next(pdir.glob("bachelor.*"),  None)
        msf = next(pdir.glob("master.*"),    None)

        id_info = extract(idf, vis_root / name)
        ba_info = extract(baf, vis_root / name)
        ms_info = extract(msf, vis_root / name)

        row_res = {
            "姓名": name,
            "身份证匹配":  eq(r["身份证号"], id_info["身份证号"]),
            "本科学校匹配": eq(r["本科学校"],  ba_info["学校"], fuzzy=True),
            "本科专业匹配": eq(r["本科专业"],  ba_info["专业"], fuzzy=True),
            "硕士学校匹配": eq(r["硕士学校"],  ms_info["学校"], fuzzy=True),
            "硕士专业匹配": eq(r["硕士专业"],  ms_info["专业"], fuzzy=True),
            "OCR_身份证":  id_info["身份证号"],
            "OCR_本科学校": ba_info["学校"],
            "OCR_本科专业": ba_info["专业"],
            "OCR_硕士学校": ms_info["学校"],
            "OCR_硕士专业": ms_info["专业"],
        }
        rows.append(row_res)

        # —— 控制台打印 ——
        print(f"\n")
        print(f"  OCR_身份证   : {row_res['OCR_身份证']}")
        print(f"  OCR_本科学校 : {row_res['OCR_本科学校']}")
        print(f"  OCR_本科专业 : {row_res['OCR_本科专业']}")
        print(f"  OCR_硕士学校 : {row_res['OCR_硕士学校']}")
        print(f"  OCR_硕士专业 : {row_res['OCR_硕士专业']}")
        print(f"  匹配结果     : 身份证={row_res['身份证匹配']} | "
              f"本科={row_res['本科学校匹配'] and row_res['本科专业匹配']} | "
              f"硕士={row_res['硕士学校匹配'] and row_res['硕士专业匹配']}")
        print("-" * 60)

    # —— 导出结果 ——
    out = root / "compare_result.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    logging.info(f"✅ 完成！比对结果 → {out.name}")
    logging.info(f"✅ OCR 可视化已保存 → {vis_root.relative_to(root)}")

if __name__ == "__main__":
    main()
