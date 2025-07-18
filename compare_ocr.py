
#!/usr/bin/env python
# compare_ocr.py  ——  在 PyCharm 右键 ▶ Run 即可
from __future__ import annotations

import sys
import re
import logging
import warnings
import shutil
import difflib
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from tkinter import filedialog, Tk

import numpy as np
import cv2
import fitz  # PyMuPDF ≤ 1.22.3 更稳
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 全局常量
# ─────────────────────────────────────────────────────────────────────────────
ID_RE = re.compile(r"\d{17}[\dXx]")            # 身份证号
SCHOOL_KWS = ["大学", "学院", "学校"]          # 学校关键词

# 在身份证行内截断的关键词（提取性别/民族等字段时使用）
ID_INLINE_STOP_KWS = [
    "性别", "民族", "出生", "出生日期", "出生地", "出生地", "住址",
    "公民身份号码", "公民", "号码", "学位", "学历", "籍贯"
]

# 穷举学历 / 学位关键词（可按需扩展；全都小写比较）
DEGREE_KWS = [
    "学士", "硕士", "博士", "士", "master", "phd", "md", "mba"
]
EDU_KWS = [
    "本科", "研究生", "硕士研究生", "博士研究生", "专科", "大专",
    "高中", "中专", "初中", "小学", "研究所"
]

# 动态字体路径（跨平台）
if sys.platform.startswith("win"):
    _fp = Path(__file__).with_name("simfang.ttf")  # 放一个中文字体文件与 exe 同级
    FONT_PATH: Optional[str] = str(_fp) if _fp.is_file() else None
else:  # mac / linux
    _fp = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    FONT_PATH = str(_fp) if _fp.is_file() else None

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")

# 初始化 PaddleOCR（中文）
ocr = PaddleOCR(lang="ch", show_log=False)


# ─────────────────────────────────────────────────────────────────────────────
# OCR 基础工具
# ─────────────────────────────────────────────────────────────────────────────
def pdf_to_images(pdf: Path, dpi: int = 300) -> Iterable[np.ndarray]:
    """PyMuPDF 将 PDF 每页渲染成 RGB ndarray。"""
    doc = fitz.open(pdf)
    for p in doc:
        pix = p.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        # PyMuPDF 输出顺序是 RGB，不是 BGR；PaddleOCR 接受 numpy RGB 也可以
        yield img


def save_vis(img_src, ocr_res, save_path: Path):
    """
    将 OCR 检测框 + 文本画到图片并保存。
    img_src 可以是 ndarray 或 文件路径。
    """
    if isinstance(img_src, (str, Path)):
        bgr = cv2.imread(str(img_src))
        if bgr is None:
            return
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_src.copy()

    boxes = [r[0] for r in ocr_res]
    texts = [r[1][0] for r in ocr_res]
    scores = [r[1][1] for r in ocr_res]

    if FONT_PATH:
        vis = draw_ocr(img, boxes, texts, scores, font_path=FONT_PATH)
    else:
        vis = draw_ocr(img, boxes, texts, scores)

    Image.fromarray(vis).save(save_path, quality=95)


# ─────────────────────────────────────────────────────────────────────────────
# 文本抽取辅助
# ─────────────────────────────────────────────────────────────────────────────
def pick_school(texts: List[str]) -> str:
    """从 OCR 行列表中提取学校名（识别到“学校名称”则返回下一行）。"""
    for i, t in enumerate(texts):
        if any(k in t for k in SCHOOL_KWS):
            if "名称" not in t:
                return t.strip()
            # 标签行 → 下一行若仍包含学校关键词，取之
            if i + 1 < len(texts):
                nxt = texts[i + 1].strip()
                if any(k in nxt for k in SCHOOL_KWS):
                    return nxt
    return ""


def pick_inline_segment(line: str, label: str, stops: List[str]) -> str:
    """
    从同一行中提取 label 之后的文本，遇到 stops 截断。
    例: "性别男民族汉" pick_inline_segment(line,"民族") -> "汉"
    """
    if label not in line:
        return ""
    seg = line.split(label, 1)[1]
    for s in stops:
        if s != label and s in seg:
            seg = seg.split(s, 1)[0]
    return seg.replace("：", "").replace(":", "").strip()


_BIRTH_PATTERNS = [
    re.compile(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?"),   # 2000年06月02日 / 2000-06-02
    re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})"),             # 2000.6.2
]


def _normalize_date(y: str, m: str, d: str) -> str:
    try:
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except Exception:
        return ""

BIRTH_PAT = re.compile(r"(19|20\d{2}|\d{4})\D+(\d{1,2})\D+(\d{1,2})")

def find_birth(full_text: str) -> str:
    m = BIRTH_PAT.search(full_text)
    if not m:
        return ""
    y = int(m.group(1)[-4:])  # 容错：以最后4位当年
    mm = int(m.group(2))
    dd = int(m.group(3))
    return f"{y:04d}-{mm:02d}-{dd:02d}"

def eq_date(exp, act) -> bool:
    return normalize_excel_date(exp) == normalize_excel_date(act)

def extract_birth(text: str) -> str:
    """从任意字符串中抓出生日期模式。"""
    for pat in _BIRTH_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize_date(*m.groups())
    return ""


def pick_following(label_kw: str, texts: List[str]) -> str:
    """找标签行并取下一行；否则取同行 label 后内容。"""
    for i, t in enumerate(texts):
        if t.strip().startswith(label_kw):
            return texts[i + 1].strip() if i + 1 < len(texts) else ""
    for t in texts:
        if label_kw in t and len(t.strip()) > len(label_kw):
            return t.replace(label_kw, "").replace("：", "").strip()
    return ""


def locate_degree_edu(full_text: str) -> Dict[str, str]:
    """
    从全文文本粗略找学历/学位（只要出现关键词就取最长那段）。
    返回 {"学历":..., "学位":...}
    """
    lower = full_text.lower()
    edu, deg = "", ""

    # 学历：按 EDU_KWS 顺序匹配；取首次出现（可按优先级排序）
    for kw in EDU_KWS:
        if kw.lower() in lower:
            edu = kw
            break

    # 学位：尝试中文学位词
    for kw in DEGREE_KWS:
        if kw.lower() in lower:
            # 匹配中文展示（避免全部小写返回）
            if kw in full_text:
                deg = kw
            else:
                deg = kw  # fallback
            break

    return {"学历": edu, "学位": deg}


def parse_excel_date(val) -> str:
    """
    将 Excel 读出的日期（文本 / 数字 / pandas Timestamp）标准化 YYYY-MM-DD。
    """
    if val == "" or pd.isna(val):
        return ""
    if isinstance(val, str):
        return extract_birth(val) or val.strip()
    # 数字（Excel 序列）
    if isinstance(val, (int, float)):
        try:
            # Excel 序列起点 1899-12-30；用 pandas 转换
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(float(val), unit="D")
        except Exception:
            return str(val)
    # pandas 时间戳
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)

def normalize_excel_date(v: object) -> str:
    """把 Excel 单元格值转成 YYYY-MM-DD 字符串。
    支持：Excel 序列号、datetime、各种可解析的日期字符串、包含中文年月日的字符串。
    非法/空 -> ''"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if not s:
        return ""

    # 纯数字：可能是 Excel 序列号
    if s.isdigit():
        try:
            dt = pd.to_datetime(float(s), unit="d", origin="1899-12-30", errors="raise")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # 普通日期解析
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m-%d")

    # 中文格式: 2000年6月2日 / 2000年06月02日 等
    m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", s)
    if m:
        y, mth, d = map(int, m.groups())
        return f"{y:04d}-{mth:02d}-{d:02d}"

    return s  # 实在不行原样返回

# ─────────────────────────────────────────────────────────────────────────────
# 主 OCR 抽取函数
# ─────────────────────────────────────────────────────────────────────────────
def extract(file: Path, vis_dir: Path | None = None) -> Dict[str, str]:
    """
    OCR 文件并抽取字段。
    返回：
      身份证号 / 学校 / 专业 / 性别 / 出生日期 / 民族 / 籍贯 / 出生地 / 学历 / 学位
    """
    empty = {f: "" for f in
             ["身份证号", "学校", "专业", "性别", "出生日期", "民族",
              "籍贯", "出生地", "学历", "学位"]}
    if not file or not file.exists():
        return empty

    is_pdf = file.suffix.lower() == ".pdf"
    pages = pdf_to_images(file) if is_pdf else [str(file)]
    texts: List[str] = []

    # page OCR
    for idx, pg in enumerate(pages, 1):
        res = ocr.ocr(pg, cls=True)[0]  # 每页
        texts.extend(r[1][0] for r in res)

        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{file.stem}_page{idx}.jpg" if is_pdf else f"{file.stem}_vis.jpg"
            save_vis(pg, res, vis_dir / out_name)

    full_text = "\n".join(texts)

    # ----------------- 行扫描（身份证图 / 履历表） -----------------
    gender = ""
    birth = ""
    nat = ""
    jg = ""
    birthplace = ""

    for line in texts:
        ln = line.strip()

        # 性别：行含“性别”，或行中“性别男民族汉”
        if not gender:
            if "性别" in ln:
                seg = pick_inline_segment(ln, "性别", ID_INLINE_STOP_KWS)
                if seg:
                    gender = seg[0]  # 取首字
                else:
                    # 找下一个行
                    gender = pick_following("性别", texts) or gender

        # 民族：行含“民族”
        if "民族" in ln and not nat:
            seg = pick_inline_segment(ln, "民族", ID_INLINE_STOP_KWS)
            if seg:
                # 去掉“族族”等重复，保留到“族”字为止
                # 常见格式 “汉” / “汉族”
                if "族" in seg:
                    seg = seg.split("族", 1)[0] + "族"
                nat = seg.strip()

        # 出生日期：行含“出生”；若 inline 拿不到有效日期，再下一行
        if ("出生" in ln or "出生日期" in ln) and not birth:
            seg = pick_inline_segment(ln, "出生日期", ID_INLINE_STOP_KWS) \
                  or pick_inline_segment(ln, "出生", ID_INLINE_STOP_KWS)
            birth = extract_birth(seg) if seg else ""
            if not birth:
                # 下一行
                tmp = pick_following("出生日期", texts) or pick_following("出生", texts)
                birth = extract_birth(tmp) if tmp else ""

        # 籍贯
        if "籍贯" in ln and not jg:
            jg = pick_inline_segment(ln, "籍贯", ID_INLINE_STOP_KWS) or \
                 pick_following("籍贯", texts)

        # 出生地
        if ("出生地" in ln or "出生于" in ln) and not birthplace:
            birthplace = pick_inline_segment(ln, "出生地", ID_INLINE_STOP_KWS) or \
                         pick_following("出生地", texts)

    # 学历/学位：全文关键词粗定位（不依赖行结构）
    loc = locate_degree_edu(full_text)
    edu = loc["学历"]
    deg = loc["学位"]

    # 汇总
    info = {
        "身份证号": (m := ID_RE.search(full_text)) and m.group(0) or "",
        "学校": pick_school(texts),
        "专业": pick_following("专业", texts),
        "性别": gender,
        "出生日期": birth,
        "民族": nat,
        "籍贯": jg,
        "出生地": birthplace,
        "学历": edu,
        "学位": deg,
    }
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 比对
# ─────────────────────────────────────────────────────────────────────────────
def eq(exp, act, fuzzy=False, th=90) -> bool:
    exp, act = str(exp or "").strip(), str(act or "").strip()
    if not exp or not act:
        return False
    if fuzzy:
        return difflib.SequenceMatcher(None, exp, act).ratio() * 100 >= th
    return exp == act


# ─────────────────────────────────────────────────────────────────────────────
# 文件选择（交互）
# ─────────────────────────────────────────────────────────────────────────────
def ask_user_inputs() -> tuple[Path, Path]:
    root = Tk()
    root.withdraw()               # 先隐藏主窗口
    root.attributes('-topmost', True)   # ✨ 让对话框永远置顶
    # --------- 下面不变 ----------
    excel = filedialog.askopenfilename(
        title="选择 data.xlsx", filetypes=[("Excel 文件", "*.xlsx")]
    )
    if not excel:
        exit("❌未选择 Excel")
    docs = filedialog.askdirectory(title="选择资料文件夹（docs）")
    if not docs:
        exit("❌未选择资料文件夹")
    return Path(excel), Path(docs)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    proj_root = Path.cwd()

    excel_path, docs_path = ask_user_inputs()

    # 把选中的 Excel 复制到项目根目录（方便用户看到）
    try:
        shutil.copy(excel_path, proj_root / "data.xlsx")
    except Exception:
        pass

    vis_root = proj_root / "vis_results"
    df = pd.read_excel(excel_path, dtype=str).fillna("")

    # Excel 规范化（出生日期字段标准化）
    if "出生日期" in df.columns:
        df["出生日期"] = df["出生日期"].apply(parse_excel_date).astype(str)

    rows = []

    for _, r in df.iterrows():
        name = str(r["姓名"]).strip()
        pdir = docs_path / name
        logging.info(f"[{name}] 处理中 …")

        # 匹配文件：id.*, bachelor.*, master.*, lvli.*（忽略大小写）
        def glob_one(pattern: str):
            # 支持多扩展名
            g = list(pdir.glob(pattern))
            return g[0] if g else None

        idf = glob_one("id.*")
        baf = glob_one("bachelor.*")
        msf = glob_one("master.*")
        lvf = glob_one("lvli.*")  # 可选（履历）

        id_info = extract(idf, vis_root / name)
        ba_info = extract(baf, vis_root / name)
        ms_info = extract(msf, vis_root / name)
        lv_info = extract(lvf, vis_root / name) if lvf else {k: "" for k in id_info}

        # 填充缺失（身份证未抽到就用履历）
        sid = id_info["身份证号"] or lv_info["身份证号"]
        sex = id_info["性别"] or lv_info["性别"]
        bdt = id_info["出生日期"] or lv_info["出生日期"]
        nat = id_info["民族"] or lv_info["民族"]

        row = {
            "姓名": name,

            # 身份证区
            "身份证匹配": eq(r.get("身份证号", ""), sid),
            "性别匹配": eq(r.get("性别", ""), sex),
            "出生日期匹配": eq_date(r["出生日期"], id_info["出生日期"] or lv_info["出生日期"]),
            "民族匹配": eq(r.get("民族", ""), nat),
            "学位匹配": eq(r.get("学位", ""), lv_info.get("学位", "")),

            # 模糊字段
            "籍贯匹配": eq(r.get("籍贯", ""), lv_info.get("籍贯", ""), fuzzy=True),
            "学历匹配": eq(r.get("学历", ""), lv_info.get("学历", ""), fuzzy=True),
            "出生地匹配": eq(r.get("出生地", ""), lv_info.get("出生地", ""), fuzzy=True),

            # 学历报告
            "本科学校匹配": eq(r.get("本科学校", ""), ba_info.get("学校", ""), fuzzy=True),
            "本科专业匹配": eq(r.get("本科专业", ""), ba_info.get("专业", ""), fuzzy=True),
            "硕士学校匹配": eq(r.get("硕士学校", ""), ms_info.get("学校", ""), fuzzy=True),
            "硕士专业匹配": eq(r.get("硕士专业", ""), ms_info.get("专业", ""), fuzzy=True),
        }

        # OCR 字段回填（附带来源信息供人工核对）
        row.update({
            "OCR_身份证": sid,
            "OCR_性别": sex,
            "OCR_出生日期": bdt,
            "OCR_民族": nat,
            "OCR_籍贯": lv_info.get("籍贯", ""),
            "OCR_出生地": lv_info.get("出生地", ""),
            "OCR_学历": lv_info.get("学历", ""),
            "OCR_学位": lv_info.get("学位", ""),
            "OCR_本科学校": ba_info.get("学校", ""),
            "OCR_本科专业": ba_info.get("专业", ""),
            "OCR_硕士学校": ms_info.get("学校", ""),
            "OCR_硕士专业": ms_info.get("专业", ""),
        })
        rows.append(row)

        # 控制台打印（快速查看）
        print("\n----------------------------------------")
        print(f"姓名: {name}")
        print(f"  OCR_身份证   : {row['OCR_身份证']}")
        print(f"  OCR_性别     : {row['OCR_性别']}")
        print(f"  OCR_出生日期 : {row['OCR_出生日期']}")
        print(f"  OCR_民族     : {row['OCR_民族']}")
        print(f"  OCR_本科学校 : {row['OCR_本科学校']}")
        print(f"  OCR_本科专业 : {row['OCR_本科专业']}")
        print(f"  OCR_硕士学校 : {row['OCR_硕士学校']}")
        print(f"  OCR_硕士专业 : {row['OCR_硕士专业']}")
        print("  匹配结果: "
              f"身份证={row['身份证匹配']} | "
              f"性别={row['性别匹配']} | "
              f"出生日期={row['出生日期匹配']} | "
              f"民族={row['民族匹配']} | "
              f"本科={(row['本科学校匹配'] and row['本科专业匹配'])} | "
              f"硕士={(row['硕士学校匹配'] and row['硕士专业匹配'])}")

    # 导出
    out = proj_root / "compare_result.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    logging.info(f"✅ 完成，比对结果 → {out.name}")
    logging.info(f"✅ OCR 可视化 → {vis_root.relative_to(proj_root)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
