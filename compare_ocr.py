
#!/usr/bin/env python
# compare_ocr.py  ——  在 PyCharm 右键 ▶ Run 即可
from __future__ import annotations
import re, logging, warnings, shutil, difflib
from pathlib import Path
from typing import Dict, List
from tkinter import filedialog, Tk
import os
import numpy as np
import cv2, fitz                     # PyMuPDF ≤ 1.22.3
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import sys
import platform
from pathlib import Path
def _default_font() -> str | None:
    """Return a reasonable CJK font path, or None."""
    if platform.system() == "Windows":
        # 常见宋体/仿宋/黑体；按存在优先返回
        for fn in ("simfang.ttf", "simsun.ttc", "msyh.ttc", "msyh.ttf"):
            p = Path(r"C:\Windows\Fonts") / fn
            if p.exists():
                return str(p)
        return None
    elif platform.system() == "Darwin":
        # macOS
        for fn in (
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ):
            if Path(fn).exists():
                return fn
        return None
    else:
        # Linux
        for fn in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                   "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
            if Path(fn).exists():
                return fn
        return None
# ───────────── 全局常量 ──────────────
ID_RE       = re.compile(r"\d{17}[\dXx]")          # 身份证正则
SCHOOL_KWS  = ["大学", "学院", "学校"]             # 学校关键词
ID_INLINE_STOP_KWS = ["性别", "民族", "出生", "出生日期", "住址",
                      "公民身份号码", "公民", "号码"]  # 行内字段终止关键词（新增“出生日期”）
FONT_PATH = _default_font()

EDU_KEYWORDS = [
    "博士研究生", "博士",
    "硕士研究生", "研究生", "硕士",
    "大学本科", "本科",
    "大专", "专科",
    "中专", "高中", "职高", "技校",
]

# 学位关键词（授予学位 / 学士硕士博士）
DEGREE_KEYWORDS = [
    "博士学位", "博士",
    "硕士学位", "硕士",
    "学士学位", "学士", "双学士",
]
EDU_CANON = {
    "博士研究生": "博士",
    "博士": "博士",
    "硕士研究生": "研究生",
    "研究生": "研究生",
    "硕士": "研究生",
    "大学本科": "本科",
    "本科": "本科",
    "大专": "大专",
    "专科": "大专",
    "中专": "中专",
    "高中": "高中",
    "职高": "高中",
    "技校": "中专",
}

DEGREE_CANON = {
    "博士学位": "博士",
    "博士": "博士",
    "硕士学位": "硕士",
    "硕士": "硕士",
    "学士学位": "学士",
    "学士": "学士",
    "双学士": "学士",  # 如需单独处理可改
}
EXACT_FIELDS = ["身份证号", "性别", "出生日期", "民族", "学位"]
FUZZY_FIELDS = ["籍贯", "学历", "出生地"]

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")

ocr = PaddleOCR(lang="ch", show_log=False)         # 单例

# ───────────── OCR 工具函数 ──────────────
def pdf_to_images(pdf: Path, dpi: int = 300) -> List[np.ndarray]:
    """PyMuPDF 把 PDF 每页渲染成 RGB ndarray"""
    doc = fitz.open(pdf)
    for p in doc:
        pix = p.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def pick_edu_keyword(texts: List[str]) -> str:
    """遍历 OCR 文本，按 EDU_KEYWORDS 顺序找学历关键词并标准化。"""
    full = "".join(texts)
    for kw in EDU_KEYWORDS:
        if kw in full:
            return EDU_CANON.get(kw, kw)
    # 再逐行（冗余防守，避免 join 后被断字）
    for t in texts:
        for kw in EDU_KEYWORDS:
            if kw in t:
                return EDU_CANON.get(kw, kw)
    return ""

def app_root() -> Path:
    """Return directory where the program is running.
    Works for PyInstaller (sys._MEIPASS) and normal script."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent

def pick_degree_keyword(texts: List[str]) -> str:
    """遍历 OCR 文本，按 DEGREE_KEYWORDS 顺序找学位关键词并标准化。"""
    full = "".join(texts)
    for kw in DEGREE_KEYWORDS:
        if kw in full:
            return DEGREE_CANON.get(kw, kw)
    for t in texts:
        for kw in DEGREE_KEYWORDS:
            if kw in t:
                return DEGREE_CANON.get(kw, kw)
    # 常见“工学学士”“文学硕士”等：抓末尾学士/硕士/博士
    joined = "".join(texts)
    m = re.search(r"(博士|硕士|学士)", joined)
    if m:
        return DEGREE_CANON.get(m.group(1), m.group(1))
    return ""


def save_vis(img_src, ocr_res, save_path: Path):
    """把 OCR 检测框 + 文本画在图片并保存"""
    img = (cv2.cvtColor(cv2.imread(str(img_src)), cv2.COLOR_BGR2RGB)
           if isinstance(img_src, (str, Path)) else img_src.copy())
    boxes  = [r[0] for r in ocr_res]
    texts  = [r[1][0] for r in ocr_res]
    scores = [r[1][1] for r in ocr_res]
    vis = draw_ocr(img, boxes, texts, scores, font_path=FONT_PATH)
    Image.fromarray(vis).save(save_path, quality=95)

# ───────────── 字段抽取 ──────────────
def pick_school(texts: list[str]) -> str:
    """提取学校名"""
    for i, t in enumerate(texts):
        if any(k in t for k in SCHOOL_KWS):
            if "名称" not in t:
                return t.strip()
            if i + 1 < len(texts):
                nxt = texts[i + 1].strip()
                if any(k in nxt for k in SCHOOL_KWS):
                    return nxt
    return ""

DATE_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")

def pick_birth(texts: List[str]) -> str:
    """
    从 OCR 列表中提取出生日期。
    优先解析同一行（'出生2000年6月2日' / '出生日期：2000-06-02'），
    找不到再 fallback 到标签行下一行模式。
    """
    # ① 行内截取
    for t in texts:
        if "出生" in t:
            # 去掉全角空格
            line = t.replace("\u3000", "").strip()
            # 截 label 后面
            if "出生日期" in line:
                seg = line.split("出生日期", 1)[1]
            else:
                seg = line.split("出生", 1)[1]
            seg = seg.replace("：", "").replace(":", "").strip()
            # 切掉下一标签（住址等）
            for stop in ID_INLINE_STOP_KWS:
                if stop in seg:
                    seg = seg.split(stop, 1)[0]
            seg = seg.strip()
            if seg:
                return seg

    # ② 没有行内，走原来的“下一行”逻辑
    val = pick_following("出生日期", texts)
    if not val:
        val = pick_following("出生", texts)
    return val

def normalize_date(s: str) -> str:
    """
    将各种日期形式标准化为 YYYY-MM-DD；无法解析则返回原串去空格。
    支持：2000年6月2日 / 2000-06-02 / 2000/6/2 / 20000602 / 2000年06月02日 等。
    """
    if not s:
        return ""
    s = str(s).strip().translate(DATE_DIGIT_TRANS)
    # 常见分隔符统一
    s_clean = re.sub(r"[./／_年月日\s]+", "-", s)
    s_clean = s_clean.strip("-")
    # 纯 8 位数字
    m = re.fullmatch(r"(\d{4})[-]?(\d{2})[-]?(\d{2})", s_clean)
    if m:
        y, mth, d = m.groups()
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    # 非纯数字：抓 3 组数
    m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s_clean)
    if m:
        y, mth, d = m.groups()
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    return s  # fallback 原样
def pick_inline_segment(line: str, label: str, stops: List[str]) -> str:
    """
    从单行中提取 label 后面的内容，遇到 stops 中任一标签就截断。
    例: line='性别男民族汉' → pick_inline_segment(...,'性别') 返回 '男'
    """
    if label not in line:
        return ""
    seg = line.split(label, 1)[1]  # label 之后的部分
    for s in stops:
        if s != label and s in seg:
            seg = seg.split(s, 1)[0]
    return seg.replace("：", "").replace(":", "").strip()

def pick_following(label_kw: str, texts: list[str]) -> str:
    """先找‘标签行 → 下一行’；否则从同一行截取"""
    for i, t in enumerate(texts):
        if t.strip().startswith(label_kw):
            return texts[i + 1].strip() if i + 1 < len(texts) else ""
    for t in texts:
        if label_kw in t and len(t.strip()) > len(label_kw):
            return (t.replace(label_kw, "")
                     .replace("：", "")
                     .replace(":", "")
                     .strip())
    return ""

# ───────────── 身份证行内字段（性别 / 民族 / 出生日期）解析 ──────────────
def extract_id_inline_fields(texts: List[str]) -> Dict[str, str]:
    """
    扫描 OCR 文本列表，解析身份证常见同行格式：
    '性别男民族汉' / '性别 女 民族 汉族 出生 2000年06月02日' / ...
    """
    sex = nat = birth = ""

    for line in texts:
        # 性别
        if "性别" in line and not sex:
            seg = pick_inline_segment(line, "性别", ID_INLINE_STOP_KWS)
            if seg:
                # 常规：取第一个男女字
                m = re.search(r"[男女]", seg)
                sex = m.group(0) if m else seg[:1]

        # 民族
        if "民族" in line and not nat:
            seg = pick_inline_segment(line, "民族", ID_INLINE_STOP_KWS)
            if seg:
                # 去掉可能重复的“族”
                nat = seg.replace("族族", "族").strip()

        # 出生 / 出生日期
        if ("出生日期" in line or "出生" in line) and not birth:
            # 优先长标签
            seg = pick_inline_segment(line, "出生日期", ID_INLINE_STOP_KWS)
            if not seg:
                seg = pick_inline_segment(line, "出生", ID_INLINE_STOP_KWS)
            if seg:
                # 正则规范化日期
                m = re.search(r"\d{4}[年./-]?\s*\d{1,2}[月./-]?\s*\d{1,2}日?", seg)
                birth = m.group(0) if m else seg.strip()

    # 若同行未取到，用正则全局兜底
    joined = "\n".join(texts)
    if not sex:
        m = re.search(r"性别[:：]?\s*([男女])", joined)
        if m: sex = m.group(1)
    if not nat:
        m = re.search(r"民族[:：]?\s*(\S{1,5})", joined)
        if m: nat = m.group(1)
    if not birth:
        m = re.search(r"出生(?:日期)?[:：]?\s*(\d{4}[年./-]?\s*\d{1,2}[月./-]?\s*\d{1,2}日?)", joined)
        if m: birth = m.group(1)

    return {"性别": sex, "民族": nat, "出生日期": birth}

# ───────────── 核心抽取入口 ──────────────
def extract(file: Path, vis_dir: Path | None = None) -> Dict[str, str]:
    """OCR 抽取全部字段"""
    if not file or not file.exists():
        return {f: "" for f in ["身份证号", "学校", "专业",
                                "性别", "出生日期", "民族", "籍贯", "出生地", "学历", "学位"]}

    is_pdf = file.suffix.lower() == ".pdf"
    pages  = pdf_to_images(file) if is_pdf else [str(file)]
    texts: List[str] = []

    for idx, pg in enumerate(pages, 1):
        res = ocr.ocr(pg, cls=True)[0]
        texts.extend(r[1][0] for r in res)

        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_vis(pg, res,
                     vis_dir / f"{file.stem}_{'page'+str(idx) if is_pdf else 'vis'}.jpg")

    full_text = "\n".join(texts)

    # —— 新增：身份证同行字段抽取 —— #
    inline = extract_id_inline_fields(texts)

    # 原 find_gender() 改为内部函数，保留你的逻辑 + 正则兜底
    def find_gender() -> str:
        # 优先 inline
        if inline.get("性别"):
            return inline["性别"]
        # 精准正则
        m = re.search(r"性别[:：]?\s*([男女])", full_text)
        if m:
            return m.group(1)
        # 全局兜底（避免文件中出现两个字误判：谁先出现取谁）
        pos_m = full_text.find("男")
        pos_f = full_text.find("女")
        if pos_m != -1 and (pos_f == -1 or pos_m < pos_f):
            return "男"
        if pos_f != -1:
            return "女"
        return ""

    # 出生日期（优先标签行，次要 inline）
    birth_raw = pick_birth(texts)
    birth_val = birth_raw or inline.get("出生日期", "")
    # 民族（优先标签行，次要 inline）
    nat_val = inline.get("民族", "")
    if not nat_val:
        for _line in texts:
            if "民族" in _line:
                nat_val = pick_inline_segment(_line, "民族", ID_INLINE_STOP_KWS)
                if nat_val:
                    break
    if not nat_val:
        nat_val = pick_following("民族", texts)
    if not nat_val:
        for _line in texts:
            if "性别" in _line and "民族" not in _line:
                tail = _line.split("性别", 1)[1]
                tail = tail.replace("：", "").replace(":", "").strip()
                tail = tail.lstrip("男女 ")
                cand = tail[:3].strip()
                cand = cand.replace("性", "").replace("别", "").replace("男", "").replace("女", "").strip()
                nat_val = cand
                break
    return {
        "身份证号": (m := ID_RE.search(full_text)) and m.group(0) or "",
        "学校":     pick_school(texts),
        "专业":     pick_following("专业", texts),

        # 新增字段（增强提取）
        "性别":       find_gender(),
        "出生日期":   birth_val,
        "民族":       nat_val,

        # 下列保持原逻辑（不改）
        "籍贯":     pick_following("籍贯", texts),
        "出生地":   pick_following("出生地", texts),
        "学历":     pick_edu_keyword(texts) or pick_following("学历", texts),
        "学位":     pick_degree_keyword(texts) or pick_following("学位", texts),
    }

# ───────────── 比对函数 ──────────────
def eq(exp, act, fuzzy=False, th=90) -> bool:
    exp, act = str(exp or "").strip(), str(act or "").strip()
    if not exp or not act:
        return False
    if fuzzy:
        return difflib.SequenceMatcher(None, exp, act).ratio() * 100 >= th
    return exp == act

def ask_user_inputs() -> tuple[Path, Path]:
    """弹窗选 Excel 和资料文件夹"""
    root = Tk(); root.withdraw()
    excel = filedialog.askopenfilename(title="选择 data.xlsx",
                                       filetypes=[("Excel 文件", "*.xlsx")])
    if not excel: exit("❌未选择 Excel")
    docs = filedialog.askdirectory(title="选择资料文件夹（docs）")
    if not docs: exit("❌未选择资料文件夹")
    return Path(excel), Path(docs)

BIRTH_PAT = re.compile(r"(19|20\d{2}|\d{4})\D+(\d{1,2})\D+(\d{1,2})")
def eq_date(exp, act) -> bool:
    return normalize_excel_date(exp) == normalize_excel_date(act)
def find_birth(full_text: str) -> str:
    m = BIRTH_PAT.search(full_text)
    if not m:
        return ""
    y = int(m.group(1)[-4:])  # 容错：以最后4位当年
    mm = int(m.group(2))
    dd = int(m.group(3))
    return f"{y:04d}-{mm:02d}-{dd:02d}"

def _find_doc(pdir: Path, stem: str) -> Path | None:
    """Return first match for id/bachelor/master/lvli ignoring case & multi-ext."""
    if not pdir.exists():
        return None
    patterns = [f"{stem}.*", f"{stem.upper()}.*", f"{stem.capitalize()}.*"]
    exts = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
    # 先匹配 stem，不行再 stem+所有扩展
    for pat in patterns:
        hits = list(pdir.glob(pat))
        if hits:
            return hits[0]
    # fallback: 遍历所有文件，看文件名包含 stem（不区分大小写）
    for ext in exts:
        for f in pdir.glob(ext):
            if stem.lower() in f.stem.lower():
                return f
    return None

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

# ───────────── 主流程 ──────────────
def main():
    proj_root = app_root()
    excel, docs = ask_user_inputs()          # 弹窗交互
    try:
        shutil.copy(excel, proj_root / "data.xlsx")
    except Exception:
        pass

    vis_root = proj_root / "vis_results"
    df   = pd.read_excel(excel, dtype=str).fillna("")
    # 出生日期列归一化
    if "出生日期" in df.columns:
        df["出生日期"] = df["出生日期"].apply(normalize_excel_date)

    rows = []

    for _, r in df.iterrows():
        name = r["姓名"].strip()
        pdir = docs / name
        logging.info(f"[{name}] 处理中 …")

        idf = _find_doc(pdir, "id")
        baf = _find_doc(pdir, "bachelor")
        msf = _find_doc(pdir, "master")
        lvf = _find_doc(pdir, "lvli")     # 🆕 履历表 (可选)

        id_info = extract(idf, vis_root / name)
        ba_info = extract(baf, vis_root / name)
        ms_info = extract(msf, vis_root / name)
        lv_info = extract(lvf, vis_root / name)     # 新表字段


        row = {
            "姓名": name,

            # —— 身份证信息 ——（来源优先 id_info，没有则用 lv_info）
            "身份证匹配": eq(r["身份证号"], id_info["身份证号"] or lv_info["身份证号"]),
            "性别匹配":   eq(r["性别"],   lv_info["性别"]),
            "出生日期匹配": eq_date(r["出生日期"], id_info["出生日期"] or lv_info["出生日期"]),

            "民族匹配":   eq(r["民族"],   id_info["民族"]),
            "学位匹配":   eq(r["学位"],   lv_info["学位"]),

            # —— 模糊字段 ——
            "籍贯匹配":   eq(r["籍贯"],   lv_info["籍贯"],   fuzzy=True),
            "学历匹配":   eq(r["学历"],   lv_info["学历"],   fuzzy=True),
            "出生地匹配": eq(r["出生地"], lv_info["出生地"], fuzzy=True),

            # —— 学历报告比对 ——
            "本科学校匹配": eq(r["本科学校"], ba_info["学校"], fuzzy=True),
            "本科专业匹配": eq(r["本科专业"], ba_info["专业"], fuzzy=True),
            "硕士学校匹配": eq(r["硕士学校"], ms_info["学校"], fuzzy=True),
            "硕士专业匹配": eq(r["硕士专业"], ms_info["专业"], fuzzy=True),

            # —— OCR 字段回填 ——
            **{f"OCR_{k}": v for k, v in {
                **id_info, **ba_info, **ms_info, **lv_info}.items()}
        }
        rows.append(row)

    # 控制台结果（保持你原始格式）
    print(f"匹配结果：")
    for key in [
        "身份证", "性别", "出生日期", "民族", "学位",
        "籍贯", "学历", "出生地",
        "本科学校", "本科专业",
        "硕士学校", "硕士专业"
    ]:
        print(f"  {key:<8}匹配: {row.get(f'{key}匹配', '')}")

    print(f"  本科整体匹配: {row['本科学校匹配'] and row['本科专业匹配']}")
    print(f"  硕士整体匹配: {row['硕士学校匹配'] and row['硕士专业匹配']}")
    print("-" * 60)

    # 导出
    out = proj_root / "compare_result.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    logging.info(f"✅ 完成，比对结果 → {out.name}")
    logging.info(f"✅ OCR 可视化 → {vis_root.relative_to(proj_root)}")

if __name__ == "__main__":
    main()
