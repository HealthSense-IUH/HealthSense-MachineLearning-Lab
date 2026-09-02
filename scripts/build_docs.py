# -*- coding: utf-8 -*-
"""Gộp toàn bộ docs của HealthSense ML thành 1 file docs/index.html duy nhất.

Cách tái tạo khi sửa sơ đồ:
1. Sửa spec: docs/component/components.architecture.json
   hoặc docs/component/pipeline_v4.dataflow.json
2. Render lại 2 file HTML trung gian bằng archify (validate + deliver showcase):
   components.html và pipeline_v4.html, đặt vào docs/component/
3. Khôi phục docs/component/GIAI_THICH_THUAT_NGU.md nếu cần sửa glossary
   (nội dung hiện tại đã nằm trong index.html, tab "Giải thích thuật ngữ")
4. Chạy: venv/Scripts/python scripts/build_docs.py
5. Xóa các file trung gian, chỉ giữ index.html

LƯU Ý: script này hiện KHÔNG chạy được — cả 3 file đầu vào ở bước 2-3
(components.html, pipeline_v4.html, GIAI_THICH_THUAT_NGU.md) đều đã bị xóa
khỏi repo. Chỉ còn 2 file spec .json. Muốn dựng lại index.html thì phải
render 2 file HTML trung gian trước, và viết lại phần glossary.
"""
import os

import html
import markdown

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(_HERE), 'docs')
COMPONENT = os.path.join(DOCS, 'component')

def _read(name):
    """Đọc file trung gian trong docs/component/, báo lỗi rõ ràng nếu thiếu."""
    path = os.path.join(COMPONENT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Thiếu file đầu vào: docs/component/{name}\n'
            f'Xem hướng dẫn ở đầu file này — cần render lại các file trung '
            f'gian từ 2 spec .json trước khi dựng index.html.')
    with open(path, encoding='utf-8') as f:
        return f.read()


components_html = _read('components.html')
pipeline_html = _read('pipeline_v4.html')
glossary_md = _read('GIAI_THICH_THUAT_NGU.md')

glossary_html = markdown.markdown(glossary_md, extensions=["tables"])
# Bỏ dòng link cuối trỏ tới file cũ
glossary_html = glossary_html.replace(
    '<p><em>Xem sơ đồ pipeline tương tác tại <a href="pipeline_v4.html"><code>docs/pipeline_v4.html</code></a>.</em></p>', '')

comp_escaped = html.escape(components_html, quote=True)
pipe_escaped = html.escape(pipeline_html, quote=True)

page = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HealthSense ML — Tài liệu</title>
<style>
  :root {
    --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b;
    --accent: #0d9488; --border: #e2e8f0;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8; --border: #334155; }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.6; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8rem; margin-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; }
  .badge { display: inline-block; background: var(--accent); color: #fff; border-radius: 6px; font-size: .72rem; padding: 2px 8px; vertical-align: middle; margin-left: 8px; }
  nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; position: sticky; top: 0; background: var(--bg); padding: 10px 0; z-index: 10; border-bottom: 1px solid var(--border); }
  nav button { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 16px; font-size: .9rem; cursor: pointer; }
  nav button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  section { display: none; }
  section.active { display: block; }
  .stats { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; margin-bottom: 24px; }
  .stats h2 { font-size: 1.05rem; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; margin: 12px 0; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  iframe { width: 100%; height: 88vh; border: 1px solid var(--border); border-radius: 12px; background: #fff; }
  .fsbtn { margin: 8px 0 16px; background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 6px 14px; cursor: pointer; font-size: .85rem; }
  .glossary h1 { font-size: 1.4rem; margin: 18px 0 10px; }
  .glossary h2 { font-size: 1.15rem; margin: 22px 0 10px; color: var(--accent); }
  .glossary hr { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
  .glossary p, .glossary li { font-size: .92rem; }
  .glossary ul { margin: 8px 0 8px 22px; }
  .glossary strong { color: var(--accent); }
  .foot { color: var(--muted); font-size: .82rem; margin-top: 24px; }
  .links a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>🩺 HealthSense ML <span class="badge">v4</span></h1>
  <p class="sub">Phát hiện Rung Nhĩ (AFib) từ tín hiệu PPG — pipeline chống data leakage, kiểm định 3 tầng. Toàn bộ tài liệu trong 1 file.</p>

  <nav>
    <button data-tab="overview" class="active">📊 Tổng quan &amp; Kết quả</button>
    <button data-tab="components">🏗️ Kiến trúc hệ thống</button>
    <button data-tab="pipeline">📐 Pipeline ML</button>
    <button data-tab="glossary">📖 Giải thích thuật ngữ</button>
  </nav>

  <section id="overview" class="active">
    <div class="stats">
      <h2>Kết quả tóm tắt (không data leakage)</h2>
      <table>
        <tr><th>Bài kiểm định</th><th>Model</th><th>Accuracy</th><th>Recall</th><th>ROC-AUC</th></tr>
        <tr><td>LOSO MIMIC — mức bệnh nhân (35 người)</td><td>Random Forest</td><td>94.3%</td><td><b>100%</b> (0 FN)</td><td>0.931</td></tr>
        <tr><td>Cross-dataset: MIMIC → AFDB</td><td>XGBoost</td><td>94.6%</td><td>97.0%</td><td>0.987</td></tr>
        <tr><td>Cross-dataset: AFDB → MIMIC</td><td>Random Forest</td><td>91.9%</td><td>97.4%</td><td>0.976</td></tr>
        <tr><td><b>Pooled LOSO 60 bệnh nhân (model cuối)</b></td><td><b>XGBoost</b></td><td><b>95.9%</b></td><td><b>97.2%</b></td><td><b>0.988</b></td></tr>
      </table>
    </div>
    <div class="stats">
      <h2>Dữ liệu &amp; mô hình</h2>
      <table>
        <tr><th>Thành phần</th><th>Chi tiết</th></tr>
        <tr><td>MIMIC PERform AF</td><td>35 bệnh nhân PPG 125 Hz (19 AFib / 16 Normal), 4.130 cửa sổ 30s</td></tr>
        <tr><td>MIT-BIH AFDB</td><td>25 bệnh nhân ECG 250 Hz (AF kịch phát), 28.903 cửa sổ từ QRS annotation</td></tr>
        <tr><td>Đặc trưng</td><td>13 HRV (time / frequency / nonlinear), loại nhóm LF không đủ tin cậy ở 30s</td></tr>
        <tr><td>Model triển khai</td><td><code>models/final/healthsense_afib_pipeline.pkl</code> — XGBoost + StandardScaler, train gộp 60 bệnh nhân cân bằng nguồn; đặc tả trong <code>model_card.json</code></td></tr>
        <tr><td>Tái lập</td><td>4 lệnh: <code>run_v4_extraction.py</code> → <code>run_v4_benchmark.py</code> → <code>run_cross_dataset.py</code> → <code>run_final_model.py</code></td></tr>
      </table>
      <p class="links">Chi tiết đầy đủ: <a href="../README.md">README.md</a> · <a href="../models/final/model_card.json">model_card.json</a></p>
    </div>
    <p class="foot">⚠️ Mô hình phục vụ nghiên cứu/sàng lọc, chưa kiểm định trên PPG cổ tay MAX30102 — không phải thiết bị chẩn đoán y tế. Cập nhật: 30/08/2026.</p>
  </section>

  <section id="components">
    <button class="fsbtn" onclick="document.getElementById('if-comp').requestFullscreen()">⛶ Toàn màn hình</button>
    <iframe id="if-comp" srcdoc="__COMPONENTS__"></iframe>
  </section>

  <section id="pipeline">
    <button class="fsbtn" onclick="document.getElementById('if-pipe').requestFullscreen()">⛶ Toàn màn hình</button>
    <iframe id="if-pipe" srcdoc="__PIPELINE__"></iframe>
  </section>

  <section id="glossary">
    <div class="stats glossary">__GLOSSARY__</div>
  </section>
</div>
<script>
  document.querySelectorAll('nav button').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('nav button').forEach(function(b){ b.classList.remove('active'); });
      document.querySelectorAll('section').forEach(function(s){ s.classList.remove('active'); });
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
</script>
</body>
</html>
"""

page = page.replace("__COMPONENTS__", comp_escaped)
page = page.replace("__PIPELINE__", pipe_escaped)
page = page.replace("__GLOSSARY__", glossary_html)

out = os.path.join(DOCS, 'index.html')
with open(out, "w", encoding="utf-8") as f:
    f.write(page)
print(f"OK: {out} ({len(page)/1024:.0f} KB)")
