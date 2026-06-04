# AAAI-27 投稿日 cheat sheet

**Deadline**: 2026-07-28 (UTC-12 / Anywhere on Earth)
**Portal**: AAAI-27 OpenReview(具体 URL 通常 deadline 前 4-6 周开放)
**预跑时间**:final compile + 双检 + 上传 + 系统填表 ~1.5 h,建议 deadline 前 24-48 h 提交

---

## 📦 上传文件(4 项,~3 MB 总)

### Required(必上传)

| # | 文件 | 绝对路径 | 大小 |
|:---:|---|---|:---:|
| 1 | **Main paper PDF** | `/home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/aaai27/main_v2_aaai27.pdf` | 556 KB |
| 2 | **Supplementary PDF** | `/home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/aaai27/supplementary_dtmargin.pdf` | 277 KB |

### Optional(若 portal 允许)

| # | 文件 | 绝对路径 | 大小 |
|:---:|---|---|:---:|
| 3 | **Demo video**(30s overview)| `/home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/figures/fig_webots_va_atacom/supplementary_demo_30s.mp4` | 573 KB |
| 4 | **Source bundle**(若要求 LaTeX source)| `/home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/aaai27/` 整目录 + `figures/` + `references.bib` zipped | ~1 MB |

### 不需上传

- `repro_checklist.tex` — 已 `\input` 进 main paper page 9-10(随 main PDF 一起提交)
- Audit docs(STATUS / PAPER_AUDIT / TABLE1_AUDIT 等)— 主仓 internal 用,不投稿

---

## 🎬 投稿日 step-by-step

### Step 1:final compile(投稿前最后 1 次)

```bash
cd /home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/aaai27
rm -f *.aux *.bbl *.blg *.log *.out
pdflatex -interaction=nonstopmode main_v2_aaai27.tex
bibtex main_v2_aaai27
pdflatex -interaction=nonstopmode main_v2_aaai27.tex
pdflatex -interaction=nonstopmode main_v2_aaai27.tex

pdflatex -interaction=nonstopmode supplementary_dtmargin.tex
pdflatex -interaction=nonstopmode supplementary_dtmargin.tex

# 核 page count + errors
pdfinfo main_v2_aaai27.pdf | grep Pages       # 期望 10
pdfinfo supplementary_dtmargin.pdf | grep Pages   # 期望 3
```

### Step 2:最后合规 self-check(可一键跑)

```bash
cd /home/miclsirr/work/miclmasters/czz-safe-manifold/paper/aaai_2027/aaai27
# 核 body ≤ 7 + Refs page 7-8 + Checklist page 9-10
pdftotext main_v2_aaai27.pdf - | awk -v RS='\f' '
  /Conclusion/ && !c { print "Conclusion: page " NR; c=1 }
  /^Achiam, J\.;/ && !r { print "Refs 1st  : page " NR; r=1 }
  /Reproducibility Checklist/ && !rc { print "Checklist : page " NR; rc=1 }'

# 核 fonts embedded
pdffonts main_v2_aaai27.pdf | tail -n +3 | awk '{print $4}' | sort -u
#   ↑ 应只显示 'yes'

# 核 PDF metadata 无 author leak
pdfinfo main_v2_aaai27.pdf | grep -iE '^(Author|Title|Creator|Producer)'
#   ↑ 期望:Author/Title 字段空,Creator=TeX,Producer=pdfTeX
```

### Step 3:portal 上传

1. 登录 AAAI-27 OpenReview submission portal
2. 上传 `main_v2_aaai27.pdf` 作 Main Paper
3. 上传 `supplementary_dtmargin.pdf` 作 Supplementary Material
4. (可选)上传 `supplementary_demo_30s.mp4` 作 Supplementary Video
5. (若 portal 要求 source)打包 `aaai27/` zip 上传

### Step 4:portal 填表(submission metadata)

- **Title**:`VA-ATACOM: Velocity-Augmented Constraint Manifolds for Safe Reinforcement Learning at Coarse Control Rates`
- **Track**:Main Technical Track
- **Topics**(AAAI keywords,选 1-3):
  - Safe RL / Constrained RL
  - Robot Learning
  - Safety in AI
- **Authors / Affiliations**:portal 单独填(本身不在 paper PDF 里,因双盲)
- **Reviewer conflicts**:填 group + 共同作者 institutions
- **Reproducibility Checklist**:portal 可能问 24 题(答案与 `repro_checklist.tex` 一致即可)

---

## ✅ 投稿前 final self-check(7 项,~5 min)

| # | 检查 | 期望 |
|:---:|---|---|
| 1 | Main body ≤ 7 pages | ✅ Conclusion + Refs 起均 page 7 |
| 2 | Total Main PDF 10 pages | ✅ |
| 3 | Supp 3 pages | ✅ |
| 4 | PDF Author/Title metadata empty | ✅ no leak |
| 5 | 所有 fonts embedded | ✅ 27/27 |
| 6 | `\author{Anonymous Submission}` + `\affiliations{Anonymous Institution}` | ✅ |
| 7 | Reproducibility Checklist 24 答案全填 | ✅ 24/24 yes/NA |

详见 `AAAI_COMPLIANCE_AUDIT.md`(14/14 hard requirements pass)。

---

## 📞 投稿后

- AAAI-27 review timeline 历年:**Phase 1(8 月)reviewers 看;Phase 2(10-11 月)讨论;Decision 12 月**
- 期间 author response window(~9 月)— 可看 reviewer comments 并 rebut,paper 不能改
- 若 conditional accept:**camera-ready 可加到 8 pages**(AAAI 历年惯例),supplementary 不限
- camera-ready 时换 author / affiliations 真名,`[final]` option(去 [submission])

---

## 🗂 备忘 — 主要 artefact 总览

```
paper/aaai_2027/
├── aaai27/                          ← 投稿用(本目录全打包可上传 source)
│   ├── main_v2_aaai27.{tex,pdf}     ★ main paper
│   ├── supplementary_dtmargin.{tex,pdf}  ★ supplementary
│   ├── repro_checklist.tex          ★ (\input 进 main)
│   ├── aaai2027.{sty,bst}           ← 官方 sty
│   ├── references.bib  → ../references.bib
│   └── figures/        → ../figures
├── figures/fig_webots_va_atacom/supplementary_demo_30s.mp4  ★ optional video
├── references.bib                   ← 30 entries
└── *.md                             ← 14 个 audit docs(不投稿)
```

---

## ⚠️ 关键 reminders

1. **主仓不 push GitHub**(per `feedback_main_repo_no_push`)— paper 全本地。**投稿时手动下载 PDF 不用 git push**
2. **`aaai2027.sty` 内部已设 `\bibliographystyle`** — 不要自己写 `\bibliographystyle{}`(bibtex 报 "another \bibstyle" 错)
3. **`[submission]` 选项**让 `\pdfinfo` 变 no-op + 隐藏作者 + 加 review notice — 投稿时必须保留
4. **deadline 时区是 UTC-12 / AoE** — 北京时间 deadline 是 **2026-07-29 早上 8 点**(比 UTC-12 早 20 小时)

---

最后更新 2026-06-04(P0+P1+P2 + 各续 全 12 task done + 16/16 AAAI compliance + 中稿估 66-80%)。
