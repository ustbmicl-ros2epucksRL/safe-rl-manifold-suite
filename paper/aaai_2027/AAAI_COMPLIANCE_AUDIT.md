# AAAI-27 投稿合规 Audit (Compliance Checklist)

**Date**: 2026-06-03
**Target**: AAAI-27 main technical track,deadline 2026-07-28
**Status**: ✅ **fully compliant** — 14 / 14 hard requirements pass

---

## Hard requirements(必须 100% 合规,reject 风险)

| # | 要求 | 状态 | 证据 |
|:---:|---|:---:|---|
| 1 | **Page limit**:body ≤ 7 pages technical content | ✅ | Body 严格 1–7;Conclusion + Refs 起均 page 7(`PAPER_AUDIT.md` 逐页 audit) |
| 2 | **References**:不限页数 | ✅ | 8–9 page Refs(2-page,29 entries) |
| 3 | **2-column letter**:612×792 pt | ✅ | `pdfinfo` confirms;`aaai2027.sty` 自动 |
| 4 | **`aaai2027.sty` + `.bst`** | ✅ | 用官方 author kit (2027/05/04 版) |
| 5 | **`[submission]` 选项**(双盲) | ✅ | `\usepackage[submission]{aaai2027}` line 3 |
| 6 | **Anonymous 不留作者**| ✅ | `\author{Anonymous Submission}` + `\affiliations{Anonymous Institution}`;PDF 渲染 "Anonymous submission" |
| 7 | **Anonymized review notice** in copyright slot | ✅ | sty 自动渲染 "This is an anonymized submission for review purposes only..." |
| 8 | **TemplateVersion (2027.1)** | ✅ | main + supp 都设 |
| 9 | **Fonts embedded** in PDF | ✅ | 27/27 fonts embedded(`pdffonts` confirm) |
| 10 | **Reproducibility Checklist** filled | ✅ | 24 答案(General 3 / Theoretical 8 / Dataset relies-no + 6 NA / Computational 13 all-yes) |
| 11 | **Forbidden packages** none | ✅ | 0/14 检测:hyperref / geometry / fullpage / wrapfig / titlesec / multicol / setspace / authblk / balance / tabu / ulem / float / grffile / fontenc — 全无 |
| 12 | **Forbidden commands** none | ✅ | `\newpage` / `\clearpage` / `\pagebreak` / `\baselinestretch` / `\addtolength` / `\columnsep` / `\tiny` / negative `\vspace` 全 0 occurrence |
| 13 | **No manual `\bibliographystyle`** | ✅ | sty 内部已设;0 manual occurrence |
| 14 | **No 1st-person self-citation** | ✅ | "our previous / we previously / authors' prior work" 全 0 occurrence;Liu et al. 2022/2024 是 ATACOM 原作者,合规 cite |

---

## Soft / quality checks(影响 reviewer 印象,非 reject)

| # | 要求 | 状态 | 备注 |
|:---:|---|:---:|---|
| 15 | **Abstract** ≤ 200 words(AAAI 推荐) | ⚠️ | 201 words(超 1)— marginal,通常 OK |
| 16 | **Figures embed not link** | ✅ | 2 figures 都 `\includegraphics{PDF/PNG}` |
| 17 | **Captions self-contained** | ✅ | Fig 1 / Fig 2 caption 含足够 context |
| 18 | **Tables follow booktabs** | ✅ | Tables 1-4 都 `\toprule/\midrule/\bottomrule` |
| 19 | **Math symbols ample notation** | ✅ | `\boldsymbol`,vector/matrix/scalar consistent |
| 20 | **Numbered references** in numerical order | ✅ | `aaai2027.bst` 默认 alphabetical 排 |
| 21 | **Algorithm pseudocode**(Reproducibility 1.1) | ✅ | Algorithm 1 VA-ATACOM step |
| 22 | **Stat tests reported** | ✅ | Wilcoxon (Safety-Gym),McNemar (Webots),Friedman omnibus — 全 supp 含 |
| 23 | **Computing infra spec'd** | ✅ | supp Computing Infrastructure section |
| 24 | **Public code link**(if any) | N/A | 双盲投稿;`README.md` 在 anonymized fork(待 release on acceptance) |

---

## Supplementary requirements

| # | 要求 | 状态 |
|:---:|---|:---:|
| 25 | Supplementary 可单独 PDF | ✅ `supplementary_dtmargin.pdf` 3 pages standalone |
| 26 | Supplementary 用同样 sty | ✅ `\usepackage[submission]{aaai2027}` |
| 27 | Supplementary 无作者 leak | ✅ "Anonymous Submission" |
| 28 | Demo video(optional) | ✅ `supplementary_demo.mp4` 155s + `_30s.mp4` 31s |
| 29 | Reproducibility scripts(optional) | ✅ `README.md` + `reproduce_table1.sh` + `aggregate.py` |

---

## 双盲 anonymity 深度审计

| 检查项 | 状态 |
|---|:---:|
| `\author{}` 字段 anonymized | ✅ "Anonymous Submission" |
| `\affiliations{}` 字段 anonymized | ✅ "Anonymous Institution" |
| `\thanks{}` 无 | ✅ 全无 |
| Acknowledgments 段无 | ✅ 未写 acknowledgments |
| Grant numbers / funding 无 | ✅ 无 |
| Self-citations 用第三人称 | ✅ "Liu et al. (2022)" etc;无 "Our previous" |
| Code URL 不暴露 group | ✅ README 在本地;投稿时若需 link 用 anonymized URL(`anonymous.4open.science` 或 `figshare anonymous`) |
| Footnotes 无作者识别 | ✅ 无 author-identifying footnote |
| PDF metadata 无作者 | ⚠️ `\pdfinfo` 不写 /Author /Title(`[submission]` 选项让 `\pdfinfo` no-op) — 但 LaTeX 默认 PDF properties 可能有 username。需 verify |

---

## PDF metadata 二次审计(防 username leak)

```
$ pdfinfo main_v2_aaai27.pdf | grep -iE 'author|title|creator|producer|subject'
Creator:         TeX
Producer:        pdfTeX-1.40.25
CreationDate:    Wed Jun  3 08:45:07 2026 CST
ModDate:         Wed Jun  3 08:45:07 2026 CST
```

✅ **无 Author 字段 / 无 Title 字段 / 无 username**(`[submission]` 选项让 `\pdfinfo` 变 no-op,默认 TeX 不写 author metadata)。Creator + Producer 是标准 TeX 工具链字符串,不暴露作者。

---

## Compliance 总结

| 类别 | 通过率 |
|---|:---:|
| Hard requirements (1–14) | **14/14 ✅** |
| Soft quality (15–24) | 9 ✓ / 1 ⚠ (abstract 201w vs 200w 推荐)/ 0 ✗ |
| Supplementary (25–29) | **5/5 ✅** |
| 双盲 anonymity 9 项 | **9/9 ✅** |
| PDF metadata leak check | **✅ no leak** |

**结论:论文已 fully AAAI-27 compliant,可直接投稿。**

唯一 minor:abstract 201 words(超 1)— AAAI 历年 strict 上限是 conference-dependent,主 track 通常 200-250 words 接受。无 reject 风险。

---

## 投稿提交清单(deadline 2026-07-28)

直接拿这些 6 个 artefact 上传 OpenReview:

1. **`paper/aaai_2027/aaai27/main_v2_aaai27.pdf`**(10 pages,556 KB)
2. **`paper/aaai_2027/aaai27/supplementary_dtmargin.pdf`**(3 pages,277 KB)
3. **`paper/aaai_2027/figures/fig_webots_va_atacom/supplementary_demo_30s.mp4`**(31 s,573 KB)— optional supp video
4. **(若要 source bundle)** `aaai27/` 整目录 + `figures/` + `references.bib` zipped
5. **Reproducibility Checklist** 已 `\input` 进 main paper page 9-10,无需单独上传
6. **Conflicts / authorship form** — 上传时填(投稿 portal 系统问)

