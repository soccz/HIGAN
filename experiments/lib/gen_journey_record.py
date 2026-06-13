"""Generate the FULL experiment-record page (complete, not summarized) for
/projects/higan-generalization/ — every track + every control version + data
manifest, straight from on-disk data."""
import json, glob, re, html
from pathlib import Path

PAPER = Path("/mnt/20t/study/HIGAN/paper")
OUT = Path("/mnt/20t/soccz.github.io/projects/higan-generalization")

# ---------- experiment ledger (56 tracks) ----------
agg = (PAPER / "experiments/out/_aggregate_summary.txt").read_text()
tracks = []
for b in re.split(r'\n(?=\[Track )', agg):
    if not b.startswith("[Track"): continue
    head = b.split("\n", 1)[0].strip("[]")
    body = b.split("\n", 1)[1].strip() if "\n" in b else ""
    tracks.append((head, body))

def verdict(head, body):
    h = (head + " " + body).lower()
    if "spearman r = none" in h or "argmax-in-canonical rate: 0.125" in h: return "fail"
    if "park reproduction" in h and "0.04" in h: return "fail"
    if "sal_corr_mean=+0.056" in h or "real lsun" in h or "encoder transfer" in h: return "fail"
    if any(k in h for k in ["multi-clip","dinov2","noise robustness","wall-clock",
        "higher-order","8x14","cluster labels","truncation","intrinsic dim","fd validation"]): return "ok"
    if "cross-domain k=2" in h and "0.778" in h: return "mixed"
    return "mixed"

ledger = []
for head, body in tracks:
    v = verdict(head, body)
    ledger.append(
        f'<tr class="r-{v}"><td><span class="tag {v}">{v.upper()}</span></td>'
        f'<td class="tk">{html.escape(head)}</td>'
        f'<td><pre>{html.escape(body) or "—"}</pre></td></tr>')  # FULL, no truncation

# ---------- control campaign (95 versions) ----------
def vnum(f):
    m = re.search(r'control_v(\d+)', f); return int(m.group(1)) if m else 0
ctrl = []
for f in sorted(glob.glob(str(PAPER/"experiments/protocols/control_v*.json")), key=vnum):
    try: d = json.load(open(f))
    except: continue
    base = Path(f).stem.replace("control_", "")
    purpose = (d.get("purpose") or d.get("description") or "").strip()
    exps = d.get("experiments"); nexp = len(exps) if isinstance(exps, list) else 1
    n = vnum(f)
    phase = ("controller" if n <= 30 else "feasible" if n <= 40
             else "predictive" if n <= 71 else "stress")
    if "audit" in base or "evidence_table" in base or "readiness" in base: phase = "audit"
    # pull per-experiment keys (claim_tested) so each version shows what it actually ran
    sub = ""
    if isinstance(exps, list) and exps:
        items = []
        for e in exps:
            key = e.get("key", "") or e.get("protocol_key", "")
            claim = e.get("claim_tested", "") or e.get("claim", "")
            if key or claim:
                items.append(f'<li><code>{html.escape(str(key))}</code> {html.escape(str(claim))}</li>')
        if items:
            sub = '<details><summary>실행 ' + str(len(items)) + '개</summary><ul>' + "".join(items) + '</ul></details>'
    ctrl.append(
        f'<tr class="c-{phase}"><td class="vn">v{n}</td><td><span class="ph ph-{phase}">{phase}</span></td>'
        f'<td class="ne">{nexp}</td><td class="pp">{html.escape(purpose)}{sub}</td></tr>')  # FULL purpose

n_out = len(list((PAPER/"experiments/out").glob("*/")))
n_dev = len(list((PAPER.parent/"higan_dev/scripts").glob("[0-9]*.py")))

# ---------- assemble page ----------
PAGE = """<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8"><link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HIGAN 일반화 — 전체 실험 기록</title>
<meta name="description" content="HIGAN을 cross-architecture 미분기하로 일반화하려던 연구 여정의 전체 실험 기록 — 56개 분석 트랙, 95개 control 검증 버전, 210개 결과 폴더. 성공과 실패를 과장 없이 전부 남긴다.">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Noto+Serif+KR:wght@600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#fafaf9;--bg2:#f5f5f4;--card:#fff;--tx:#1c1917;--tx2:#57534e;--mut:#a8a29e;--ac:#44403c;--acd:rgba(68,64,60,.07);--red:#b91c1c;--grn:#15803d;--amb:#b45309;--bd:#e7e5e4;--bdl:#d6d3d1;}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:'Noto Sans KR',sans-serif;background:var(--bg);color:var(--tx);line-height:1.72;-webkit-font-smoothing:antialiased;}
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:.9rem 2rem;background:rgba(250,250,249,.88);backdrop-filter:blur(20px);border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;}
nav a.home{font-family:'Noto Serif KR',serif;font-weight:700;color:var(--tx);text-decoration:none;font-size:.92rem;}
nav .nr a{color:var(--tx2);text-decoration:none;font-size:.8rem;margin-left:1.2rem;}
nav .nr a:hover{color:var(--ac);}
.container{max-width:1040px;margin:0 auto;padding:0 2rem;}
.hero{min-height:78vh;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:7rem 2rem 3.5rem;background:radial-gradient(ellipse at 30% 40%,rgba(120,113,108,.04) 0%,transparent 55%);}
.hl{font-size:.7rem;letter-spacing:.28em;text-transform:uppercase;color:var(--mut);border:1px solid var(--bdl);border-radius:100px;padding:.4rem 1.1rem;display:inline-block;margin-bottom:1.6rem;}
.hero h1{font-family:'Noto Serif KR',serif;font-size:clamp(1.9rem,5vw,3rem);font-weight:700;line-height:1.32;margin-bottom:1.4rem;}
.hero h1 em{font-style:normal;color:var(--amb);}
.hsub{font-size:1rem;color:var(--tx2);max-width:720px;margin:0 auto 2.4rem;line-height:1.85;}
.hstats{display:flex;justify-content:center;gap:2.2rem;flex-wrap:wrap;margin-bottom:2.4rem;}
.hs{text-align:center;}.hsn{font-family:'Noto Serif KR',serif;font-size:2rem;font-weight:700;color:#292524;display:block;line-height:1.2;}
.hsl{font-size:.72rem;color:var(--mut);letter-spacing:.04em;}
.hb{display:flex;justify-content:center;gap:.8rem;flex-wrap:wrap;}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.65rem 1.3rem;border:1px solid var(--bdl);border-radius:8px;color:var(--tx2);text-decoration:none;font-size:.86rem;font-weight:500;transition:all .15s;background:var(--card);}
.btn:hover{border-color:var(--ac);color:var(--ac);}.btn.pr{background:var(--ac);color:var(--bg);border-color:var(--ac);}.btn.pr:hover{background:#292524;color:#fff;}
hr{border:none;border-top:1px solid var(--bd);margin:0;}
section.bl{padding:4rem 0;}section.bl.alt{background:var(--bg2);}
.lab{font-size:.66rem;letter-spacing:.25em;text-transform:uppercase;color:var(--mut);margin-bottom:.7rem;}
h2{font-family:'Noto Serif KR',serif;font-size:1.6rem;font-weight:700;line-height:1.4;margin-bottom:1rem;}
p.lead{font-size:1rem;color:var(--tx2);margin-bottom:1.2rem;}p{color:var(--tx2);margin-bottom:.9rem;}
em{font-style:normal;color:var(--amb);font-weight:500;}strong{color:var(--tx);font-weight:700;}
code{font-family:'JetBrains Mono',monospace;font-size:.84em;background:var(--bg2);padding:.1rem .35rem;border-radius:4px;border:1px solid var(--bd);color:var(--ac);}
.tag{display:inline-block;font-size:.64rem;font-weight:700;padding:.1rem .45rem;border-radius:4px;letter-spacing:.03em;}
.tag.ok{background:rgba(21,128,61,.1);color:var(--grn);}.tag.fail{background:rgba(185,28,28,.09);color:var(--red);}
.tag.mixed{background:rgba(180,83,9,.1);color:var(--amb);}.tag.live{background:rgba(21,128,61,.1);color:var(--grn);}.tag.dead{background:rgba(185,28,28,.09);color:var(--red);}.tag.weak{background:rgba(180,83,9,.1);color:var(--amb);}
/* logical flow steps */
.flow{margin:1.6rem 0;}
.fstep{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:.9rem;}
.fstep .fhd{font-weight:700;color:var(--tx);font-size:1.02rem;margin-bottom:.9rem;display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;}
.fstep .fn{font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:600;color:#fff;background:var(--ac);padding:.16rem .55rem;border-radius:5px;letter-spacing:.04em;}
.frow{display:grid;grid-template-columns:3.2rem 1fr;gap:.7rem;margin-bottom:.5rem;align-items:start;}
.fk{font-size:.64rem;font-weight:700;letter-spacing:.06em;padding:.14rem 0;border-radius:4px;text-align:center;}
.fk.think{color:var(--tx2);}.fk.try{color:var(--ac);}.fk.find{color:var(--amb);}
.frow span:last-child{font-size:.9rem;color:var(--tx2);line-height:1.6;}
.fnext{margin-top:.9rem;padding-top:.8rem;border-top:1px dashed var(--bdl);font-size:.88rem;font-weight:500;color:var(--grn);}
.fnext::before{content:'→ ';font-weight:700;}
.fstep.fail .fnext{color:var(--red);}
/* tables */
.tbl-wrap{overflow-x:auto;border:1px solid var(--bd);border-radius:10px;margin:1.4rem 0;background:var(--card);}
table.led{width:100%;border-collapse:collapse;font-size:.82rem;}
table.led th{text-align:left;padding:.7rem .9rem;background:var(--bg2);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);position:sticky;top:0;}
table.led td{padding:.6rem .9rem;border-top:1px solid var(--bd);vertical-align:top;}
table.led td.tk{font-weight:600;color:var(--tx);min-width:180px;}
table.led pre{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--tx2);white-space:pre-wrap;line-height:1.5;}
table.ctl details{margin-top:.4rem;}
table.ctl summary{cursor:pointer;font-size:.72rem;color:var(--ac);font-weight:600;}
table.ctl details ul{margin:.4rem 0 0 1rem;padding:0;}
table.ctl details li{font-size:.72rem;color:var(--tx2);line-height:1.5;list-style:disc;margin-bottom:.2rem;}
table.ctl details code{font-size:.68rem;}
table.led tr.r-fail td.tk{color:var(--red);}
table.ctl{width:100%;border-collapse:collapse;font-size:.8rem;}
table.ctl th{text-align:left;padding:.6rem .8rem;background:var(--bg2);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);}
table.ctl td{padding:.5rem .8rem;border-top:1px solid var(--bd);vertical-align:top;}
table.ctl td.vn{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--tx);white-space:nowrap;}
table.ctl td.ne{font-family:'JetBrains Mono',monospace;color:var(--tx2);text-align:center;}
table.ctl td.pp{color:var(--tx2);line-height:1.55;}
.ph{font-size:.62rem;font-weight:700;padding:.08rem .4rem;border-radius:4px;letter-spacing:.02em;white-space:nowrap;}
.ph-controller{background:rgba(68,64,60,.1);color:var(--ac);}.ph-feasible{background:rgba(180,83,9,.1);color:var(--amb);}
.ph-predictive{background:rgba(21,128,61,.1);color:var(--grn);}.ph-stress{background:rgba(120,113,108,.12);color:var(--tx2);}
.ph-audit{background:rgba(28,25,23,.07);color:var(--mut);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin:1.5rem 0;}
@media(max-width:680px){.grid2{grid-template-columns:1fr;}.hstats{gap:1.3rem;}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:1.4rem 1.5rem;}
.card.g{border-left:4px solid var(--grn);}.card.r{border-left:4px solid var(--red);}
.card h4{font-size:.92rem;font-weight:700;margin-bottom:.5rem;}.card p{font-size:.86rem;margin-bottom:.45rem;line-height:1.6;}
.card .num{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--tx);}
.callout{background:var(--acd);border-left:4px solid var(--ac);border-radius:0 8px 8px 0;padding:1.2rem 1.5rem;margin:1.5rem 0;}
.callout p{margin-bottom:.35rem;font-size:.9rem;}.callout p:last-child{margin-bottom:0;}
.tl{border-left:2px solid var(--bdl);margin:1.4rem 0;padding-left:1.7rem;}
.tli{position:relative;padding-bottom:1.5rem;}
.tli::before{content:'';position:absolute;left:-2.26rem;top:.35rem;width:11px;height:11px;border-radius:50%;background:var(--bg);border:2px solid var(--ac);}
.tli.dead::before{border-color:var(--red);}.tli.live::before{border-color:var(--grn);}.tli.weak::before{border-color:var(--amb);}
.tlv{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mut);}
.tlt{font-weight:700;color:var(--tx);margin:.12rem 0 .25rem;}.tld{font-size:.88rem;color:var(--tx2);line-height:1.6;}
.note{font-size:.78rem;color:var(--mut);margin-top:.6rem;}
footer{padding:3.2rem 2rem;text-align:center;border-top:1px solid var(--bd);color:var(--mut);font-size:.8rem;}
footer a{color:var(--tx2);text-decoration:none;}footer a:hover{color:var(--ac);}
</style></head><body>
<nav><a class="home" href="/">소찬 · soccz</a>
<div class="nr"><a href="/projects/higan/">← HIGAN v2 보고서</a><a href="https://github.com/soccz/HIGAN" target="_blank" rel="noopener">GitHub ↗</a></div></nav>

<section class="hero"><div>
<div class="hl">RESEARCH LOGIC · 연구의 논리 흐름</div>
<h1>한 장의 saliency를 <em>이론</em>으로<br>키우려던 추론의 흐름</h1>
<p class="hsub">"침실 GAN의 saliency가 1차 미분이면, 2차(곡률)는 무엇인가?" 이 한 질문에서 시작해
곡률을 <strong>편집 위험을 재는 신호</strong>로 만들려 했다. 각 결과가 다음 수를 정했고 —
시도 → 발견 → 후퇴 → 재시도가 한 줄로 이어진다. 이 페이지는 그 <strong>논리 흐름</strong>을
따라가고, 강한 신호가 왜 매번 무너졌는지 보여준다. (모든 트랙·버전 raw는 맨 아래 부록.)</p>
<div class="hstats">
<div class="hs"><span class="hsn">∂I/∂α → ∂²I/∂α²</span><span class="hsl">출발 질문</span></div>
<div class="hs"><span class="hsn">6</span><span class="hsl">뒤집힌 결론</span></div>
<div class="hs"><span class="hsn">%(NC)d</span><span class="hsl">검증 버전</span></div>
<div class="hs"><span class="hsn">1</span><span class="hsl">살아남은 결과</span></div>
</div>
<div class="hb">
<a class="btn pr" href="https://github.com/soccz/HIGAN" target="_blank" rel="noopener">GitHub: soccz/HIGAN</a>
<a class="btn" href="/projects/higan/">← HIGAN v2 보고서</a>
<a class="btn" href="#data">전체 데이터 부록 ↓</a></div>
</div></section><hr>

<section class="bl"><div class="container">
<div class="lab">01 · 출발 질문</div><h2>saliency는 1차 미분이다 — 그럼 2차는?</h2>
<p class="lead">HIGAN v2는 잠재방향이 이미지의 <em>어디</em>에 작용하는지 픽셀 saliency로 보였다. 그건 결국
생성기 <code>G(α)</code>의 <strong>1차 pushforward ∂I/∂α</strong> — 방향으로 한 발 움직일 때 이미지가
어떻게 변하는지다. 자연스러운 다음 질문: <strong>2차(곡률) ∂²I/∂α²</strong>는 무엇을 뜻하나?
곡률이 크다 = 그 방향이 비선형적으로 휜다 = 조금만 더 편집해도 예측 못 하게 변한다.</p>
<p>그래서 <strong>가설</strong>을 세웠다: <em>곡률 ρ가 크면 그 방향으로 편집할 때 정체성이 더 망가진다.</em>
만약 사실이면 — 한 도메인의 그림이 아니라, 곡률로 <strong>안전한 편집을 고르는 cross-architecture 신호</strong>가
된다. 이게 한 장의 saliency를 top-venue 논문으로 만들 베팅이었다. 검증하려면 곡률을 정확히 재야 했다.</p>
</div></section><hr>

<section class="bl alt"><div class="container">
<div class="lab">02 · 논리 흐름 — 시도와 후퇴의 연쇄</div>
<h2>각 발견이 다음 수를 정했다</h2>
<p class="lead">핵심 주장 하나(곡률=편집위험)를 두고, <strong>생각 → 시도 → 발견 → 그래서 다음</strong>이
한 줄로 이어졌다. 강한 주장이 나올 때마다 직접 공격했고, 그때마다 신호가 약해졌다.</p>
<div class="flow">
<div class="fstep">
<div class="fhd"><span class="fn">도구</span>정확한 forward-mode 곡률 측정기</div>
<div class="frow"><span class="fk think">생각</span><span>FD는 2차 미분에 부정확하고, 역방향 AD는 3HW 출력에서 메모리가 터진다.</span></div>
<div class="frow"><span class="fk try">시도</span><span>forward-mode JVP — 1 pass로 정확한 pushforward, 2번 합성하면 정확한 곡률, 상수 메모리. StyleGAN1·2·SD에 적용.</span></div>
<div class="frow"><span class="fk find">발견</span><span>작동. toy 해석적 정답에서 JVP <span class="num">2.5e-17</span> vs FD 최선 <span class="num">8.6e-9</span> — <span class="tag ok">OK</span> 도구는 깨끗.</span></div>
<div class="fnext">도구가 됐으니, 곡률이 편집 위험을 <strong>제어</strong>하는지 본다</div>
</div>
<div class="fstep fail">
<div class="fhd"><span class="fn">시도 1</span>곡률로 안전한 편집을 고르는 제어기 <span class="tag fail">FAIL</span></div>
<div class="frow"><span class="fk think">생각</span><span>ρ 낮은 방향을 고르면 같은 의미 변화에서 손상이 적을 것이다.</span></div>
<div class="frow"><span class="fk try">시도</span><span>risk-aware controller + 음성대조(risk를 shuffle/invert) + cross-domain(bedroom·church). <code>control_v1~v30</code>.</span></div>
<div class="frow"><span class="fk find">발견</span><span>gain-only·random baseline을 안정적으로 못 이김. shuffle/invert와 차이 모호. church(StyleGAN2)에선 오히려 짐.</span></div>
<div class="fnext">"제어"는 너무 강한 주장. <strong>예측</strong>으로 후퇴한다</div>
</div>
<div class="fstep">
<div class="fhd"><span class="fn">시도 2</span>제어 말고 예측 — 손상을 맞히기만 하면</div>
<div class="frow"><span class="fk think">생각</span><span>제어기로 못 쓰더라도, semantic gain을 통제한 뒤 ρ가 손상을 <em>예측</em>하면 그것만으로 의미 있다.</span></div>
<div class="frow"><span class="fk try">시도</span><span>predictive validity — gain 통제 후 ρ↔손상 상관. <code>control_v31~</code>.</span></div>
<div class="frow"><span class="fk find">발견</span><span>중간 정도 Spearman. 하지만 <strong>null-sensitive</strong> — permutation에 취약.</span></div>
<div class="fnext">예측이 진짜인지 <strong>가정을 전부 흔들어</strong> 본다</div>
</div>
<div class="fstep">
<div class="fhd"><span class="fn">시도 3</span>가정 스트레스 가틀릿</div>
<div class="frow"><span class="fk think">생각</span><span>예측이 견고하면 추정기·프롬프트·평가셋·매칭을 바꿔도 살아남아야 한다.</span></div>
<div class="frow"><span class="fk try">시도</span><span>FD estimator(v42), prompt 템플릿(v43/v46/v51), n_test 2배(v47), stricter matching(v48), DINO(v64), seed-scaling(v68~), permutation null(v82/v90). <code>control_v42~v95</code>.</span></div>
<div class="frow"><span class="fk find">발견</span><span>매번 신호가 약해지거나 null 범위 안. <strong>743 실행, 0 재현실패</strong> — 그러나 주장은 끝내 단단해지지 않음.</span></div>
<div class="fnext">"약한데 안 죽는" 신호 — 강한 버전을 <strong>직접 반박</strong>해보기로</div>
</div>
<div class="fstep fail">
<div class="fhd"><span class="fn">반전</span>적대적 감사 — 강한 신호는 전부 artifact <span class="tag fail">결정적</span></div>
<div class="frow"><span class="fk think">생각</span><span>강한 주장이 나오면 독립 에이전트가 raw로 재계산하고 <em>반박</em>한다(확인이 아니라 깨기).</span></div>
<div class="frow"><span class="fk try">시도</span><span>매 강한 결과를 다수 에이전트가 공격. 6번의 결론 뒤집힘(아래 타임라인).</span></div>
<div class="frow"><span class="fk find">발견</span><span>강한 신호 = 측정 artifact. tautology(무작위 selector가 재현)·순환논리(손상을 손상으로 통제)·outlier 의존·leakage·between-attr proxy.</span></div>
<div class="fnext">도달한 floor: <strong>ρ는 약하고 baseline 정의에 fragile하다</strong></div>
</div>
</div>
</div></section><hr>

<section class="bl"><div class="container">
<div class="lab">03 · 6번 뒤집힘 — 같은 데이터, 매번 바뀐 결론</div>
<h2>주장이 양 극단을 오가며 수렴한 지점</h2>
<div class="tl">
<div class="tli dead"><div class="tlv">v1 · 과대</div><div class="tlt">"ρ가 편집 손상을 예측한다" <span class="tag dead">폐기</span></div><div class="tld">곡률이 trivial한 편집 크기(magnitude)를 넘는 가치를 거의 못 줬다.</div></div>
<div class="tli dead"><div class="tlv">v2–v3 · 반대로 과대</div><div class="tlt">"magnitude가 충분통계량, geometry=0" <span class="tag dead">폐기</span></div><div class="tld">"0/41"의 baseline이 outcome과 결합된 leaky 4-feature MAX — 그것도 artifact.</div></div>
<div class="tli weak"><div class="tlv">v4 · 절충</div><div class="tlt">"real-but-weak residual" <span class="tag weak">불안정</span></div><div class="tld">"6/6 ordinal residual"이 사실 between-attribute 난이도 proxy였다.</div></div>
<div class="tli dead"><div class="tlv">v5 · 다시 과소</div><div class="tlt">"ρ는 actionable signal이 아니다" <span class="tag dead">또 과대</span></div><div class="tld">세 기둥(C4b leak, LPIPS 부호반전, within-unit 붕괴)이 tautology·순환논리·n=6 노이즈로 판명.</div></div>
<div class="tli live"><div class="tlv">v6 · 데이터-진실</div><div class="tlt">곡률 = 약하고 baseline에 fragile한 신호 <span class="tag live">honest</span></div><div class="tld">"ρ가 magnitude를 이기냐"는 baseline 정의에 따라 갈린다. 이 fragility 자체가 robust한 발견.</div></div>
</div>
</div></section><hr>

<section class="bl alt"><div class="container">
<div class="lab">04 · 결산 — 무엇이 살아남고 무엇이 무너졌나</div><h2>같은 원장에 공존하는 성공과 실패</h2>
<div class="grid2">
<div class="card g"><h4>살아남음</h4>
<p>① forward-mode JVP 곡률 도구 (cross-arch, 정확)</p>
<p>② FD는 곡률 복원 못 함 — toy로 airtight (<span class="num">8.6e-9</span> vs <span class="num">2.5e-17</span>)</p>
<p>③ per-layer 곡률 단조감소 (8/8, Spearman −0.96~−0.80)</p>
<p>④ 곡률-vs-magnitude의 baseline-fragility를 규명한 것</p></div>
<div class="card r"><h4>무너짐 (정직하게)</h4>
<p>① edit-risk를 예측·제어하는 actionable 신호 — fragile</p>
<p>② cross-encoder 보편성 — 1개 attr 빼면 Pearson 0.99→0.10</p>
<p>③ FFHQ C5 인코더 전이 — sal_corr 0.07 정체</p>
<p>④ Park NeurIPS23 재현 ρ=0.04 / real-LSUN 전이 +1148%</p></div>
</div>
<div class="callout"><p><strong>도구 + FD-non-recoverability만 깨끗하게 살아남았다.</strong> 짧은 methods note로 banking.</p></div>
</div></section><hr>

<section class="bl"><div class="container">
<div class="lab">05 · 피벗 & 메타 교훈</div><h2>다음 질문도 같은 패턴 — 왜 반복됐나</h2>
<p class="lead"><strong>피벗:</strong> HIGAN이 tool-first라 약했나 싶어, problem-first로 "오픈 TSFM zero-shot SOTA가
오염인가"를 감사했다. 그런데 "Chronos가 baseline 박살" gap은 <strong>baseline-starvation artifact</strong>(짧은
series)였고, 제대로 된 baseline엔 겨우 0.03 이김 — 부풀릴 advantage가 없었다. <em>같은 패턴, 같은 날 shelved.</em></p>
<p><strong>메타 교훈:</strong> 두 번 다 강한 신호=artifact, 진짜 효과=thin. 구조적이다 — "주장이 진짜냐"를 엄격히
감사하면 대부분 부풀려져 있으니 <em>깎는(deflationary) 결과</em>로 수렴한다. top-venue가 원하는 "새로 되는 것"이
아니라 "주장이 안 맞다" 쪽. 그래서 멈췄다. 깨끗한 것을 banking하고, 다음 큰 시도는 감사가 아니라
<strong>positive-construction</strong>으로 — 새 시점에.</p>
<div class="callout"><p><strong>그래도 rigor는 남는다.</strong> 6번의 뒤집힘을 워크플로우 6번이 아니라 Day 3에 잡게 된 것 —
자기 자신을 속이지 않는 법은 작동했다. 743 실행 0 실패, 17자리 bit-identical 재현이 그 증거다.</p></div>
</div></section><hr>

<section class="bl alt" id="data"><div class="container">
<div class="lab">부록 · 전체 데이터 (raw)</div><h2>모든 트랙·버전, 잘림 없이</h2>
<p class="lead">위 논리 흐름의 근거 데이터 전부. 펼쳐서 보거나 <code>github.com/soccz/HIGAN</code>에서 raw JSON으로.</p>
<details><summary style="cursor:pointer;font-weight:600;color:var(--ac);padding:.5rem 0;">▶ 전체 실험 원장 — %(NT)d개 분석 트랙 (raw 결과)</summary>
<div class="tbl-wrap"><table class="led">
<thead><tr><th>판정</th><th>트랙</th><th>결과 (raw)</th></tr></thead>
<tbody>
%(LEDGER)s
</tbody></table></div></details>
<details><summary style="cursor:pointer;font-weight:600;color:var(--ac);padding:.5rem 0;">▶ control 검증 캠페인 — %(NC)d개 버전 (시도+실행내역)</summary>
<div class="tbl-wrap"><table class="ctl">
<thead><tr><th>버전</th><th>단계</th><th>exp</th><th>시도한 것 (purpose)</th></tr></thead>
<tbody>
%(CONTROL)s
</tbody></table></div></details>
<p class="note">%(NO)d개 결과 폴더 · %(ND)d개 분석 스크립트 · HIGAN note(<code>note/</code>) ·
TSFM(<code>tsfm_audit/STATUS.md</code>) · 실험 코드 전체 <code>github.com/soccz/HIGAN/tree/experiments</code>.</p>
</div></section>

<footer><p>HIGAN 일반화 연구 — 논리 흐름 · 2026 · 정직한 회고</p>
<p style="margin-top:.6rem;"><a href="/projects/higan/">HIGAN v2 보고서</a> · <a href="https://github.com/soccz/HIGAN" target="_blank" rel="noopener">github.com/soccz/HIGAN</a> · <a href="/">소찬 · soccz</a></p></footer>
</body></html>"""

page = PAGE
for k, val in [("%(NT)d", len(tracks)), ("%(NC)d", len(ctrl)), ("%(NO)d", n_out),
               ("%(ND)d", n_dev), ("%(LEDGER)s", "\n".join(ledger)),
               ("%(CONTROL)s", "\n".join(ctrl))]:
    page = page.replace(k, str(val))
(OUT / "index.html").write_text(page)
# clean temp fragments
for t in ["_ledger_rows.html", "_control_rows.html", "_stats.json"]:
    p = OUT / t
    if p.exists(): p.unlink()
print(f"wrote {OUT/'index.html'}  ({len(page)} bytes)")
print(f"  {len(tracks)} tracks, {len(ctrl)} controls, {n_out} out dirs, {n_dev} scripts")
