#!/usr/bin/env python3
"""文本模型测试工具 v1：普通版长文本 + 进阶版13条。
只测 Groq API 文本模型。密钥不写进代码和日志。
用法见 README.md。"""
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
UA = "ModelTestTool/1.0"
API = "https://api.groq.com/openai/v1/chat/completions"


def get_key():
    """只读环境变量 GROQ_API_KEY，不写进代码和日志。
    没有 key 时直接报错退出，不静默走其他渠道。"""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "未设置 GROQ_API_KEY 环境变量：请先在 Groq 控制台申请免费 key，"
            "再执行 export GROQ_API_KEY=你的key 后重试。")
    return key


def call_model(model, prompt, max_tokens, reasoning_effort=None):
    import urllib.request
    key = get_key()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    body = json.dumps(body).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "User-Agent": UA,
        "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return msg, time.time() - t0, usage


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text.strip()


def run_code_tests(code, tests):
    for t in tests:
        prog = code + f"\nprint({t['call']})\n"
        try:
            r = subprocess.run([sys.executable, "-c", prog],
                               capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode != 0:
            return False, r.stderr.strip()[-300:]
        if r.stdout.strip() != str(t["expected"]):
            return False, f"got {r.stdout.strip()!r} want {t['expected']!r}"
    return True, ""


def auto_score(q, text):
    checks = q.get("checks", {})
    passed, failed, manual = [], [], []
    for s in checks.get("contains", []):
        (passed if s in text else failed).append(f"contains:{s}")
    for s in checks.get("not_contains", []):
        (passed if s not in text else failed).append(f"not_contains:{s}")
    if "answer" in q:
        (passed if q["answer"] in text else failed).append("final_answer")
    if checks.get("code_tests"):
        ok, detail = run_code_tests(extract_code(text), checks["code_tests"])
        (passed if ok else failed).append("code_run:" + ("" if ok else detail))
    manual = checks.get("manual", [])
    if not checks.get("contains") and "answer" not in q and not checks.get("code_tests"):
        return None, passed, failed, manual  # 纯人工判分，不硬判
    return (len(failed) == 0), passed, failed, manual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["normal", "advanced"], required=True)
    ap.add_argument("--models", default="qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--compress", action="store_true",
                    help="工具省 token 模式（统一适用所有模型，途中不换模型）："
                    "提示词去多余空白，推理模型统一用 reasoning_effort=low，"
                    "不放宽任何题的上限。注意：这只是工具本身省 token，"
                    "不是高效压缩作业测试，后者见 compress_compare.py")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="每个模型本次总 token 上限，超了就停掉该模型剩余题目、记 skipped，"
                    "避免一轮跑下来超预算。0 表示不设限。")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    qs = json.load(open(BASE / f"questions_{args.version}.json"))
    if args.tier:
        qs = [q for q in qs if q.get("tier") == args.tier]
    if args.limit:
        qs = qs[:args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        total_cap = sum(q.get("max_tokens", 300) for q in qs) * len(models)
        print(f"版本={args.version} 题目={len(qs)} 模型={len(models)} "
              f"输出上限合计约 {total_cap} token（未计输入，实际一般远低于上限）")
        for q in qs:
            print(f"  {q['id']} [{q.get('tier','-')}/{q['type']}] 上限{q.get('max_tokens',300)}")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.out) if args.out else BASE / "results" / f"{args.version}-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    effort = args.reasoning_effort or ("low" if args.compress else None)
    cum_tokens = {m: 0 for m in models}
    for model in models:
        for q in qs:
            if args.token_budget and cum_tokens[model] >= args.token_budget:
                results.append({
                    "model": model, "id": q["id"], "tier": q.get("tier", ""),
                    "type": q["type"], "auto_score": None, "passed": [],
                    "failed": ["skipped_token_budget"], "manual_check": [],
                    "seconds": 0, "prompt_tokens": None, "completion_tokens": None,
                    "total_tokens": None, "chars": 0, "error": "token_budget_reached",
                    "output": "",
                })
                print(f"{model} {q['id']} skipped（已达 token 预算 {args.token_budget}）", flush=True)
                continue
            prompt = " ".join(q["prompt"].split()) if args.compress else q["prompt"]
            try:
                text, secs, usage = call_model(
                    model, prompt, q.get("max_tokens", 300), effort)
                err = ""
            except Exception as e:
                # 撞到限速只等 60 秒重试一次，再撞就记失败不停刷，避免硬刷限速
                if "429" in str(e):
                    print(f"  429 限速，等 60 秒重试一次：{model} {q['id']}", flush=True)
                    time.sleep(60)
                    try:
                        text, secs, usage = call_model(
                            model, prompt, q.get("max_tokens", 300), effort)
                        err = ""
                    except Exception as e2:
                        text, secs, usage, err = "", 0, {}, f"{type(e2).__name__}: {str(e2)[:300]}"
                else:
                    text, secs, usage, err = "", 0, {}, f"{type(e).__name__}: {str(e)[:300]}"
            if usage.get("total_tokens"):
                cum_tokens[model] += usage["total_tokens"]
            score, passed, failed, manual = auto_score(q, text) if not err else (False, [], ["call_failed"], [])
            results.append({
                "model": model, "id": q["id"], "tier": q.get("tier", ""),
                "type": q["type"], "auto_score": score, "passed": passed,
                "failed": failed, "manual_check": manual,
                "seconds": round(secs, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "chars": len(text), "error": err, "output": text,
            })
            print(f"{model} {q['id']} score={score} tokens={usage.get('total_tokens')} "
                  f"secs={secs:.1f} err={err}", flush=True)
            time.sleep(args.sleep)  # 放慢节奏，避开免费层限速
    json.dump(results, open(outdir / "results.json", "w"), ensure_ascii=False, indent=2)

    auto = [r for r in results if r["auto_score"] is not None]
    lines = [f"# 测试报告 {args.version} {stamp}", "",
             f"题目 {len(qs)} 道，模型 {len(models)} 个，自动可判 {len(auto)} 条，"
             "长文本/主观题只列人工复核栏，不计入正确率。",
             f"工具省 token 模式：{'开（--compress，提示词去空白、思考档统一 low、不放宽上限）' if args.compress else '关'}；"
             f"思考档 {effort or '默认'}；每模型 token 预算 {args.token_budget or '不限'}。", ""]
    # 2026-10-31 重跑教训：撞 429 没跑出来的题不是模型答错，正确率分母要去掉，
    # 否则 qwen 普通版 2/7 那种数会误导（其中 3 道是 429）。三模型全挂的题单独列，
    # 它们更可能是题目或判分过严，需人工复核，不是某个模型的问题。
    def is_rate_limited(r):
        return "429" in (r.get("error") or "") or "Too Many Requests" in (r.get("error") or "")
    for model in models:
        rs = [r for r in auto if r["model"] == model]
        limited = [r for r in rs if is_rate_limited(r)]
        valid = [r for r in rs if not is_rate_limited(r)]
        ok = sum(1 for r in valid if r["auto_score"])
        toks = sum(r["total_tokens"] or 0 for r in results if r["model"] == model)
        skipped = sum(1 for r in results if r["model"] == model and r["error"] == "token_budget_reached")
        avg = round(toks / max(1, len([r for r in results if r["model"] == model and r["total_tokens"]])), 1)
        lines.append(f"- {model}：自动判分 {ok}/{len(valid)}（不含 429 失败 {len(limited)} 题），"
                     f"总 token {toks}，平均每题 {avg} token，预算跳过 {skipped} 题，"
                     f"失败题 {[r['id'] for r in valid if not r['auto_score']]}，"
                     f"429 未跑出题 {[r['id'] for r in limited]}")
    if len(models) > 1:
        failed_sets = []
        for model in models:
            valid = [r for r in auto if r["model"] == model and not is_rate_limited(r)]
            failed_sets.append({r["id"] for r in valid if not r["auto_score"]})
        common = sorted(set.intersection(*failed_sets)) if failed_sets else []
        lines.append(f"- 三模型全挂题（需人工复核题目或判分是否过严）：{common}")
    lines += ["", "## 人工复核（长文本/主观题）"]
    for r in results:
        if r["manual_check"]:
            lines.append(f"- {r['model']} {r['id']}：{'; '.join(r['manual_check'])}")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成，结果在 {outdir}")


if __name__ == "__main__":
    main()
