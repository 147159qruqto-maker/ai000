#!/usr/bin/env python3
"""高效压缩作业对比：同一道题，正常版 vs 压缩版，同一模型跑两遍，不换模型。
测的是：用压缩 token 作业，正确率、速度、效率保不保得住。
用法：python3 compress_compare.py --models qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b --sleep 3
输出：results/compress-<时间戳>/results.json + report.md"""
import argparse, json, time
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run import call_model, auto_score

BASE = Path(__file__).parent


def run_one(model, prompt, q, effort):
    try:
        text, secs, usage = call_model(model, prompt, q["max_tokens"], effort)
        err = ""
    except Exception as e:
        if "429" in str(e):
            print(f"  429 等 60 秒重试一次：{model} {q['id']}", flush=True)
            time.sleep(60)
            try:
                text, secs, usage = call_model(model, prompt, q["max_tokens"], effort)
                err = ""
            except Exception as e2:
                text, secs, usage, err = "", 0, {}, f"{type(e2).__name__}: {str(e2)[:300]}"
        else:
            text, secs, usage, err = "", 0, {}, f"{type(e).__name__}: {str(e)[:300]}"
    score, passed, failed, manual = auto_score(q, text) if not err else (False, [], ["call_failed"], [])
    return {"score": score, "failed": failed, "manual": manual,
            "seconds": round(secs, 2), "total_tokens": usage.get("total_tokens"),
            "error": err, "output": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--reasoning-effort", default="low")
    args = ap.parse_args()
    qs = json.load(open(BASE / "questions_compress.json"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = BASE / "results" / f"compress-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for q in qs:
            n = run_one(model, q["normal_prompt"], q, args.reasoning_effort)
            time.sleep(args.sleep)
            c = run_one(model, q["compressed_prompt"], q, args.reasoning_effort)
            time.sleep(args.sleep)
            nt, ct = n["total_tokens"] or 0, c["total_tokens"] or 0
            saving = round((nt - ct) / nt * 100, 1) if nt else None
            rows.append({"model": model, "id": q["id"], "type": q["type"],
                         "normal": n, "compressed": c, "token_saving_pct": saving})
            print(f"{model} {q['id']} 正常 score={n['score']} tokens={nt} | "
                  f"压缩 score={c['score']} tokens={ct} 节省={saving}%", flush=True)
    json.dump(rows, open(outdir / "results.json", "w"), ensure_ascii=False, indent=2)
    lines = [f"# 高效压缩作业对比报告 {stamp}", "",
             "测法：同一道题正常版 vs 压缩版，同一模型跑两遍，途中不换模型；"
             "所有模型统一用 reasoning_effort=low，同一提示词对所有模型相同。", ""]
    # 2026-10-31 重跑教训：多约束决策题（compress_sensitive，如 C07）压缩后全模型丢分，
    # 不能和普通题混着算 headline；拒绝/事实陷阱题压缩后 token 可能反增，须逐题列出。
    # 所以 headline 只算非敏感题，敏感题单独列，429 失败题也不计入分母。
    def is_rate_limited(side):
        err = side.get("error") or ""
        return "429" in err or "Too Many Requests" in err
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        auto = [r for r in rs if r["normal"]["score"] is not None
                and not is_rate_limited(r["normal"]) and not is_rate_limited(r["compressed"])]
        sensitive_ids = {q["id"] for q in qs if q.get("compress_sensitive")}
        main = [r for r in auto if r["id"] not in sensitive_ids]
        sens = [r for r in auto if r["id"] in sensitive_ids]
        n_ok = sum(1 for r in main if r["normal"]["score"])
        c_ok = sum(1 for r in main if r["compressed"]["score"])
        nt = sum((r["normal"]["total_tokens"] or 0) for r in rs)
        ct = sum((r["compressed"]["total_tokens"] or 0) for r in rs)
        saving = round((nt - ct) / nt * 100, 1) if nt else None
        increased = [r["id"] for r in rs
                     if (r["compressed"]["total_tokens"] or 0) > (r["normal"]["total_tokens"] or 0)]
        limited = [r["id"] for r in rows if r["model"] == model
                   and (is_rate_limited(r["normal"]) or is_rate_limited(r["compressed"]))]
        lines.append(f"- {model}：非敏感题自动判分 正常 {n_ok}/{len(main)}，压缩 {c_ok}/{len(main)}；"
                     f"总 token 正常 {nt} → 压缩 {ct}，节省 {saving}%，"
                     f"压缩后丢分题 {[r['id'] for r in main if r['normal']['score'] and not r['compressed']['score']]}；"
                     f"敏感题（多约束决策，单独列不计 headline）{[r['id'] + ':正常' + str(r['normal']['score']) + '/压缩' + str(r['compressed']['score']) for r in sens]}；"
                     f"压缩后 token 反增题 {increased}；429 未跑出题 {limited}")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成，结果在 {outdir}")


if __name__ == "__main__":
    main()
