import csv
import json
from datetime import datetime
from pathlib import Path


def load_results(csv_path: str = "benchmarks/results.csv"):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def generate_markdown_table(results):
    lines = [
        "## FineTuneBench Results",
        "",
        "| Rank | Model | Dataset | Hours | GPU Cost | Eval Score | Hallucination Rate |",
        "|---|---|---|---|---|---|---|",
    ]
    sorted_results = sorted(results, key=lambda r: float(r["eval_score"]), reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        model = r["model"].split("/")[-1]
        lines.append(
            f"| {rank} | {model} | {r['dataset'].split('/')[-1]} "
            f"| {r['hours']}h | ${r['gpu_cost_usd']} "
            f"| {r['eval_score']} | {r['hallucination_rate']} |"
        )
    lines.append("")
    lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


def generate_html_leaderboard(results, output_path: str = "benchmarks/index.html"):
    sorted_results = sorted(results, key=lambda r: float(r["eval_score"]), reverse=True)

    rows = []
    for rank, r in enumerate(sorted_results, 1):
        model = r["model"].split("/")[-1]
        eval_score = float(r["eval_score"])
        hallucination_rate = float(r["hallucination_rate"])
        cost = float(r["gpu_cost_usd"])
        hours = float(r["hours"])
        rows.append(
            f"""
            <tr>
              <td class="rank rank-{min(rank, 3)}">{rank}</td>
              <td class="model">{model}</td>
              <td>{r['dataset'].split('/')[-1]}</td>
              <td>{hours}h</td>
              <td>${cost:.2f}</td>
              <td class="score score-{'high' if eval_score >= 0.83 else 'mid' if eval_score >= 0.81 else 'low'}">{eval_score:.4f}</td>
              <td class="hallucination hall-{'low' if hallucination_rate <= 0.14 else 'mid' if hallucination_rate <= 0.16 else 'high'}">{hallucination_rate:.4f}</td>
            </tr>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FineTuneBench — Live Cost vs Quality Leaderboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f1115;
      color: #e6e6e6;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 20px;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 0.25rem;
      background: linear-gradient(90deg, #61dafb, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      color: #9ca3af;
      margin-bottom: 2rem;
      font-size: 1.05rem;
    }}
    .badge {{
      display: inline-block;
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 1.5rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #1a1d23;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}
    thead {{
      background: #111318;
    }}
    th {{
      text-align: left;
      padding: 14px 16px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #9ca3af;
    }}
    td {{
      padding: 14px 16px;
      border-top: 1px solid #2a2e38;
      font-variant-numeric: tabular-nums;
    }}
    tr:hover td {{
      background: #22262e;
    }}
    .rank {{
      font-weight: 700;
      width: 60px;
      text-align: center;
    }}
    .rank-1 {{ color: #fbbf24; }}
    .rank-2 {{ color: #94a3b8; }}
    .rank-3 {{ color: #b45309; }}
    .model {{ font-weight: 600; color: #e6e6e6; }}
    .score {{ font-weight: 700; }}
    .score-high {{ color: #10b981; }}
    .score-mid {{ color: #fbbf24; }}
    .score-low {{ color: #ef4444; }}
    .hallucination {{ font-weight: 700; }}
    .hall-low {{ color: #10b981; }}
    .hall-mid {{ color: #fbbf24; }}
    .hall-high {{ color: #ef4444; }}
    footer {{
      margin-top: 2rem;
      color: #6b7280;
      font-size: 0.85rem;
    }}
    a {{ color: #61dafb; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>FineTuneBench</h1>
    <p class="subtitle">Live Cost vs Quality Leaderboard for QLoRA Fine-Tuning</p>
    <div class="badge">70B fine-tuned for $4.20 on one GPU</div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Model</th>
          <th>Dataset</th>
          <th>Hours</th>
          <th>GPU Cost</th>
          <th>Eval Score</th>
          <th>Hallucination Rate</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    <footer>
      Generated on {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} &middot;
      <a href="https://github.com/Lubnaaziz-28/llm-finetune-lora">View Source</a>
    </footer>
  </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)


def generate_readme_table(results):
    lines = [
        "| Rank | Model | Dataset | Hours | GPU Cost | Eval Score | Hallucination Rate |",
        "|---|---|---|---|---|---|---|",
    ]
    sorted_results = sorted(results, key=lambda r: float(r["eval_score"]), reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        model = r["model"].split("/")[-1]
        lines.append(
            f"| {rank} | {model} | {r['dataset'].split('/')[-1]} "
            f"| {r['hours']}h | ${r['gpu_cost_usd']} "
            f"| {r['eval_score']} | {r['hallucination_rate']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    results = load_results()
    md = generate_markdown_table(results)
    Path("benchmarks/leaderboard.md").write_text(md)
    generate_html_leaderboard(results)
    print("Generated benchmarks/leaderboard.md and benchmarks/index.html")
    print("\n" + md)
