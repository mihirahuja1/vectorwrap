#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark.py.

Generates comparison charts and summary tables.
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any
import os


def load_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def create_comparison_charts(results: Dict[str, Any], output_dir: str = ".") -> None:
    """Create comparison charts from benchmark results."""

    backends_data = results.get("backends", {})
    successful = {
        name: data
        for name, data in backends_data.items()
        if data.get("status") == "success"
    }

    if not successful:
        print("No successful benchmarks to visualize")
        return

    # Extract metrics
    backend_names = list(successful.keys())
    insert_rates = [data["insert"]["vectors_per_sec"] for data in successful.values()]
    query_rates_10 = [data["query_top10"]["queries_per_sec"] for data in successful.values()]
    query_rates_100 = [data["query_top100"]["queries_per_sec"] for data in successful.values()]
    avg_latencies = [
        data["query_top10"]["avg_time"] * 1000 for data in successful.values()
    ]  # Convert to ms

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Insert Performance
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(backend_names, insert_rates, color="skyblue", alpha=0.8)
    ax1.set_xlabel("Vectors per Second", fontsize=12, fontweight="bold")
    ax1.set_title("Insert Throughput", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, insert_rates)):
        ax1.text(
            value + max(insert_rates) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontweight="bold",
        )

    # 2. Query Performance Comparison (top-10 vs top-100)
    ax2 = fig.add_subplot(gs[0, 1])
    x = range(len(backend_names))
    width = 0.35

    bars2a = ax2.bar(
        [i - width / 2 for i in x],
        query_rates_10,
        width,
        label="Top-10",
        color="lightcoral",
        alpha=0.8,
    )
    bars2b = ax2.bar(
        [i + width / 2 for i in x],
        query_rates_100,
        width,
        label="Top-100",
        color="lightgreen",
        alpha=0.8,
    )

    ax2.set_ylabel("Queries per Second", fontsize=12, fontweight="bold")
    ax2.set_title("Query Performance (QPS)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(backend_names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(query_rates_10 + query_rates_100) * 0.01,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 3. Average Query Latency
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(backend_names, avg_latencies, color="mediumpurple", alpha=0.8)
    ax3.set_ylabel("Milliseconds", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Backend", fontsize=12, fontweight="bold")
    ax3.set_title("Average Query Latency (Top-10)", fontsize=14, fontweight="bold")
    ax3.set_xticklabels(backend_names, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars3, avg_latencies)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_latencies) * 0.01,
            f"{value:.2f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Summary Statistics Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Create summary data
    summary_data = []
    for name in backend_names:
        data = successful[name]
        summary_data.append(
            [
                name,
                f"{data['insert']['vectors_per_sec']:.0f}",
                f"{data['query_top10']['queries_per_sec']:.1f}",
                f"{data['query_top10']['avg_time']*1000:.2f}",
            ]
        )

    # Create table
    table = ax4.table(
        cellText=summary_data,
        colLabels=["Backend", "Insert\n(vec/s)", "Query\n(q/s)", "Latency\n(ms)"],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")

    ax4.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)

    # Add metadata
    metadata = results.get("metadata", {})
    fig.suptitle(
        f"vectorwrap Benchmark Results\n"
        f"Dataset: {metadata.get('dataset_size', 'N/A'):,} vectors × {metadata.get('dimensions', 'N/A')}D",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    output_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Saved visualization to {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(output_dir, "benchmark_results.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"📄 Saved PDF to {pdf_path}")

    plt.close()


def create_markdown_report(results: Dict[str, Any], output_file: str = "BENCHMARK_RESULTS.md") -> None:
    """Create a markdown report from benchmark results."""

    with open(output_file, "w") as f:
        # Header
        metadata = results.get("metadata", {})
        f.write("# vectorwrap Benchmark Results\n\n")
        f.write(f"**Date:** {metadata.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Dataset Size:** {metadata.get('dataset_size', 'N/A'):,} vectors\n\n")
        f.write(f"**Vector Dimensions:** {metadata.get('dimensions', 'N/A')}\n\n")

        # Summary Table
        f.write("## Performance Summary\n\n")
        f.write("| Backend | Insert (vec/s) | Query Top-10 (q/s) | Query Top-100 (q/s) | Avg Latency (ms) | Status |\n")
        f.write("|---------|----------------|--------------------|--------------------|------------------|--------|\n")

        backends_data = results.get("backends", {})
        for name, data in backends_data.items():
            if data.get("status") == "success":
                insert_rate = data["insert"]["vectors_per_sec"]
                query_rate_10 = data["query_top10"]["queries_per_sec"]
                query_rate_100 = data["query_top100"]["queries_per_sec"]
                latency = data["query_top10"]["avg_time"] * 1000

                f.write(
                    f"| {name} | {insert_rate:.0f} | {query_rate_10:.1f} | "
                    f"{query_rate_100:.1f} | {latency:.2f} | ✅ |\n"
                )
            else:
                error = data.get("error", "Unknown error")
                f.write(f"| {name} | - | - | - | - | ❌ {error} |\n")

        # Detailed Results
        f.write("\n## Detailed Results\n\n")

        for name, data in backends_data.items():
            if data.get("status") == "success":
                f.write(f"### {name}\n\n")
                f.write(f"**Connection URL:** `{data.get('url', 'N/A')}`\n\n")

                # Insert metrics
                insert = data.get("insert", {})
                f.write("**Insert Performance:**\n")
                f.write(f"- Total time: {insert.get('total_time', 0):.2f}s\n")
                f.write(f"- Throughput: {insert.get('vectors_per_sec', 0):.0f} vectors/sec\n")
                f.write(f"- Avg time: {insert.get('avg_time', 0)*1000:.2f}ms per batch\n\n")

                # Query metrics (top-10)
                query = data.get("query_top10", {})
                f.write("**Query Performance (Top-10):**\n")
                f.write(f"- Throughput: {query.get('queries_per_sec', 0):.1f} queries/sec\n")
                f.write(f"- Avg latency: {query.get('avg_time', 0)*1000:.2f}ms\n")
                f.write(f"- Median latency: {query.get('median_time', 0)*1000:.2f}ms\n")
                f.write(f"- P95 latency: {query.get('p95_time', 0)*1000:.2f}ms\n")
                f.write(f"- P99 latency: {query.get('p99_time', 0)*1000:.2f}ms\n\n")

                # Filtered queries
                filtered = data.get("query_filtered", {})
                if "error" not in filtered:
                    f.write("**Filtered Query Performance:**\n")
                    f.write(f"- Throughput: {filtered.get('queries_per_sec', 0):.1f} queries/sec\n")
                    f.write(f"- Avg latency: {filtered.get('avg_time', 0)*1000:.2f}ms\n\n")

        # Recommendations
        f.write("\n## Recommendations\n\n")

        successful = {
            name: data
            for name, data in backends_data.items()
            if data.get("status") == "success"
        }

        if successful:
            # Find best performers
            best_insert = max(
                successful.items(), key=lambda x: x[1]["insert"]["vectors_per_sec"]
            )
            best_query = max(
                successful.items(), key=lambda x: x[1]["query_top10"]["queries_per_sec"]
            )

            f.write("**Best Performance:**\n")
            f.write(
                f"- Fastest Insert: **{best_insert[0]}** "
                f"({best_insert[1]['insert']['vectors_per_sec']:.0f} vec/s)\n"
            )
            f.write(
                f"- Fastest Query: **{best_query[0]}** "
                f"({best_query[1]['query_top10']['queries_per_sec']:.0f} q/s)\n\n"
            )

        f.write("**Use Case Recommendations:**\n")
        f.write("- **Prototyping:** SQLite or DuckDB (easy setup, no server)\n")
        f.write("- **Production:** PostgreSQL + pgvector (scalable, ACID)\n")
        f.write("- **Analytics:** DuckDB or ClickHouse (combine vectors + analytics)\n")
        f.write("- **High Performance:** ClickHouse (large-scale workloads)\n\n")

        f.write("---\n")
        f.write("*Generated by vectorwrap benchmark suite*\n")

    print(f"📝 Saved markdown report to {output_file}")


def main():
    """Visualize benchmark results."""
    parser = argparse.ArgumentParser(description="Visualize vectorwrap benchmark results")
    parser.add_argument(
        "results_file",
        type=str,
        default="benchmark_results.json",
        nargs="?",
        help="JSON file with benchmark results",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip markdown report generation"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    if not args.no_charts:
        print("Creating charts...")
        create_comparison_charts(results, args.output_dir)

    # Generate markdown report
    if not args.no_report:
        print("Creating markdown report...")
        report_path = os.path.join(args.output_dir, "BENCHMARK_RESULTS.md")
        create_markdown_report(results, report_path)

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
