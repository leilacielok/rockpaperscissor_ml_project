from __future__ import annotations

import argparse
from pathlib import Path

import keras
from keras.utils import plot_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate PNG architecture diagrams for saved Keras models (.keras)."
    )
    p.add_argument(
        "--models",
        default="model_a_best.keras,model_b_best.keras,model_c_best.keras",
        help="Comma-separated model filenames inside ./models (default: all best models).",
    )
    p.add_argument(
        "--outdir",
        default="models/diagrams",
        help="Output directory (relative to project root).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no_shapes", action="store_true")
    p.add_argument("--no_names", action="store_true")
    p.add_argument("--no_expand", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    out_dir = root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_files = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_files:
        raise SystemExit("No models specified.")

    print("Project root:", root)
    print("Models dir:", models_dir)
    print("Out dir:", out_dir)

    for fname in model_files:
        model_path = models_dir / fname
        if not model_path.exists():
            print(f"✗ Missing: {model_path}")
            continue

        # Load with Keras (NOT tf-keras legacy)
        model = keras.models.load_model(model_path)

        stem = model_path.stem  # e.g., model_a_best
        out_path = out_dir / f"{stem}_architecture.png"

        plot_model(
            model,
            to_file=str(out_path),
            show_shapes=not args.no_shapes,
            show_layer_names=not args.no_names,
            expand_nested=not args.no_expand,
            dpi=args.dpi,
        )
        print(f"✔ Saved: {out_path}")


if __name__ == "__main__":
    main()
    
    
# bash
# export PATH="$PATH:/c/Program Files/Graphviz/bin"
# python -m rockpaperscissors.visualize_models