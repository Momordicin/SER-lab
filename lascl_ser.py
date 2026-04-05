import argparse

from lascl.config import LaSCLConfig
from lascl.train import train_lascl
from lascl.eval import evaluate_lascl


def main():
    parser = argparse.ArgumentParser(description="LaSCL-lite SER with WavLM + label text embeddings")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train LaSCL-lite")
    p_train.add_argument("--data_root", default="dataset_split_loso_4cls")
    p_train.add_argument("--out_dir", default="lascl_ckpt")
    p_train.add_argument("--audio_model_name", default="microsoft/wavlm-base-plus")
    p_train.add_argument("--text_model_name", default="roberta-base")
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--eval_batch_size", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--audio_lr", type=float, default=1e-5)
    p_train.add_argument("--head_lr", type=float, default=1e-4)
    p_train.add_argument("--max_seconds", type=float, default=6.0)
    p_train.add_argument("--temperature", type=float, default=0.07)
    p_train.add_argument("--lambda_ce", type=float, default=0.5)
    p_train.add_argument("--lambda_scl", type=float, default=0.5)
    p_train.add_argument("--lambda_label_div", type=float, default=0.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--max_train_items", type=int, default=None)
    p_train.add_argument("--max_val_items", type=int, default=None)

    p_eval = sub.add_parser("eval", help="Evaluate LaSCL-lite checkpoint")
    p_eval.add_argument("--data_root", default="dataset_split_loso_4cls")
    p_eval.add_argument("--ckpt_path", required=True)
    p_eval.add_argument("--audio_model_name", default="microsoft/wavlm-base-plus")
    p_eval.add_argument("--text_model_name", default="roberta-base")
    p_eval.add_argument("--eval_batch_size", type=int, default=4)
    p_eval.add_argument("--max_seconds", type=float, default=6.0)
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--mode", default="nearest", choices=["nearest", "classifier"])

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = LaSCLConfig()
        cfg.audio_model_name = args.audio_model_name
        cfg.text_model_name = args.text_model_name
        cfg.batch_size = args.batch_size
        cfg.eval_batch_size = args.eval_batch_size
        cfg.epochs = args.epochs
        cfg.audio_lr = args.audio_lr
        cfg.head_lr = args.head_lr
        cfg.max_seconds = args.max_seconds
        cfg.temperature = args.temperature
        cfg.lambda_ce = args.lambda_ce
        cfg.lambda_scl = args.lambda_scl
        cfg.lambda_label_div = args.lambda_label_div
        cfg.seed = args.seed
        cfg.label_texts = ["angry", "happy", "neutral", "sad"]

        train_lascl(
            data_root=args.data_root,
            out_dir=args.out_dir,
            cfg=cfg,
            max_train_items=args.max_train_items,
            max_val_items=args.max_val_items,
        )
        return

    if args.cmd == "eval":
        cfg = LaSCLConfig()
        cfg.audio_model_name = args.audio_model_name
        cfg.text_model_name = args.text_model_name
        cfg.eval_batch_size = args.eval_batch_size
        cfg.max_seconds = args.max_seconds
        cfg.label_texts = ["angry", "happy", "neutral", "sad"]

        evaluate_lascl(
            data_root=args.data_root,
            ckpt_path=args.ckpt_path,
            cfg=cfg,
            split=args.split,
            mode=args.mode,
        )
        return


if __name__ == "__main__":
    main()
