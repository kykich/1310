# -*- coding: utf-8 -*-
"""CLI: AI-ревью pull request (docs/Final_task.md, задача 2).

Примеры:
    python review_pr.py --base origin/master --head HEAD
    python review_pr.py --base HEAD~1 --head HEAD --model DeepSeek-flash
    python review_pr.py --base HEAD~1 --head HEAD --build-index --out review.md
    python review_pr.py --base HEAD~1 --head HEAD --no-rag

В GitHub Actions base/head — SHA из события pull_request. Ключи моделей
читаются из apidpsk.txt / gigakey.txt (в CI создаются из секретов).
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtk_app import pr_review  # noqa: E402


def build_index(repo):
    """Строит RAG-индекс по папке проекта (документация + код + PDF).

    Ошибки индексации НЕ фатальны: ревью продолжится без RAG.
    """
    try:
        from rtk_app.rag import RagStore
        report = RagStore().index_docs(docs_dir=os.path.abspath(repo),
                                       verbose=True)
        print("[review] индекс: files=%s chunks=%s errors=%s"
              % (report.get("files"), report.get("chunks"),
                 report.get("errors")), file=sys.stderr)
        return report
    except Exception as exc:                     # noqa: BLE001
        print("[review] индексация не удалась (работаем без RAG): %s" % exc,
              file=sys.stderr)
        return {}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AI-ревью pull request (diff + RAG: документация и код)")
    parser.add_argument("--repo", default=".", help="путь к репозиторию")
    parser.add_argument("--base", default="HEAD~1",
                        help="базовая ревизия (SHA/ветка), по умолчанию HEAD~1")
    parser.add_argument("--head", default="HEAD",
                        help="голова изменений (SHA/ветка), по умолчанию HEAD")
    parser.add_argument("--model", default=None,
                        help="метка модели: DeepSeek-flash / GigaChat / Local "
                             "(если не указана и локальная модель включена — "
                             "используется она)")
    parser.add_argument("--no-rag", action="store_true",
                        help="не использовать RAG-контекст")
    parser.add_argument("--build-index", action="store_true",
                        help="(пере)индексировать папку проекта перед ревью")
    parser.add_argument("--out", default="",
                        help="файл для Markdown-ревью (например, review.md)")
    parser.add_argument("--max-diff-chars", type=int,
                        default=pr_review.MAX_DIFF_CHARS,
                        help="лимит diff в символах")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="лимит токенов ответа модели")
    parser.add_argument("--top-k", type=int, default=pr_review.DEFAULT_TOP_K,
                        help="сколько фрагментов RAG подмешивать")
    parser.add_argument("--local-url", default=os.environ.get(
                            "REVIEW_LOCAL_URL", ""),
                        help="URL локальной OpenAI-совместимой модели "
                             "(llama-server/Ollama/LM Studio). Если не указан — "
                             "берётся запущенная модель из блока «Локальная "
                             "модель» (session/local_model.json)")
    parser.add_argument("--local-model", default=os.environ.get(
                            "REVIEW_LOCAL_MODEL", ""),
                        help="имя модели локального сервера")
    parser.add_argument("--local-label", default=os.environ.get(
                            "REVIEW_LOCAL_LABEL", "Local"),
                        help="метка локальной модели для --model "
                             "(по умолчанию Local)")
    args = parser.parse_args(argv)

    model = args.model
    if args.local_url and not model:
        model = args.local_label

    if args.build_index:
        build_index(args.repo)

    result = pr_review.review_changes(
        repo=args.repo, base=args.base, head=args.head, model=model,
        use_rag=not args.no_rag, max_tokens=args.max_tokens,
        max_diff_chars=args.max_diff_chars, top_k=args.top_k,
        local_url=(args.local_url or None),
        local_model=(args.local_model or None),
        local_label=args.local_label)

    review = result.get("review") or ""
    if not review:
        error = " ".join(str(result.get("error") or "неизвестная ошибка")
                         .split())
        review = "## AI-ревью PR\n\n_Ревью не выполнено: %s_" % error
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            fh.write(review)
        print("[review] записано: %s" % args.out, file=sys.stderr)
    print(review)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
