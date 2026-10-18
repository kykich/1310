# -*- coding: utf-8 -*-
"""AI-ревью pull request (docs/Final_task.md, задача 2).

Пайплайн:
  1. собрать diff и список изменённых файлов между base и head (git);
  2. найти контекст проекта в RAG (документация + код) по путям/идентификаторам
     из diff (если индекс доступен);
  3. отправить diff + контекст модели и получить Markdown-ревью:
     потенциальные баги, архитектурные проблемы, рекомендации.

Модуль не зависит от веб-сервера и работает в CI: ``review_pr.py`` использует
его для локального запуска и GitHub Actions.
"""
import os
import re
import subprocess

__all__ = [
    "collect_changes",
    "build_rag_query",
    "rag_context",
    "build_messages",
    "review_changes",
    "format_review",
    "resolve_review_source",
    "discover_local_model",
    "local_model_endpoint",
    "probe_local_model",
    "MAX_DIFF_CHARS",
]

# Лимит diff, отправляемого модели (символов): защита от гигантских PR.
MAX_DIFF_CHARS = 120000
# Сколько символов RAG-контекста максимум подмешивать в промпт.
MAX_RAG_CHARS = 6000
# Сколько фрагментов RAG запрашивать.
DEFAULT_TOP_K = 8
# Сколько изменённых файлов перечислять в промпте.
MAX_FILES_IN_PROMPT = 200

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Служебные слова, которые не помогают искать контекст в RAG.
_IDENT_STOP = frozenset((
    "the", "and", "for", "with", "this", "that", "from", "return", "def",
    "class", "import", "self", "none", "true", "false", "print", "str",
    "int", "float", "list", "dict", "set", "len", "var", "let", "const",
    "function", "async", "await", "try", "except", "raise", "pass", "elif",
    "else", "while", "lambda", "yield", "global", "assert", "del", "not",
    "и", "в", "на", "для", "что", "как", "это", "или", "не", "по", "из",
))


def _run_git(repo, args, timeout=120):
    """Выполняет git-команду в репозитории repo. Возвращает (stdout, error)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo)] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False)
    except FileNotFoundError:
        return None, "git не найден в PATH"
    except subprocess.TimeoutExpired:
        return None, "git: превышено время ожидания (%s c)" % timeout
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        return None, err or ("git вернул код %d" % proc.returncode)
    return out, None


def _short_rev(repo, rev):
    """Короткий SHA ревизии (для сообщений об ошибках); "" если нет."""
    out, err = _run_git(repo, ["rev-parse", "--short", str(rev or "")])
    if err or not out:
        return ""
    return out.strip().splitlines()[0]


def _parse_name_status(text):
    """Разбирает вывод `git diff --name-status` в список файлов.

    Возвращает [{status, path, old_path?}], где status — буква git
    (A/M/D/R/C/T), для переименований old_path — прежний путь.
    """
    files = []
    for line in str(text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        if status[:1] in ("R", "C") and len(parts) >= 3:
            files.append({"status": status, "path": parts[2].strip(),
                          "old_path": parts[1].strip()})
        else:
            files.append({"status": status, "path": parts[1].strip()})
    return files


def local_review_diff_limit(ctx_tokens=8192, max_tokens=4000, reserve=800):
    """Лимит diff (символов) для локальной модели с учётом её контекста.

    Локальный сервер (llama.cpp) отклоняет запрос целиком, если prompt +
    max_tokens не влезают в ctx (обычно 8192). Оставляем резерв на системный
    промпт, список файлов и RAG-контекст, а остаток переводим в символы
    (~3 символа/токен для кода и русского текста).
    """
    try:
        ctx = int(ctx_tokens or 8192)
    except (TypeError, ValueError):
        ctx = 8192
    try:
        out = int(max_tokens or 4000)
    except (TypeError, ValueError):
        out = 4000
    budget = max(1000, ctx - out - int(reserve or 0))
    return min(MAX_DIFF_CHARS, max(4000, budget * 3))


def collect_changes(repo=".", base="HEAD~1", head="HEAD",
                    max_diff_chars=MAX_DIFF_CHARS):
    """Собирает diff и изменённые файлы между base и head.

    Используется трёхточечный diff (``base...head``, как в GitHub PR):
    сравнивается с общим предком. Если merge-base недоступен — откат на
    двухточечный ``base..head``.

    Возвращает dict:
        ok, repo, base, head, diff, files[], stat, truncated, error
    """
    repo = os.path.abspath(str(repo or "."))
    base = str(base or "HEAD~1").strip() or "HEAD~1"
    head = str(head or "HEAD").strip() or "HEAD"
    diff, err = _run_git(repo, ["diff", "--no-color", "--find-renames",
                                "%s...%s" % (base, head)])
    if err:
        diff, err = _run_git(repo, ["diff", "--no-color",
                                    "%s..%s" % (base, head)])
    if err:
        return {"ok": False, "repo": repo, "base": base, "head": head,
                "diff": "", "files": [], "stat": "", "truncated": False,
                "error": err}
    names, _ = _run_git(repo, ["diff", "--name-status",
                               "%s...%s" % (base, head)])
    stat, _ = _run_git(repo, ["diff", "--stat", "%s...%s" % (base, head)])
    truncated = False
    if max_diff_chars and len(diff) > int(max_diff_chars):
        diff = diff[:int(max_diff_chars)]
        diff += ("\n\n... [diff обрезан: показаны первые %d символов]"
                 % int(max_diff_chars))
        truncated = True
    return {"ok": True, "repo": repo, "base": base, "head": head,
            "diff": diff, "files": _parse_name_status(names),
            "stat": stat or "", "truncated": truncated, "error": None}


def build_rag_query(files, diff="", limit=600):
    """Собирает поисковый запрос для RAG из путей файлов и идентификаторов.

    Пути дают привязку к модулям, идентификаторы из diff (функции, классы,
    переменные) — смысловые термины для поиска по документации и коду.
    """
    paths = []
    for item in files or []:
        path = str((item or {}).get("path") or "").strip()
        if path:
            paths.append(path)
    counts = {}
    for token in _IDENT_RE.findall(str(diff or "")):
        low = token.lower()
        if low in _IDENT_STOP:
            continue
        counts[low] = counts.get(low, 0) + 1
    top = sorted(counts, key=lambda k: (-counts[k], k))[:40]
    return " ".join(paths + top)[:int(limit or 600)]


def rag_context(repo, query, top_k=DEFAULT_TOP_K, max_chars=MAX_RAG_CHARS):
    """Ищет контекст проекта в RAG (документация + код).

    Возвращает (context_text, hits, note): note пустой при успехе, иначе
    причина, по которой контекст не получен (индекс пуст, Ollama недоступна).
    Ошибки НЕ ломают ревью — работаем без RAG.
    """
    query = str(query or "").strip()
    if not query:
        return "", [], "пустой запрос"
    try:
        from . import rag
        store = rag.RagStore()
        status = store.status()
        if not status.get("available"):
            return "", [], "индекс RAG пуст (нужна индексация)"
        context, hits = store.build_context(
            query, top_k=top_k, docs_dir=os.path.abspath(str(repo)))
        return str(context or "")[:int(max_chars)], hits, ""
    except Exception as exc:                     # noqa: BLE001
        return "", [], "RAG недоступен: %s" % exc


def build_messages(changes, rag_context_text=""):
    """Собирает сообщения для модели: diff + список файлов + RAG-контекст."""
    files = list((changes or {}).get("files") or [])
    shown = files[:MAX_FILES_IN_PROMPT]
    file_lines = []
    for item in shown:
        status = str(item.get("status") or "?").strip()
        path = str(item.get("path") or "").strip()
        old = str(item.get("old_path") or "").strip()
        line = "- %s %s" % (status, path)
        if old:
            line += " (было: %s)" % old
        file_lines.append(line)
    if len(files) > len(shown):
        file_lines.append("- ... ещё %d файлов" % (len(files) - len(shown)))
    system = "\n".join([
        "Ты — старший инженер, который ревьюит pull request.",
        "Анализируй ТОЛЬКО diff и предоставленный контекст проекта.",
        "Верни Markdown строго с тремя разделами (заголовки ##):",
        "## Потенциальные баги",
        "## Архитектурные проблемы",
        "## Рекомендации",
        "В каждом пункте укажи файл (и строку, если видно из diff),",
        "серьёзность [critical|high|medium|low], суть проблемы и что сделать.",
        "Не выдумывай факты вне diff/контекста. Если по разделу замечаний",
        "нет — напиши «нет».",
    ])
    rag_block = str(rag_context_text or "").strip() or "контекст не предоставлен"
    user = "\n".join([
        "Изменённые файлы:",
        "\n".join(file_lines) or "- (нет)",
        "",
        "Статистика: %s" % (str((changes or {}).get("stat") or "").strip()
                            or "нет"),
        "",
        "КОНТЕКСТ ПРОЕКТА (RAG: документация и код):",
        rag_block,
        "",
        "DIFF:",
        "```diff",
        str((changes or {}).get("diff") or "").strip(),
        "```",
    ])
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def format_review(content, changes=None, model="", rag_used=False,
                  rag_note=""):
    """Формирует итоговый Markdown-комментарий для PR."""
    changes = changes or {}
    files = changes.get("files") or []
    if rag_used:
        rag_text = "использован"
    elif rag_note:
        rag_text = "не использован (%s)" % rag_note
    else:
        rag_text = "не использован"
    meta = ["- модель: `%s`" % (model or "default")]
    base = str(changes.get("base") or "").strip()
    head = str(changes.get("head") or "").strip()
    if base or head:
        meta.append("- диапазон: `%s...%s`" % (base or "?", head or "?"))
    meta += ["- файлов изменено: %d" % len(files),
             "- RAG: %s" % rag_text]
    if changes.get("truncated"):
        meta.append("- diff обрезан по лимиту")
    body = str(content or "").strip() or "_Модель не вернула ревью._"
    return "\n".join(["## AI-ревью PR", ""] + meta + ["", body])


def local_model_endpoint(settings=None):
    """Возвращает (url, model, enabled) из сохранённых настроек блока.

    settings — содержимое session/local_model.json (или dict). URL строится
    из params.host/params.port (по умолчанию config.LOCAL_HOST/PORT).
    """
    from .local_server import server_url
    from . import config
    settings = settings or {}
    params = settings.get("params") or {}
    url = server_url(params.get("host"), params.get("port"))
    model = str(params.get("alias") or "").strip()
    if not model:
        path = str(params.get("model") or "").strip()
        if path:
            model = os.path.splitext(os.path.basename(path))[0]
    if not model:
        model = str(getattr(config, "LOCAL_LABEL", "Local") or "Local")
    return url, model, bool(settings.get("enabled"))


def probe_local_model(url, timeout=3):
    """GET <url>/models локального сервера. Возвращает список id моделей."""
    import json
    import urllib.request
    try:
        req = urllib.request.Request(str(url).rstrip("/") + "/models",
                                     method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [str(item.get("id")) for item in (data.get("data") or [])
                if isinstance(item, dict) and item.get("id")]
    except Exception:                            # noqa: BLE001
        return []


def discover_local_model(settings_file=None, timeout=3):
    """Находит ЗАПУЩЕННУЮ локальную модель, сохранённую блоком «Локальная модель».

    Читает session/local_model.json (как UI), проверяет /v1/models.
    Возвращает {url, model, enabled, running} или None, если сервер не отвечает.
    """
    from .local_server import LocalServerManager
    try:
        settings = LocalServerManager(settings_file).load_settings()
    except Exception:                            # noqa: BLE001
        settings = {}
    url, model, enabled = local_model_endpoint(settings)
    models = probe_local_model(url, timeout=timeout)
    if not models:
        return None
    alias = str((settings.get("params") or {}).get("alias") or "").strip()
    if alias and alias in models:
        model = alias
    elif model not in models:
        model = models[0]
    return {"url": url, "model": model, "enabled": enabled, "running": True}


def resolve_review_source(agent, model=None):
    """Определяет (provider, model) для ревью; неизвестная метка — ошибка.

    В отличие от Agent._resolve_model (мягкий откат на GigaChat), в ревью
    ошибка в имени модели не должна приводить к незаметной смене провайдера.
    """
    label = str(model or "").strip()
    if not label:
        return agent._resolve_model(None)
    src = agent.by_label.get(label)
    if src is not None:
        return src[0], src[1]
    for provider, name, _lbl, _cls in agent.sources:
        if name == label:
            return provider, name
    raise ValueError("неизвестная модель %r; доступны: %s"
                     % (label, ", ".join(agent.enabled_models) or "—"))


def _default_caller(model=None, temperature=0.2, max_tokens=4000,
                    local_url=None, local_model=None, local_label=None):
    """Возвращает функцию вызова модели (DeepSeek/GigaChat/Local) для ревью.

    local_url   — OpenAI-совместимый локальный сервер (llama.cpp
                  llama-server, Ollama, LM Studio), например
                  http://127.0.0.1:8080/v1;
    local_model — имя модели для запроса (может быть пустым);
    local_label — метка для выбора в --model (по умолчанию «Local»).

    Если local_url НЕ задан, но модель запущена через блок «Локальная модель»
    (session/local_model.json) и отвечает на /v1/models, используется она:
    при `--model Local` — всегда, при пустом `--model` — если блок включён.
    """
    from .agent import Agent
    from .key_store import read_api_key
    from . import config
    try:
        api_key = read_api_key(config.DS_KEY_FILE)
    except FileNotFoundError:
        api_key = ""
    agent = Agent(api_key)
    label = str(model or "").strip()
    label_local = str(local_label or getattr(config, "LOCAL_LABEL", "Local")
                      or "Local").strip()
    local_used = False
    if str(local_url or "").strip():
        agent.set_local_model(local_url, local_model, label_local)
        local_used = True
    elif not label or label == label_local:
        # Пробуем найти запущенный сервер, сохранённый блоком UI.
        info = discover_local_model()
        if info and (label == label_local or info.get("enabled")):
            agent.set_local_model(info["url"],
                                  local_model or info["model"], label_local)
            local_used = True
        elif label == label_local:
            raise ValueError(
                "локальная модель %r не запущена или не отвечает на /v1/models; "
                "запустите её в блоке «Локальная модель» или укажите --local-url"
                % label)
    resolve_label = label or (label_local if local_used else None)
    provider, name = resolve_review_source(agent, resolve_label)

    def _call(messages):
        return agent._chat_text(provider, name, messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                disable_thinking=(provider == "local"))

    return _call


def review_changes(repo=".", base="HEAD~1", head="HEAD", model=None,
                   use_rag=True, caller=None, max_tokens=4000,
                   max_diff_chars=MAX_DIFF_CHARS, top_k=DEFAULT_TOP_K,
                   local_url=None, local_model=None, local_label=None):
    """Полный цикл ревью PR: diff -> RAG -> модель -> Markdown-ревью.

    caller — функция (messages -> {content, prompt_tokens, completion_tokens});
    по умолчанию вызывается модель (DeepSeek/GigaChat или локальная, если
    задан local_url). Возвращает dict:
    ok, review, raw, files, stat, diff_truncated, rag_used, rag_hits,
    rag_note, usage, error.
    """
    changes = collect_changes(repo, base, head, max_diff_chars=max_diff_chars)
    if not changes.get("ok"):
        return {"ok": False, "error": changes.get("error"), "review": "",
                "raw": "", "files": [], "stat": "", "diff_truncated": False,
                "rag_used": False, "rag_hits": 0, "rag_note": "",
                "usage": None}
    # Пустой diff (например, /review origin/master, когда origin/master == HEAD):
    # ревьюить нечего — не тратим токены и объясняем, что именно совпало.
    if not str(changes.get("diff") or "").strip() and not changes.get("files"):
        base_sha = _short_rev(changes["repo"], changes["base"])
        head_sha = _short_rev(changes["repo"], changes["head"])
        if base_sha and head_sha and base_sha == head_sha:
            reason = "обе ревизии указывают на коммит %s" % base_sha
        elif base_sha and head_sha:
            reason = ("коммиты %s и %s не отличаются по содержимому"
                      % (base_sha, head_sha))
        else:
            reason = "изменений нет"
        error = ("Нет изменений между %s и %s (%s). Укажите разные ревизии: "
                 "--base HEAD~1 --head HEAD (в чате: /review HEAD~1 HEAD)."
                 % (changes["base"], changes["head"], reason))
        return {"ok": False, "error": error, "review": "", "raw": "",
                "files": [], "stat": changes.get("stat") or "",
                "diff_truncated": False, "rag_used": False, "rag_hits": 0,
                "rag_note": "", "usage": None}
    rag_text, hits, rag_note = "", [], "отключён"
    if use_rag:
        query = build_rag_query(changes["files"], changes["diff"])
        rag_text, hits, rag_note = rag_context(changes["repo"], query,
                                               top_k=top_k)
    messages = build_messages(changes, rag_text)
    try:
        call = caller or _default_caller(
            model, max_tokens=max_tokens, local_url=local_url,
            local_model=local_model, local_label=local_label)
        res = call(messages)
    except Exception as exc:                     # noqa: BLE001
        where = (" (локальная модель %s)" % local_url) if local_url else ""
        return {"ok": False, "error": "модель недоступна%s: %s" % (where, exc),
                "review": "", "raw": "", "files": changes["files"],
                "stat": changes["stat"],
                "diff_truncated": changes["truncated"],
                "rag_used": bool(rag_text), "rag_hits": len(hits),
                "rag_note": rag_note, "usage": None}
    res = res or {}
    content = str(res.get("content") or "").strip()
    review = format_review(content, changes, model, bool(rag_text), rag_note)
    usage = {"input": int(res.get("prompt_tokens", 0) or 0),
             "output": int(res.get("completion_tokens", 0) or 0)}
    return {"ok": bool(content), "review": review, "raw": content,
            "files": changes["files"], "stat": changes["stat"],
            "diff_truncated": changes["truncated"],
            "rag_used": bool(rag_text), "rag_hits": len(hits),
            "rag_note": rag_note, "usage": usage,
            "error": None if content else "модель вернула пустой ответ"}


