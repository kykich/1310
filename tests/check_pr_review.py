# -*- coding: utf-8 -*-
"""Проверка AI-ревью PR (docs/Final_task.md, задача 2) — офлайн.

Покрывает:
  1. сбор diff и списка изменённых файлов (git, временный репозиторий);
  2. обрезку слишком большого diff, ошибки ревизий и пустой диапазон;
  3. запрос к RAG по путям/идентификаторам diff;
  4. промпт ревью (три обязательных раздела, diff и RAG-контекст);
  5. полный цикл с ФЕЙКОВОЙ моделью (без сети);
  6. деградацию без RAG и оформление итогового комментария;
  7. локальную модель, reasoning-модели и отключение thinking;
  8. автоопределение локального сервера из настроек UI;
  9. команду /review в чате (parse_review, _review_payload).

Запуск: python -m tests.check_pr_review
"""
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import check, section, finish, ensure_utf8  # noqa: E402
from rtk_app import pr_review, local_llm  # noqa: E402
from web import server as web_server  # noqa: E402


def _git(repo, *args):
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = "Test"
    env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    proc = subprocess.run(["git", "-C", repo] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    return proc.stdout.decode("utf-8", "replace")


def _make_repo():
    """Временный git-репозиторий: два коммита и изменения в двух файлах."""
    repo = tempfile.mkdtemp(prefix="pr_review_")
    _git(repo, "init")
    with open(os.path.join(repo, "app.py"), "w", encoding="utf-8") as fh:
        fh.write("def add(a, b):\n    return a + b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    with open(os.path.join(repo, "app.py"), "a", encoding="utf-8") as fh:
        fh.write("\n\ndef sub(a, b):\n    return a - b\n")
    with open(os.path.join(repo, "new_module.py"), "w",
              encoding="utf-8") as fh:
        fh.write("def mul(a, b):\n    return a * b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feat: add sub and mul")
    return repo


def test_collect_changes():
    section("1. Сбор diff и изменённых файлов")
    repo = _make_repo()
    changes = pr_review.collect_changes(repo, "HEAD~1", "HEAD")
    files = changes.get("files") or []
    paths = [f.get("path") for f in files]
    statuses = {f.get("path"): f.get("status") for f in files}
    check("diff собран", changes.get("ok") is True)
    check("два изменённых файла", len(files) == 2)
    check("app.py — изменён", statuses.get("app.py") == "M")
    check("new_module.py — добавлен", statuses.get("new_module.py") == "A")
    check("в diff есть новый код", "def sub" in changes.get("diff", ""))
    check("статистика непустая", bool(changes.get("stat")))
    check("пути файлов на месте", "app.py" in paths)


def test_limits_and_errors():
    section("2. Лимит diff и ошибки ревизий")
    repo = _make_repo()
    short = pr_review.collect_changes(repo, "HEAD~1", "HEAD",
                                      max_diff_chars=40)
    check("обрезка отмечена", short.get("truncated") is True)
    check("маркер обрезки в diff", "обрезан" in short.get("diff", ""))
    bad = pr_review.collect_changes(repo, "no-such-ref", "HEAD")
    check("неверная ревизия — ok=False", bad.get("ok") is False)
    check("есть описание ошибки", bool(bad.get("error")))


def test_empty_diff():
    section("2b. Пустой diff (совпадающие ревизии)")
    repo = _make_repo()
    changes = pr_review.collect_changes(repo, "HEAD", "HEAD")
    check("сборщик не падает", changes.get("ok") is True)
    check("файлов нет", changes.get("files") == [])
    check("diff пуст", not str(changes.get("diff") or "").strip())

    called = []

    def boom(messages):
        called.append(messages)
        return {"content": "ревью", "prompt_tokens": 1, "completion_tokens": 1}

    res = pr_review.review_changes(repo, "HEAD", "HEAD", use_rag=False,
                                   caller=boom)
    error = res.get("error") or ""
    check("ревью не выполняется", res.get("ok") is False)
    check("модель не вызывалась", called == [])
    check("причина — нет изменений", "Нет изменений" in error)
    check("указаны обе ревизии", "HEAD" in error and "HEAD~1" in error)
    check("назван общий коммит", "указывают на" in error)
    check("подсказка для чата", "/review HEAD~1 HEAD" in error)

    text = pr_review.format_review("x", {"base": "origin/master", "head": "HEAD",
                                         "files": []}, "Local")
    check("в шапке диапазон ревизий",
          "- диапазон: `origin/master...HEAD`" in text)

    check("локальный лимит diff < общего",
          pr_review.local_review_diff_limit(8192) < pr_review.MAX_DIFF_CHARS)
    check("лимит растёт с контекстом",
          pr_review.local_review_diff_limit(32768)
          > pr_review.local_review_diff_limit(8192))
    check("лимит не меньше нижней границы",
          pr_review.local_review_diff_limit(4096, 4000) >= 4000)


def test_rag_query():
    section("3. Запрос к RAG по путям и идентификаторам")
    query = pr_review.build_rag_query(
        [{"path": "rtk_app/rag.py"}],
        "+def calculate_total(x):\n+    return x * 2")
    check("путь файла в запросе", "rtk_app/rag.py" in query)
    check("идентификатор из diff в запросе", "calculate_total" in query)
    check("длина ограничена", len(query) <= 600)


def test_messages():
    section("4. Промпт ревью: разделы, diff, RAG")
    changes = {"files": [{"status": "M", "path": "app.py"}],
               "stat": "app.py | 2 +", "diff": "+x = 1"}
    messages = pr_review.build_messages(changes, "RAG CONTEXT")
    system = messages[0]["content"]
    user = messages[1]["content"]
    check("раздел «Потенциальные баги»", "## Потенциальные баги" in system)
    check("раздел «Архитектурные проблемы»",
          "## Архитектурные проблемы" in system)
    check("раздел «Рекомендации»", "## Рекомендации" in system)
    check("файл в промпте", "app.py" in user)
    check("RAG-контекст в промпте", "RAG CONTEXT" in user)
    check("diff в промпте", "+x = 1" in user)


def test_review_fake_model():
    section("5. Полный цикл с фейковой моделью (без сети)")
    repo = _make_repo()
    calls = []

    def caller(messages):
        calls.append(messages)
        return {"content": "## Потенциальные баги\n- нет\n"
                           "## Архитектурные проблемы\n- нет\n"
                           "## Рекомендации\n- покрыть тестами",
                "prompt_tokens": 100, "completion_tokens": 20}

    res = pr_review.review_changes(repo, "HEAD~1", "HEAD", model="Fake",
                                   use_rag=False, caller=caller)
    check("ревью ok", res.get("ok") is True)
    check("модель вызвана один раз", len(calls) == 1)
    check("в промпт попал diff", "def sub" in calls[0][1]["content"])
    check("заголовок комментария", "## AI-ревью PR" in res.get("review", ""))
    check("разделы сохранены",
          "## Потенциальные баги" in res.get("review", ""))
    check("RAG отключён", res.get("rag_used") is False)
    check("учтены файлы", len(res.get("files") or []) == 2)
    check("usage посчитан", res.get("usage", {}).get("input") == 100)


def test_rag_degradation():
    section("6. Без индекса RAG — предупреждение, ревью не падает")
    from rtk_app import rag

    class _FakeStore(object):
        def status(self):
            return {"available": False}

    old = rag.RagStore
    rag.RagStore = lambda *a, **k: _FakeStore()
    try:
        context, hits, note = pr_review.rag_context(".", "app.py")
    finally:
        rag.RagStore = old
    check("контекст пуст", context == "")
    check("фрагментов нет", hits == [])
    check("причина указана", "индекс RAG пуст" in note)

    text = pr_review.format_review("ревью", {"files": [1], "truncated": True},
                                   "M", False, note)
    check("в комментарии причина отсутствия RAG", note in text)
    check("отмечена обрезка diff", "diff обрезан" in text)


def test_local_model():
    section("7. Локальная модель (OpenAI-совместимый сервер)")
    from rtk_app.agent import Agent

    agent = Agent("")
    unknown = False
    try:
        pr_review.resolve_review_source(agent, "Local")
    except ValueError:
        unknown = True
    check("без local_url метка Local неизвестна", unknown)

    agent.set_local_model("http://127.0.0.1:8080/v1", "qwen", "Local")
    provider, name = pr_review.resolve_review_source(agent, "Local")
    check("Local -> провайдер local", provider == "local")
    check("в поле модели URL|имя",
          name.startswith("http://127.0.0.1:8080/v1|"))
    check("без метки — модель по умолчанию",
          pr_review.resolve_review_source(agent, None)[0]
          in ("gigachat", "deepseek", "local"))

    caller = pr_review._default_caller(
        "Local", local_url="http://127.0.0.1:59999/v1",
        local_model="qwen")
    check("caller для Local создан без сети", callable(caller))

    # /review с Local должен отключать reasoning (иначе Qwen3.5 тратит весь
    # max_tokens на reasoning_content и возвращает пустой content).
    captured_thinking = {}

    def fake_review_chat(self, provider, model, messages, temperature=None,
                         max_tokens=None, disable_thinking=False):
        captured_thinking["provider"] = provider
        captured_thinking["disable"] = disable_thinking
        return {"content": "ok", "prompt_tokens": 0, "completion_tokens": 0}

    orig_chat_text = Agent._chat_text
    Agent._chat_text = fake_review_chat
    try:
        pr_review._default_caller(
            "Local", local_url="http://127.0.0.1:59999/v1",
            local_model="qwen")([{"role": "user", "content": "x"}])
    finally:
        Agent._chat_text = orig_chat_text
    check("review: Local получает disable_thinking=True",
          captured_thinking.get("provider") == "local"
          and captured_thinking.get("disable") is True)

    # local_llm: пустой content + reasoning -> retry с enable_thinking=false.
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append(payload)
        if (payload.get("chat_template_kwargs") or {}).get(
                "enable_thinking") is False:
            message = {"content": "Ок"}
        else:
            message = {"content": "", "reasoning_content": "размышления"}
        return {"choices": [{"message": message}], "usage": {}}

    orig_post = local_llm._post_chat
    local_llm._post_chat = fake_post
    try:
        res = local_llm.chat([{"role": "user", "content": "привет"}],
                             model="Local", base_url="http://127.0.0.1:1/v1",
                             max_tokens=10)
        check("reasoning-empty -> retry", len(calls) == 2)
        check("retry c enable_thinking=false",
              (calls[1].get("chat_template_kwargs") or {}).get(
                  "enable_thinking") is False)
        check("ответ после retry получен", res.get("content") == "Ок")

        calls[:] = []
        res = local_llm.chat([{"role": "user", "content": "привет"}],
                             model="Local", base_url="http://127.0.0.1:1/v1",
                             max_tokens=10, disable_thinking=True)
        check("disable_thinking с первого запроса", len(calls) == 1)
        check("disable_thinking: content получен", res.get("content") == "Ок")
    finally:
        local_llm._post_chat = orig_post

    repo = _make_repo()
    res = pr_review.review_changes(repo, "HEAD~1", "HEAD",
                                   model="NoSuchModel", use_rag=False)
    check("неизвестная модель — ok=False", res.get("ok") is False)
    check("в ошибке есть имя модели",
          "NoSuchModel" in (res.get("error") or ""))


def test_local_autodiscovery():
    section("8. Автоопределение локальной модели из блока UI")
    settings = {"enabled": True,
                "params": {"port": 8123, "alias": "Local",
                           "model": r"e:\Models\llama\models\Qwen3.5-9B-Q6_K.gguf"}}
    url, model, enabled = pr_review.local_model_endpoint(settings)
    check("URL из params.port", url == "http://127.0.0.1:8123/v1")
    check("alias имеет приоритет", model == "Local")
    check("enabled прочитан", enabled is True)

    folder = tempfile.mkdtemp(prefix="local_cfg_")
    path = os.path.join(folder, "local_model.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh)

    orig_probe = pr_review.probe_local_model
    pr_review.probe_local_model = lambda url, timeout=3: ["Local"]
    try:
        info = pr_review.discover_local_model(settings_file=path)
    finally:
        pr_review.probe_local_model = orig_probe
    check("discover нашёл запущенный сервер", bool(info) and info["running"])
    check("discover: URL", (info or {}).get("url")
          == "http://127.0.0.1:8123/v1")
    check("discover: enabled", (info or {}).get("enabled") is True)

    from rtk_app.agent import Agent
    captured = {}
    orig_chat = Agent._chat_text
    orig_discover = pr_review.discover_local_model

    def fake_chat(self, provider, model, messages, temperature=None,
                  max_tokens=None, disable_thinking=False):
        captured["provider"] = provider
        captured["model"] = model
        return {"content": "ok", "prompt_tokens": 0, "completion_tokens": 0}

    Agent._chat_text = fake_chat
    try:
        pr_review.discover_local_model = lambda *a, **k: {
            "url": "http://127.0.0.1:8123/v1", "model": "Local",
            "enabled": True, "running": True}
        caller = pr_review._default_caller(None)
        caller([{"role": "user", "content": "x"}])
        check("пустая метка + enabled -> local",
              captured.get("provider") == "local")
        check("локальная модель = alias",
              str(captured.get("model") or "").endswith("|Local"))

        captured.clear()
        pr_review.discover_local_model = lambda *a, **k: {
            "url": "http://127.0.0.1:8123/v1", "model": "Local",
            "enabled": False, "running": True}
        caller = pr_review._default_caller(None)
        caller([{"role": "user", "content": "x"}])
        check("enabled=False -> облачная по умолчанию",
              captured.get("provider") == "gigachat")

        pr_review.discover_local_model = lambda *a, **k: None
        raised = False
        try:
            pr_review._default_caller("Local")
        except ValueError:
            raised = True
        check("Local без запущенного сервера — понятная ошибка", raised)
    finally:
        Agent._chat_text = orig_chat
        pr_review.discover_local_model = orig_discover


def test_chat_review_command():
    section("9. Команда /review в чате")
    spec = web_server.parse_review("/review")
    check("/review: base по умолчанию", spec["base"] == "HEAD~1")
    check("/review: head по умолчанию", spec["head"] == "HEAD")
    check("/review: RAG включён", spec["use_rag"] is True)

    spec = web_server.parse_review("/review origin/master HEAD --no-rag")
    check("ревизии из аргументов",
          spec["base"] == "origin/master" and spec["head"] == "HEAD")
    check("--no-rag отключает RAG", spec["use_rag"] is False)
    check("один аргумент — base",
          web_server.parse_review("/review main")["head"] == "HEAD")
    check("обычный текст — не команда",
          web_server.parse_review("сделай ревью") is None)

    calls = {}
    orig = pr_review.review_changes

    def fake_review(**kwargs):
        calls.update(kwargs)
        return {"ok": True,
                "review": "## AI-ревью PR\n\n## Потенциальные баги\n- нет",
                "files": [{"status": "M", "path": "app.py"}],
                "rag_used": False, "rag_hits": 0, "diff_truncated": False,
                "usage": {"input": 10, "output": 5}, "error": None,
                "stat": "app.py | 2 +"}

    pr_review.review_changes = fake_review
    try:
        payload = web_server.WebRequestHandler._review_payload(
            object(), {"base": "HEAD~1", "head": "HEAD", "use_rag": False},
            "DeepSeek-flash", r"G:\repo")
    finally:
        pr_review.review_changes = orig
    check("ревью выполнено",
          payload.get("ok") is True and payload.get("review") is True)
    check("репозиторий/ревизии переданы",
          calls.get("repo") == r"G:\repo"
          and calls.get("base") == "HEAD~1"
          and calls.get("head") == "HEAD")
    check("модель передана", calls.get("model") == "DeepSeek-flash")
    check("RAG отключён", calls.get("use_rag") is False)
    check("текст содержит разделы", "Потенциальные баги" in payload["text"])
    check("html сформирован", bool(payload.get("html")))
    check("файлы в ответе", len(payload.get("files") or []) == 1)

    # Локальная модель: diff автоматически ограничивается под ctx, иначе
    # llama-server с ctx 8192 отвечает 400 на большой diff.
    calls.clear()
    pr_review.review_changes = fake_review
    try:
        web_server.WebRequestHandler._review_payload(
            object(), {"base": "HEAD~1", "head": "HEAD", "use_rag": False},
            "Local", r"G:\repo")
    finally:
        pr_review.review_changes = orig
    check("Local: diff ограничен контекстом",
          int(calls.get("max_diff_chars") or 0) < pr_review.MAX_DIFF_CHARS)

    def fake_error(**kwargs):
        return {"ok": False, "review": "", "error": "нет такой ревизии",
                "files": [], "rag_used": False}

    pr_review.review_changes = fake_error
    try:
        payload = web_server.WebRequestHandler._review_payload(
            object(), {"base": "bad", "head": "HEAD", "use_rag": False},
            "GigaChat", r"G:\repo")
    finally:
        pr_review.review_changes = orig
    check("ошибка оформлена в чат",
          payload.get("ok") is False
          and "Ревью не выполнено" in payload["text"])
    check("причина ошибки в тексте", "нет такой ревизии" in payload["text"])


def main():
    ensure_utf8()
    test_collect_changes()
    test_limits_and_errors()
    test_empty_diff()
    test_rag_query()
    test_messages()
    test_review_fake_model()
    test_rag_degradation()
    test_local_model()
    test_local_autodiscovery()
    test_chat_review_command()
    finish()


if __name__ == "__main__":
    main()
