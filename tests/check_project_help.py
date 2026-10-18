# -*- coding: utf-8 -*-
"""Проверка задачи 1 из docs/Final_task.md (ассистент разработчика).

Покрывает БЕЗ сети и без LLM:
  1. RAG по документации и коду: выбор файлов (.md/.py), чтение текста,
     чанки без «страниц», относительные пути источников;
  2. MCP-сервер «Проект»: git_branch / git_status / git_diff / list_files
     (только чтение, локальный git);
  3. Команду /help: разбор запроса и детерминированная справка без модели.

Запуск: python -m tests.check_project_help
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import check, section, finish, ensure_utf8  # noqa: E402
from rtk_app import rag  # noqa: E402
from rtk_app.agent import Agent  # noqa: E402

import project_server  # noqa: E402
from web import server as web_server  # noqa: E402


def _tmp_tree():
    """Временное дерево проекта: README, docs/*.md, код и служебные папки."""
    root = tempfile.mkdtemp(prefix="proj_help_")
    os.makedirs(os.path.join(root, "docs"))
    os.makedirs(os.path.join(root, "pkg"))
    os.makedirs(os.path.join(root, ".git"))
    os.makedirs(os.path.join(root, "session"))
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Проект\n\nОписание проекта и быстрый старт.\n")
    with open(os.path.join(root, "docs", "guide.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# Руководство\n\nКак устроен RAG и MCP.\n")
    with open(os.path.join(root, "pkg", "mod.py"), "w",
              encoding="utf-8") as fh:
        fh.write("def hello():\n    return 'world'\n")
    with open(os.path.join(root, ".git", "config"), "w",
              encoding="utf-8") as fh:
        fh.write("[core]\n")
    with open(os.path.join(root, "session", "hidden.md"), "w",
              encoding="utf-8") as fh:
        fh.write("служебная сессия\n")
    with open(os.path.join(root, "logo.png"), "wb") as fh:
        fh.write(b"\x89PNG\r\n")
    # Файл с секретом: не должен попадать в индекс RAG.
    with open(os.path.join(root, "apidpsk.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("sk-secret-key\n")
    return root


def test_doc_selection():
    section("1. RAG: выбор документов и кода")
    root = _tmp_tree()
    store = rag.RagStore(index_file=os.path.join(root, "i.db"),
                         settings_file=os.path.join(root, "s.json"))
    files = [os.path.relpath(p, root).replace(os.sep, "/")
             for p in store._doc_files(root)]
    check("README.md найден", "README.md" in files)
    check("docs/*.md найден", "docs/guide.md" in files)
    check("код .py найден", "pkg/mod.py" in files)
    check(".git пропущен", not any(x.startswith(".git/") for x in files))
    check("session пропущен", not any(x.startswith("session/") for x in files))
    check("двоичный .png пропущен", not any(x.endswith(".png") for x in files))
    check("файл с секретом (apidpsk.txt) пропущен",
          "apidpsk.txt" not in files)

    exts = rag.doc_extensions()
    check("extensions: pdf/md/py", ".pdf" in exts and ".md" in exts
          and ".py" in exts)
    check("is_pdf: README.md — False", rag.is_pdf("README.md") is False)
    check("is_pdf: doc.PDF — True", rag.is_pdf("doc.PDF") is True)


def test_text_extract():
    section("2. RAG: чтение текстового файла и чанки без страниц")
    root = _tmp_tree()
    md = os.path.join(root, "README.md")
    text, err = rag.extract_text_file(md)
    check("текст прочитан без ошибки", err is None)
    check("текст содержит заголовок", "Проект" in text)
    check("несуществующий файл — ошибка",
          rag.extract_text_file(os.path.join(root, "no.md"))[1] is not None)

    empty = os.path.join(root, "empty.md")
    with open(empty, "w", encoding="utf-8") as fh:
        fh.write("   \n")
    check("пустой файл — ошибка", rag.extract_text_file(empty)[1] is not None)

    chunks = rag.RagStore._file_chunks([text], first_page=0)
    check("чанки построены", len(chunks) >= 1)
    check("у текстового файла page=0", all(c["page"] == 0 for c in chunks))
    check("нумерация чанков с 1", chunks[0]["chunk"] == 1)
    cid = rag.format_citation_id("README.md", chunks[0]["page"],
                                 chunks[0]["chunk"])
    check("id без «стр.» для текста", cid == "README.md#chunk.1")


def test_sources_paths():
    section("3. RAG: источники с относительным путём")
    hits = [
        {"file": r"G:\proj\docs\guide.md", "page": 0, "chunk": 2,
         "score": 0.9},
        {"file": r"G:\proj\README.md", "page": 0, "chunk": 1, "score": 0.5},
    ]
    srcs = rag.format_sources(hits, docs_dir=r"G:\proj")
    by_file = {s["file"]: s for s in srcs}
    check("два уникальных источника", len(srcs) == 2)
    check("path у guide.md", by_file["guide.md"]["path"] == "docs/guide.md")
    check("path у README.md", by_file["README.md"]["path"] == "README.md")
    check("pages пустые (текст)", by_file["guide.md"]["pages"] == [])
    check("citations собраны",
          by_file["guide.md"]["citations"] == ["guide.md#chunk.2"])

    # Старое поведение без docs_dir сохраняется (совместимость тестов).
    old = rag.format_sources([{"file": "/docs/a.pdf", "page": 2, "chunk": 1}])
    check("без docs_dir path=file", old[0]["path"] == "a.pdf")

    html = Agent._render_sources_html(srcs).replace("&#x2F;", "/")
    check("ссылка использует относительный путь",
          "/docs/docs/guide.md" in html)


def test_project_server():
    section("4. MCP «Проект»: git branch / status / files")
    check("ping отвечает", "pong" in project_server.ping())
    branch = project_server.git_branch()
    check("git_branch вернул ветку", "ветка:" in branch)
    check("git_branch вернул HEAD", "HEAD:" in branch)
    status = project_server.git_status()
    check("git_status вернул текст", isinstance(status, str) and bool(status))
    diff = project_server.git_diff()
    check("git_diff не падает", isinstance(diff, str))
    files = project_server.list_files(ext=".md", limit=5)
    check("list_files видит README.md", "README.md" in files)
    check("list_files не показывает .git", ".git/" not in files)
    check("list_files: путь вне проекта отклонён",
          "вне проекта" in project_server.list_files(subdir="..\\.."))
    # repo=... — git-инструменты работают с указанным репозиторием:
    # подпапка проекта должна найти корень репозитория подъёмом по дереву.
    sub = project_server.git_branch(
        repo=os.path.join(project_server.BASE_DIR, "rtk_app"))
    check("repo=подпапка находит корень репозитория",
          "ветка:" in sub and "репозиторий:" in sub)
    missing = project_server.git_branch(
        repo=os.path.join(tempfile.gettempdir(), "no_git_repo_xyz"))
    check("repo без .git — заглушка «Нет данных о GIT»",
          missing.startswith(project_server.NO_GIT_TEXT))

    # Linked worktree: .git — файл (gitdir: ...), а не папка.
    wt_repo = tempfile.mkdtemp(prefix="wt_repo_")
    wt_path = wt_repo + "_wt"
    env = dict(os.environ)
    env.update({"GIT_AUTHOR_NAME": "Test", "GIT_COMMITTER_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.com",
                "GIT_COMMITTER_EMAIL": "test@example.com"})

    def _run(*args):
        proc = subprocess.run(["git", "-C", wt_repo] + list(args),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace"))

    _run("init")
    with open(os.path.join(wt_repo, "README.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# worktree test\n")
    _run("add", "-A")
    _run("commit", "-m", "init")
    _run("worktree", "add", "-b", "wtbranch", wt_path)
    wt_branch = project_server.git_branch(repo=wt_path)
    check("worktree (.git-файл) — ветка найдена", "ветка: wtbranch" in wt_branch)
    check("worktree — корень репозитория определён",
          "репозиторий:" in wt_branch)


def test_help_parse():
    section("5. Команда /help: разбор запроса")
    check("/help без вопроса → пустая строка",
          web_server.parse_help("/help") == "")
    check("/help с вопросом",
          web_server.parse_help("/help как устроен RAG?") ==
          "как устроен RAG?")
    check("/HELP регистр не важен",
          web_server.parse_help("  /HELP структура проекта") ==
          "структура проекта")
    check("/help: двоеточие после команды",
          web_server.parse_help("/help: где README") == "где README")
    check("обычный вопрос — не /help",
          web_server.parse_help("расскажи про /help") is None)
    check("слово help без слэша — не команда",
          web_server.parse_help("help me") is None)


def test_help_payload():
    section("6. /help без модели: детерминированная справка")

    class _FakeSession(object):
        def rag_status(self):
            return {"files": 2, "chunks": 10, "available": True}

    class _FakeHandler(object):
        session = _FakeSession()

    hits = [{"file": r"G:\proj\README.md", "page": 0, "chunk": 3,
             "score": 0.8123, "text": "Описание проекта и быстрый старт."}]
    payload = web_server.WebRequestHandler._help_payload(
        _FakeHandler(), "как устроен проект", "ветка: master\nHEAD: abc123",
        hits)
    text = payload.get("text") or ""
    check("ok=True", payload.get("ok") is True)
    check("help=True", payload.get("help") is True)
    check("есть git-ветка", "ветка: master" in text)
    check("есть структура проекта", "Структура проекта" in text)
    check("есть состояние RAG",
          "файлов 2" in text and "фрагментов 10" in text)
    check("есть найденный фрагмент", "README.md#chunk.3" in text)
    check("текст фрагмента показан", "Описание проекта" in text)
    check("LLM не вызывалась (answers пуст)", payload.get("answers") == [])
    brief = web_server.project_brief()
    check("секретные файлы не показаны в структуре",
          "apidpsk.txt" not in brief and "gigakey.txt" not in brief)
    check("служебные папки не показаны", "session/" not in brief)


def test_inline_code():
    section("7. HTML: подчёркивания в inline-коде не ломаются")
    from rtk_app import html_report
    html = html_report.apply_inline_markdown(
        "`project_server.py`, `git_branch` и `COMPOSITION_*`")
    check("имя файла не разрезано", "project<em>" not in html)
    check("подчёркивание сохранено", "project_server.py" in html)
    check("внутри кода нет <em>", "<em>" not in html)
    check("жирный текст работает",
          "<strong>жирный</strong>" in
          html_report.apply_inline_markdown("**жирный**"))
    check("курсив вне кода работает",
          "<em>курсив</em>" in html_report.apply_inline_markdown("_курсив_"))


def test_search_scope():
    section("8. Поиск ограничен папкой docs_dir (общий индекс)")
    root = tempfile.mkdtemp(prefix="rag_scope_")
    proj_a = os.path.join(root, "proj_a")
    proj_b = os.path.join(root, "proj_b")
    os.makedirs(proj_a)
    os.makedirs(proj_b)
    store = rag.RagStore(index_file=os.path.join(root, "i.db"),
                         settings_file=os.path.join(root, "s.json"))
    vec = [1.0] + [0.0] * (rag.config.RAG_EMBED_DIM - 1)
    blob = rag._pack_vector(vec)
    conn = store._conn()
    conn.executemany(
        "INSERT INTO chunks(file, page, chunk, text, vec) VALUES (?,?,?,?,?)",
        [(os.path.join(proj_a, "README.md"), 0, 1, "альфа проект", blob),
         (os.path.join(proj_b, "README.md"), 0, 1, "бета проект", blob)])
    conn.commit()
    orig = rag.embed_text

    def _fake(text, model=None, timeout=None):
        return vec

    rag.embed_text = _fake
    try:
        all_hits = store.search_multi(["проект"], top_k=5)
        a_hits = store.search_multi(["проект"], top_k=5, docs_dir=proj_a)
        b_hits = store.search_multi(["проект"], top_k=5, docs_dir=proj_b)
        none_hits = store.search_multi(["проект"], top_k=5,
                                       docs_dir=os.path.join(root, "other"))
    finally:
        rag.embed_text = orig
    check("без docs_dir — оба проекта", len(all_hits) == 2)
    check("docs_dir=A — только A",
          len(a_hits) == 1 and os.path.dirname(a_hits[0]["file"]) == proj_a)
    check("docs_dir=B — только B",
          len(b_hits) == 1 and os.path.dirname(b_hits[0]["file"]) == proj_b)
    check("папка вне индекса — пусто", none_hits == [])


def test_git_stub():
    section("9. Нет git в целевой папке → заглушка без подмены репозитория")
    calls = []

    class _FakeMcp(object):
        @staticmethod
        def mcp_call_tool(name, args=None, timeout=None, server_id=None):
            calls.append(dict(args or {}))
            if args and args.get("repo"):
                return {"ok": True, "is_error": False, "error": None,
                        "text": project_server.NO_GIT_TEXT
                                + ": папка не является git-репозиторием"}
            return {"ok": True, "is_error": False, "error": None,
                    "text": "ветка: master"}

    old = web_server.mcp_client
    web_server.mcp_client = _FakeMcp
    try:
        note = web_server.project_git_note(r"C:\no_git_folder")
        calls_no_dir = len(calls)
        note_default = web_server.project_git_note(None)
    finally:
        web_server.mcp_client = old

    check("заглушка «Нет данных о GIT»", note == web_server.NO_GIT_TEXT)
    check("fallback на проект НЕ выполняется", calls_no_dir == 1)
    check("в MCP передан repo целевой папки",
          calls and calls[0].get("repo") == r"C:\no_git_folder")
    check("без docs_dir — репозиторий ассистента",
          note_default == "ветка: master" and calls[-1] == {})


def test_tool_defaults():
    section("10. MCP: целевая папка RAG подставляется в git-инструменты")
    from rtk_app.agent import apply_tool_defaults
    target = r"G:\9potok\1310"
    check("git_status получает repo",
          apply_tool_defaults("git_status", {}, {"repo": target})
          == {"repo": target})
    check("git_diff получает repo при пустых аргументах",
          apply_tool_defaults("git_diff", None, {"repo": target})
          == {"repo": target})
    check("явный repo имеет приоритет",
          apply_tool_defaults("git_branch", {"repo": r"G:\other"},
                              {"repo": target})["repo"] == r"G:\other")
    check("пустой repo заполняется",
          apply_tool_defaults("git_status", {"repo": ""},
                              {"repo": target})["repo"] == target)
    check("list_files получает repo и сохраняет ext",
          apply_tool_defaults("list_files", {"ext": ".py"},
                              {"repo": target})
          == {"ext": ".py", "repo": target})
    check("не-git инструмент не меняется",
          apply_tool_defaults("show_board", {}, {"repo": target}) == {})
    check("без defaults аргументы не меняются",
          apply_tool_defaults("git_status", {}, None) == {})


def test_local_default():
    section("11. Локальная модель по умолчанию (блок UI)")
    orig_mgr_fn = web_server._local_manager
    saved_agent = web_server._ServerState.agent

    class _Mgr(object):
        def __init__(self):
            self.running = True
            self.ready = True

        def is_running(self):
            return self.running

        def status(self):
            return {"ready": self.ready, "models": ["Local"],
                    "url": "http://127.0.0.1:8080/v1"}

        def load_settings(self):
            return {"enabled": False,
                    "params": {"port": 8080, "model": r"e:\m\q.gguf"}}

    class _Agent(object):
        def __init__(self):
            self.args = None

        def set_local_model(self, url=None, model=None, label=None):
            self.args = (url, model, label)

    mgr = _Mgr()
    web_server._local_manager = lambda: mgr
    try:
        check("сервер готов -> метка Local",
              web_server._running_local_label() == "Local")
        mgr.running = False
        check("сервер не запущен -> пусто",
              web_server._running_local_label() == "")
        mgr.running = True
        mgr.ready = False
        check("сервер не готов -> пусто",
              web_server._running_local_label() == "")

        # Запущенный сервер добавляет источник Local даже при enabled=false
        # (модель поднята через блок UI — должна использоваться).
        mgr.ready = True
        agent = _Agent()
        web_server._ServerState.agent = agent
        web_server._sync_agent_local()
        check("running+enabled=false -> Local добавлен",
              agent.args is not None
              and agent.args[0] == "http://127.0.0.1:8080/v1")
        check("имя модели из /v1/models", agent.args[1] == "Local")

        mgr.running = False
        agent2 = _Agent()
        web_server._ServerState.agent = agent2
        web_server._sync_agent_local()
        check("не запущена и не включена -> источник убран",
              agent2.args is not None and agent2.args[0] is None)
    finally:
        web_server._local_manager = orig_mgr_fn
        web_server._ServerState.agent = saved_agent


def main():
    ensure_utf8()
    test_doc_selection()
    test_text_extract()
    test_sources_paths()
    test_project_server()
    test_help_parse()
    test_help_payload()
    test_inline_code()
    test_search_scope()
    test_git_stub()
    test_tool_defaults()
    test_local_default()
    finish()


if __name__ == "__main__":
    main()
