"""MCP-сервер «Проект» — доступ ассистента к репозиторию (docs/Final_task.md).

Задача 1: «Через MCP подключите ассистента к проекту: минимум — получение
текущей git-ветки, по желанию — список файлов или diff».

Инструменты (ТОЛЬКО ЧТЕНИЕ — репозиторий не изменяется):
  * ping        — проверка живости сервера;
  * git_branch  — текущая ветка, короткий HEAD, upstream, число изменений;
  * git_status  — `git status --porcelain` (изменённые/новые файлы);
  * git_diff    — diff рабочего дерева, `--staged` или против base-ревизии;
  * list_files  — список файлов проекта (фильтры: подпапка и расширения).

Все git-инструменты принимают необязательный `repo` — путь к репозиторию
(по умолчанию проект ассистента). Это позволяет команде /help брать ветку
того проекта, чья папка проиндексирована в RAG.

Запуск (как stdio-подпроцесс MCP):
    python project_server.py
"""

import os
import subprocess

from mcp.server.mcpserver import MCPServer

# Корень проекта — папка, где лежит этот файл (репозиторий целиком).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mcp = MCPServer("project")

# Таймаут git-команды (секунд) и лимит вывода (символов): защита от
# зависания и от гигантского diff в ответе.
GIT_TIMEOUT = 20
MAX_OUTPUT = 20000

# Папки, которые не показываем в списке файлов (служебные/бинарные).
SKIP_DIRS = frozenset((
    ".git", ".idea", ".vscode", ".venv", "venv", "env",
    "node_modules", "__pycache__", "session", "rag_images",
))


# Текст-заглушка: в целевой папке нет данных git (не репозиторий).
NO_GIT_TEXT = "Нет данных о GIT"


def _no_git(repo=""):
    """Текст заглушки для папки без git-репозитория."""
    path = os.path.abspath(str(repo or BASE_DIR))
    return "%s: папка %s не является git-репозиторием" % (NO_GIT_TEXT, path)


def _err_text(err):
    """Текст ошибки инструмента: заглушка «нет git» идёт без префикса."""
    err = str(err or "")
    if err.startswith(NO_GIT_TEXT):
        return err
    return "Ошибка: %s" % err


def _resolve_repo(repo=""):
    """Каталог git-репозитория для repo (сам путь или его предок с .git).

    repo="" — репозиторий проекта (BASE_DIR). Возвращает абсолютный путь
    корня репозитория или None, если git-репозиторий не найден.
    """
    start = os.path.abspath(str(repo or BASE_DIR))
    if not os.path.isdir(start):
        return None
    cur = start
    while True:
        dot_git = os.path.join(cur, ".git")
        # В обычном репозитории .git — папка, в linked worktree — файл
        # (gitdir: ...). Оба варианта означают корень репозитория.
        if os.path.isdir(dot_git) or os.path.isfile(dot_git):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _git(args, limit=MAX_OUTPUT, repo=""):
    """Выполняет git-команду в репозитории repo. Возвращает (stdout, error)."""
    cwd = _resolve_repo(repo)
    if not cwd:
        return None, _no_git(repo)
    try:
        proc = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        return None, "git не найден в PATH"
    except subprocess.TimeoutExpired:
        return None, "git: превышено время ожидания (%d c)" % GIT_TIMEOUT
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        return None, err or ("git вернул код %d" % proc.returncode)
    if len(out) > limit:
        out = out[:limit] + "\n… (вывод обрезан)"
    return out, None


@mcp.tool()
def ping() -> str:
    """Проверка живости сервера «Проект»."""
    return "pong: project_server (корень: %s)" % BASE_DIR


@mcp.tool()
def git_branch(repo: str = "") -> str:
    """Текущая git-ветка проекта, короткий HEAD и число изменённых файлов.

    repo — путь к репозиторию (по умолчанию — проект ассистента). Можно
    передать папку внутри репозитория — корень найдётся подъёмом по дереву.
    """
    branch, err = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo=repo)
    if err:
        return _err_text(err)
    head, _ = _git(["rev-parse", "--short", "HEAD"], repo=repo)
    upstream, _ = _git(["rev-parse", "--abbrev-ref",
                        "--symbolic-full-name", "@{u}"], repo=repo)
    status, _ = _git(["status", "--porcelain"], repo=repo)
    changed = len([line for line in (status or "").splitlines()
                   if line.strip()])
    parts = ["ветка: %s" % branch, "HEAD: %s" % (head or "?")]
    if upstream:
        parts.append("upstream: %s" % upstream)
    parts.append("изменённых файлов: %d" % changed)
    root = _resolve_repo(repo)
    if root:
        parts.append("репозиторий: %s" % root)
    return "\n".join(parts)


@mcp.tool()
def git_status(repo: str = "") -> str:
    """Изменённые и новые файлы проекта (`git status --porcelain`)."""
    out, err = _git(["status", "--porcelain"], repo=repo)
    if err:
        return _err_text(err)
    if not out:
        return "рабочее дерево чисто (нет изменений)"
    return out


@mcp.tool()
def git_diff(staged: bool = False, base: str = "", repo: str = "") -> str:
    """Diff проекта: рабочее дерево (по умолчанию), --staged или против base.

    staged — показать проиндексированные изменения;
    base   — ревизия/ветка для сравнения (например, «origin/master»);
    repo   — путь к репозиторию (по умолчанию — проект ассистента).
    """
    args = ["diff", "--no-color"]
    if staged:
        args.append("--staged")
    if base:
        args.append(str(base))
    out, err = _git(args, repo=repo)
    if err:
        return _err_text(err)
    if not out:
        return "изменений нет"
    return out


@mcp.tool()
def list_files(subdir: str = "", ext: str = "", limit: int = 200,
               repo: str = "") -> str:
    """Список файлов проекта (относительные пути, без служебных папок).

    subdir — подпапка внутри проекта ("" — весь проект);
    ext    — расширения через запятую (например, ".py,.md"); пусто — все;
    limit  — максимум строк (1..1000);
    repo   — путь к репозиторию (по умолчанию — проект ассистента).
    """
    try:
        limit = max(1, min(1000, int(limit or 200)))
    except (TypeError, ValueError):
        limit = 200
    root = _resolve_repo(repo)
    if not root:
        return _no_git(repo)
    target = os.path.realpath(os.path.join(root, str(subdir or "")))
    if target != root and not target.startswith(root + os.sep):
        return "Ошибка: путь вне проекта"
    if not os.path.isdir(target):
        return "Ошибка: папка не найдена"
    exts = []
    for item in str(ext or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.append(item if item.startswith(".") else "." + item)
    found = []
    for cur, dirs, names in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS
                         and not d.startswith("."))
        for name in sorted(names):
            if exts and not name.lower().endswith(tuple(exts)):
                continue
            full = os.path.join(cur, name)
            found.append(os.path.relpath(full, root).replace(os.sep, "/"))
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    if not found:
        return "файлов не найдено"
    return "\n".join(found)


if __name__ == "__main__":
    # Транспорт по умолчанию — stdio (как подпроцесс MCP-клиента проекта).
    mcp.run()
