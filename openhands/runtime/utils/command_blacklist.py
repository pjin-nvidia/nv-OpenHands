"""Command blacklist for blocking dangerous commands in the runtime.

This module provides a mechanism to block potentially dangerous commands
that could harm the runtime environment, such as killing the action executor
server or destroying critical system resources.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from openhands.core.logger import openhands_logger as logger


@dataclass
class BlacklistEntry:
    """A blacklisted command pattern with associated feedback."""

    pattern: str  # Regex pattern to match
    feedback: str  # Feedback message to return when blocked
    description: str  # Human-readable description of what this blocks
    enabled: bool = True  # Whether this rule is active


BLACKLIST_ENABLED = os.getenv(
    "OPENHANDS_COMMAND_BLACKLIST_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
)

# The blacklist of dangerous command patterns
COMMAND_BLACKLIST: list[BlacklistEntry] = [
    # === Block ALL killall commands (killall always targets by name, not PID) ===
    BlacklistEntry(
        pattern=r"\bkillall\b",
        feedback=(
            "ERROR: The `killall` command is not allowed.\n"
            "`killall` kills processes by name, which could terminate critical OpenHands processes.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks all killall commands",
    ),
    # === Block ALL pkill commands (pkill always targets by pattern, not PID) ===
    BlacklistEntry(
        pattern=r"\bpkill\b",
        feedback=(
            "ERROR: The `pkill` command is not allowed.\n"
            "`pkill` kills processes by pattern matching, which could terminate critical OpenHands processes.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks all pkill commands",
    ),
    # === Block kill with command substitution $(...) ===
    BlacklistEntry(
        pattern=r"\bkill\s+.*\$\(",
        feedback=(
            "ERROR: The `kill` command with command substitution is not allowed.\n"
            "Using `kill $(...)` could terminate unintended processes.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill with command substitution $()",
    ),
    # === Block kill with backtick command substitution ===
    BlacklistEntry(
        pattern=r"\bkill\s+.*`",
        feedback=(
            "ERROR: The `kill` command with command substitution is not allowed.\n"
            "Using kill with backticks could terminate unintended processes.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill with backtick command substitution",
    ),
    # === Block kill with variables ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*\$\w+",
        feedback=(
            "ERROR: The `kill` command with shell variables is not allowed.\n"
            "Using `kill $var` could terminate unintended processes if the variable contains unexpected values.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill with shell variables",
    ),
    # === Block kill -1 or kill -9 -1 (kills all user processes) ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*-1\b",
        feedback=(
            "ERROR: This command would kill all processes you own.\n"
            "`kill -1` or `kill -9 -1` sends a signal to all your processes, including the action executor.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill -1 which kills all user processes",
    ),
    # === Block kill 0 (kills all processes in the process group) ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*0\b",
        feedback=(
            "ERROR: This command would kill all processes in the current process group.\n"
            "`kill 0` sends a signal to all processes in your process group, including the action executor.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill 0 which kills the process group",
    ),

    # === Block kill with negative PIDs (process groups) ===
    # Negative PIDs like -12345 kill entire process groups
    # We need to catch: kill -12345, kill -9 -12345, etc.
    # But NOT catch: kill -9 12345 (where -9 is signal, 12345 is positive PID)
    BlacklistEntry(
        pattern=r"\bkill\s+(-[1-9]|-1[0-5]|-[A-Z]+|-SIG[A-Z]+)?\s*-([2-9]\d\d+|[1-9]\d\d\d+)\s*$",
        feedback=(
            "ERROR: The `kill` command with negative PIDs (process groups) is not allowed.\n"
            "Using negative PIDs kills entire process groups, which could terminate critical processes.\n\n"
            "SUGGESTION: Use `ps aux | grep <your_process>` to find the specific PID, "
            "then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks kill with negative PIDs (process groups)",
    ),
    # === Dangerous rm commands ===
    # Block rm -rf / or rm -rf /* (root filesystem)
    BlacklistEntry(
        pattern=r"\brm\s+(-\w+\s+)*(/\s*$|/\*)",
        feedback=(
            "ERROR: This command would destroy the entire filesystem.\n"
            "`rm -rf /` or `rm -rf /*` is a catastrophically dangerous command.\n\n"
            "SUGGESTION: Be specific about which directory you want to remove."
        ),
        description="Blocks rm -rf / or rm -rf /*",
    ),
    # Block rm of critical top-level directories (exact match, not subdirs)
    BlacklistEntry(
        pattern=r"\brm\s+(-\w+\s+)*(/(bin|usr|etc|var|home|root|opt|lib|lib64|sbin|boot|dev|proc|sys))\s*$",
        feedback=(
            "ERROR: This command would delete critical system directories.\n"
            "Deleting root-level directories like /bin, /usr, /etc, etc. is blocked.\n\n"
            "SUGGESTION: Be more specific about which files or directories you want to delete. "
            "Use absolute paths to the specific items you want to remove."
        ),
        description="Blocks rm of critical system directories",
    ),
    # === Commands that could affect the tmux session ===
    BlacklistEntry(
        pattern=r"\btmux\s+(kill-server|kill-session\s+-t\s+openhands)",
        feedback=(
            "ERROR: This command would terminate the OpenHands tmux session.\n"
            "Killing the tmux server or the openhands session would break command execution.\n\n"
            "SUGGESTION: If you need to kill a specific process, use `ps aux | grep <your_process>` "
            "to find the specific PID, then use `kill -9 <PID>` to terminate only that process."
        ),
        description="Blocks killing the openhands tmux session",
    ),
    # === Shutdown/reboot commands ===
    BlacklistEntry(
        pattern=r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b",
        feedback=(
            "ERROR: System shutdown/reboot commands are blocked.\n"
            "These commands would terminate the runtime environment.\n\n"
            "This type of command is not allowed in this environment."
        ),
        description="Blocks system shutdown/reboot commands",
    ),
    # === Dangerous dd commands ===
    BlacklistEntry(
        pattern=r"\bdd\s+.*of=\s*(/dev/sd[a-z]|/dev/nvme\w*|/dev/hd[a-z]|/dev/null)\b",
        feedback=(
            "ERROR: This dd command could overwrite disk devices.\n"
            "Writing directly to disk devices is blocked to prevent data loss.\n\n"
            "SUGGESTION: Use standard file operations instead of dd for file manipulation."
        ),
        description="Blocks dd commands that write to disk devices",
    ),
    # ==================================================================
    # === Git network access (remote fetches). =========================
    # ==================================================================
    # The task container's local git state is wiped by run_infer.py
    # (branches, tags, stash, notes, reflog, and dangling objects are all
    # deleted so nothing past `base_commit` is reachable locally). The
    # only remaining leak channel is going out over the network to fetch
    # post-base history from a remote. These rules block every git
    # subcommand or flag that can pull commits/refs from a remote URL.
    # === git fetch / pull / clone / ls-remote ===
    # The negative lookbehind `(?<!["'=\w])` ensures the subcommand word is
    # not preceded by `=`, `"`, `'`, or another word-char — i.e. not inside
    # a flag value like `--grep="fetch"` or `-S "clone"`.
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:fetch|pull|clone|ls-remote)\b",
        feedback=(
            "ERROR: Git network commands are blocked.\n"
            "`git fetch`, `git pull`, `git clone`, and `git ls-remote` reach out "
            "to a remote and can pull commits that post-date this task's base "
            "commit, which would leak the solution.\n\n"
            "SUGGESTION: Work entirely within the local checkout. Read source "
            "files directly instead of fetching remote state."
        ),
        description="Blocks git fetch/pull/clone/ls-remote",
    ),
    # === git remote add / set-url / update / rename / set-branches ===
    BlacklistEntry(
        pattern=r"\bgit\s+remote\s+(?:add|set-url|set-head|update|rename|set-branches)\b",
        feedback=(
            "ERROR: Configuring a git remote is blocked.\n"
            "Adding or re-pointing a remote is the prelude to fetching post-base "
            "commits from the network, which would leak the solution.\n\n"
            "SUGGESTION: Work entirely within the local checkout."
        ),
        description="Blocks git remote add/set-url/set-head/update/rename/set-branches",
    ),
    # === git submodule update/sync/add (all can fetch over the network) ===
    BlacklistEntry(
        pattern=r"\bgit\s+submodule\s+(?:add|update|sync|init)\b",
        feedback=(
            "ERROR: `git submodule` operations that touch the network are blocked.\n"
            "Submodule add/update/sync/init can fetch arbitrary commits from "
            "remote URLs, which would leak post-base-commit state.\n\n"
            "SUGGESTION: Do not initialize or update submodules."
        ),
        description="Blocks git submodule add/update/sync/init",
    ),
    # === git archive --remote=<url> ===
    BlacklistEntry(
        pattern=r"\bgit\s+archive\b[^\n|;&]*\s--remote\b",
        feedback=(
            "ERROR: `git archive --remote=<url>` is blocked.\n"
            "It fetches a tree from a remote, which can leak post-base-commit "
            "state.\n\n"
            "SUGGESTION: Use a local checkout; do not read remote archives."
        ),
        description="Blocks git archive --remote",
    ),
    # === Any git invocation that contains a remote URL (https://, git://,
    # ssh://, or git@host:path). This is a belt-and-suspenders catch for
    # forms like `git <sub> <URL>` that aren't covered above. The negative
    # lookbehind avoids matching URLs that appear inside flag values like
    # `-S "http://example"` or `--grep="https://..."`. ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:https?|git|ssh|ftp|ftps)://",
        feedback=(
            "ERROR: This git command references a remote URL.\n"
            "Any git subcommand that includes an http(s)/git/ssh URL can pull "
            "remote state, leaking post-base-commit commits.\n\n"
            "SUGGESTION: Work within the local checkout only."
        ),
        description="Blocks git commands that reference a remote URL (https/git/ssh/ftp)",
    ),
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])git@[\w\.\-]+:",
        feedback=(
            "ERROR: This git command references a git-over-SSH URL.\n"
            "Any git subcommand with a `git@host:path` URL can pull remote "
            "state, leaking post-base-commit commits.\n\n"
            "SUGGESTION: Work within the local checkout only."
        ),
        description="Blocks git commands that reference a git@host:path URL",
    ),
    # === References to remote-tracking refs (origin/..., upstream/..., remotes/<name>/...).
    # Post-cleanup these refs should not exist; this rule is a belt-and-
    # suspenders block in case any slip through. ===
    BlacklistEntry(
        pattern=r"\bgit\s+\S+\b[^\n|;&]*\b(?:origin|upstream|remotes/[^\s/]+)/[\w./\-]+",
        feedback=(
            "ERROR: This git command references a remote-tracking ref.\n"
            "`origin/…`, `upstream/…`, and `remotes/<name>/…` point to commits "
            "that live on a remote and may post-date the task's base commit.\n\n"
            "SUGGESTION: Work locally against HEAD; do not reference remote refs."
        ),
        description="Blocks git commands that reference origin/upstream/remotes/<name>",
    ),
    # === git show ===
    # `git show` can display the full contents (message + diff) of an
    # arbitrary commit, tag, or object. If any post-base-commit object is
    # still reachable, or a ref/SHA is supplied, this becomes a channel for
    # reading the solution diff directly. Block the subcommand outright and
    # steer the agent toward reading source files in the working tree.
    # The negative lookbehind `(?<!["'=\w])` avoids matching `show` when it
    # appears inside a flag value (e.g. `--grep="show"`).
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])show\b",
        feedback=(
            "ERROR: The `git show` command is not allowed.\n"
            "`git show` can display the message and full diff of an arbitrary "
            "commit or object, which could reveal the task's solution.\n\n"
            "SUGGESTION: Read the relevant source files directly from the "
            "working tree instead of inspecting commit contents."
        ),
        description="Blocks git show",
    ),
    # ==================================================================
    # === Git history mining (reading the task's future/fix commit). ===
    # ==================================================================
    # The July 2026 anti-cheat audit of RL training rollouts found the
    # dominant cheat was LOCAL git archaeology: `git log --all` /
    # `--grep=#<issue>` to surface the fix commit that deep-reset failed
    # to prune, then viewing/materializing it. These rules block every
    # confirmed vector. None of these commands are needed to solve a SWE
    # task from the base commit (plain `git log`, `git diff`, `git
    # checkout <branch>` all remain allowed), and none are used by the
    # harness itself (the deep-reset's `git reflog expire` and
    # `rm -f .git/packed-refs` are explicitly carved out below).
    # === git log / rev-list across hidden refs (--all/--branches/...) ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:log|rev-list|shortlog)\b[^\n|;&]*(?:--(?:all|branches|remotes|walk-reflogs)\b|\s-g\b)",
        feedback=(
            "ERROR: Walking git history across all refs is not allowed.\n"
            "`git log/rev-list --all/--branches/--remotes` can surface commits "
            "that post-date this task's base commit, which would leak the "
            "solution.\n\n"
            "SUGGESTION: Use plain `git log` (ancestry of HEAD) if you need "
            "history, and read source files in the working tree."
        ),
        description="Blocks git log/rev-list/shortlog with --all/--branches/--remotes/reflog-walk",
    ),
    # === git log --grep (searching history for the issue number) ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:log|rev-list)\b[^\n|;&]*--grep",
        feedback=(
            "ERROR: Searching git history by commit message is not allowed.\n"
            "`git log --grep` is used to locate the commit that fixes this "
            "task's issue, which would leak the solution.\n\n"
            "SUGGESTION: Solve the issue from the source code in the working "
            "tree; do not search commit history for it."
        ),
        description="Blocks git log/rev-list --grep",
    ),
    # === git reflog (reading). The harness's own cleanup runs
    # `git reflog expire ...`, hence the negative lookahead. ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])reflog\b(?!\s+(?:expire|delete)\b)",
        feedback=(
            "ERROR: Reading the git reflog is not allowed.\n"
            "The reflog can expose pre-reset HEAD positions that post-date the "
            "task's base commit, which would leak the solution.\n\n"
            "SUGGESTION: Work from the current checkout state only."
        ),
        description="Blocks git reflog reads (expire/delete still allowed for cleanup)",
    ),
    # === Raw object archaeology: cat-file / fsck / pack tools / cherry(-pick) ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:cat-file|fsck|verify-pack|unpack-objects|cherry(?:-pick)?)\b",
        feedback=(
            "ERROR: This git object-inspection command is not allowed.\n"
            "`git cat-file`, `git fsck`, pack inspection, and `git cherry(-pick)` "
            "can read or apply commits that post-date the task's base commit, "
            "which would leak the solution.\n\n"
            "SUGGESTION: Read and edit the source files in the working tree "
            "directly."
        ),
        description="Blocks git cat-file/fsck/verify-pack/unpack-objects/cherry/cherry-pick",
    ),
    # === Probing whether a commit is in the future (merge-base / --contains) ===
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:merge-base\b[^\n|;&]*--is-ancestor|(?:branch|tag|for-each-ref)\b[^\n|;&]*--contains\b)",
        feedback=(
            "ERROR: Probing commit ancestry is not allowed.\n"
            "`git merge-base --is-ancestor` and `--contains` are used to test "
            "whether a commit post-dates the task's base commit before reading "
            "it — a step in looking up the solution.\n\n"
            "SUGGESTION: You do not need commit-ancestry information to solve "
            "this task."
        ),
        description="Blocks merge-base --is-ancestor and branch/tag/for-each-ref --contains",
    ),
    # === Materializing a commit by raw hash (checkout/switch/reset/restore).
    # The harness's own reset uses `git checkout "$BASE"` (a shell variable,
    # not a raw hex literal), so it does not match. ===
    # The hex token is guarded on both sides so hex-looking FILE names
    # (e.g. `git checkout -- src/abcdef12.js`) do not match.
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:checkout|switch|reset|restore)\b[^\n|;&]*(?<![\w/.\-])[0-9a-f]{7,40}(?![\w/.\-])",
        feedback=(
            "ERROR: Checking out / resetting to a raw commit hash is not allowed.\n"
            "Materializing an arbitrary commit can bring the task's future fix "
            "into the working tree, which would leak the solution.\n\n"
            "SUGGESTION: Stay on the current branch. Use `git checkout <branch>` "
            "or `git reset --hard HEAD` if you need to discard changes."
        ),
        description="Blocks git checkout/switch/reset/restore targeting a raw commit hash",
    ),
    # === git diff between two commits (range/caret/two hashes) ===
    # Hex tokens are path-guarded; single-hash `git diff <base>` (reviewing
    # your own changes against a known commit) stays allowed.
    BlacklistEntry(
        pattern=r"\bgit\b[^\n|;&]*?(?<![\"'=\w])diff\b[^\n|;&]*(?<![\w/.\-])[0-9a-f]{7,40}(?:\.\.\.?|\^|[^\n|;&]*?\s(?<![\w/.\-])[0-9a-f]{7,40}(?![\w/.\-]))",
        feedback=(
            "ERROR: Diffing between two raw commits is not allowed.\n"
            "`git diff <hash>..<hash>` / `<hash>^` can display the full patch of "
            "an arbitrary commit, which could reveal the task's solution.\n\n"
            "SUGGESTION: Use plain `git diff` (or `git diff HEAD`) to review "
            "your own changes."
        ),
        description="Blocks git diff between two raw commit hashes (range/caret forms)",
    ),
    # === Reading git internals directly (reader commands only — the
    # harness's cleanup `rm -f .git/packed-refs` etc. must stay allowed). ===
    BlacklistEntry(
        pattern=r"\b(?:cat|less|more|head|tail|grep|egrep|fgrep|rg|find|ls|xxd|od|strings|sed|awk|cut|sort|uniq|tr|nl|tac|rev|tee|cp|dd|python[\d.]*|perl|ruby|node|vi|vim|nano|emacs)\b[^\n|;&]*\.git/(?:logs\b|packed-refs\b|ORIG_HEAD\b|FETCH_HEAD\b|refs/)",
        feedback=(
            "ERROR: Reading git internals directly is not allowed.\n"
            "Files under `.git/` (logs, packed-refs, ORIG_HEAD, refs) can expose "
            "commits that post-date the task's base commit, which would leak the "
            "solution.\n\n"
            "SUGGESTION: Use regular git commands against HEAD and read source "
            "files in the working tree."
        ),
        description="Blocks reading .git/logs, .git/packed-refs, .git/ORIG_HEAD, .git/FETCH_HEAD, .git/refs",
    ),
    # ==================================================================
    # === Online solution lookups via curl / wget. =====================
    # ==================================================================
    # Block fetching anything from github.com (including subdomains like
    # api.github.com, gist.github.com, codeload.github.com), the static-content
    # host githubusercontent.com (raw.*, gist.*, objects.*), and github.io
    # (Pages). The agent could otherwise grab the upstream repo's commit
    # history, PRs, issue threads, or the maintainer's gists to look up the
    # task's solution. The `[^\n|;&]*` body anchors the URL to the same
    # command segment as `curl`/`wget` so cross-pipe usage like
    # `cat README | grep github.com` is not blocked.
    BlacklistEntry(
        pattern=r"\b(?:curl|wget)\b[^\n|;&]*\b(?:github\.com|githubusercontent\.com|github\.io)\b",
        feedback=(
            "ERROR: Reaching out to GitHub is not allowed.\n"
            "You are not allowed to look up the solution online. Fetching from "
            "github.com / githubusercontent.com / github.io (commit history, "
            "PRs, issues, gists, raw files) is restricted"
            "evaluation.\n\n"
            "SUGGESTION: Solve the problem using only the local checkout."
        ),
        description="Blocks curl/wget to github.com and related GitHub domains",
    ),
]


@dataclass
class BlacklistCheckResult:
    """Result of checking a command against the blacklist."""

    is_blocked: bool
    matched_entry: Optional[BlacklistEntry] = None
    feedback: str = ""


def check_command_blacklist(command: str) -> BlacklistCheckResult:
    """Check if a command matches any blacklisted pattern.

    Args:
        command: The bash command to check.

    Returns:
        BlacklistCheckResult with is_blocked=True and feedback if blocked,
        or is_blocked=False if the command is allowed.
    """
    if not BLACKLIST_ENABLED:
        logger.debug("Command blacklist is disabled via environment variable")
        return BlacklistCheckResult(is_blocked=False)

    # Normalize the command for matching
    normalized_command = command.strip()

    for entry in COMMAND_BLACKLIST:
        if not entry.enabled:
            continue

        try:
            if re.search(entry.pattern, normalized_command, re.IGNORECASE):
                logger.warning(
                    f"Command blocked by blacklist: {normalized_command!r}\n"
                    f"Matched pattern: {entry.pattern}\n"
                    f"Description: {entry.description}"
                )
                return BlacklistCheckResult(
                    is_blocked=True,
                    matched_entry=entry,
                    feedback=entry.feedback,
                )
        except re.error as e:
            logger.error(f"Invalid regex pattern in blacklist: {entry.pattern}: {e}")
            continue

    return BlacklistCheckResult(is_blocked=False)
