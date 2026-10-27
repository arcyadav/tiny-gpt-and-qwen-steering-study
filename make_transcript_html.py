"""Build a standalone, clickable HTML transcript of the Claude Code session for the viva.

    python make_transcript_html.py SESSION.jsonl transcript.html [MAX_TURNS]

Reads Claude Code's own session log (the full session; `/export` only includes what is left after
the conversation was summarised). Every user message is a visible "Turn N"; the response under it
is one collapsed block with the reply text and each tool call (input and output, shortened) as a
smaller collapsible item. Internal reasoning is not included, as in /export.
"""
import datetime
import html
import json
import re
import sys

import markdown

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
MAX_IN, MAX_OUT = 1500, 2000

REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
PASTE = re.compile(r'<pasted_content id="[^"]*">(.*?)</pasted_content id="[^"]*">', re.S)


def clip(s, n):
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n:,} more characters not shown]"


def when(ts):
    t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST)
    return t.strftime("%a %d %b %Y, %H:%M IST")


def text_of(content):
    """(text, number of images) of a message's content."""
    if isinstance(content, str):
        return content, 0
    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
    images = sum(b.get("type") == "image" for b in content)
    return "\n".join(texts), images


def result_text(content):
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[image]")
    return "\n".join(parts)


def user_html(text, images):
    """User message: plain text with line breaks; pasted content in its own box."""
    out, pos = [], 0
    for m in PASTE.finditer(text):
        out.append(html.escape(text[pos:m.start()].strip()))
        out.append(f'<div class="paste"><div class="paste-h">pasted</div><pre>{html.escape(m.group(1).strip())}</pre></div>')
        pos = m.end()
    out.append(html.escape(text[pos:].strip()))
    body = "".join(f"<p>{x}</p>" if x and not x.startswith('<div class="paste">') else x for x in out if x)
    body = body.replace("\n", "<br>")
    if images:
        body += f'<p class="note">[{images} image{"s" if images > 1 else ""} attached]</p>'
    return body


LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def fix_lists(text):
    """Python-Markdown needs a blank line before a list and 4-space indents for nested items;
    the replies use GitHub style (no blank line, 2-space indents)."""
    out, prev, in_code = [], "", False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else LIST_ITEM.match(line)
        if m:
            line = " " * (len(m.group(1)) * 2) + line.lstrip()
            if prev.strip() and not LIST_ITEM.match(prev):
                out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def tool_label(name, inp):
    for key in ("description", "file_path", "path", "pattern", "query", "url", "prompt", "skill", "action"):
        if inp.get(key):
            return str(inp[key]).split("\n")[0][:120]
    if inp.get("command"):
        return inp["command"].split("\n")[0][:120]
    return ""


def tool_input(name, inp):
    if name in ("Bash", "PowerShell") and "command" in inp:
        return inp["command"]
    if name == "Edit":
        return f"{inp.get('file_path', '')}\n--- replace ---\n{inp.get('old_string', '')}\n--- with ---\n{inp.get('new_string', '')}"
    if name == "Write":
        return f"{inp.get('file_path', '')}\n--- content ---\n{inp.get('content', '')}"
    return json.dumps(inp, ensure_ascii=False, indent=1)


def note_of(text):
    """Non-turn user records (background task finished, slash command, interruption) -> short note."""
    if text.startswith("<task-notification>"):
        s = re.search(r"<summary>(.*?)</summary>", text, re.S)
        st = re.search(r"<status>(.*?)</status>", text, re.S)
        return "Background task " + (st.group(1).strip() if st else "update") + (f": {s.group(1).strip()}" if s else "")
    if "<command-name>" in text:
        n = re.search(r"<command-name>(.*?)</command-name>", text, re.S)
        a = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
        return f"The user ran {n.group(1).strip()} {a.group(1).strip() if a else ''}".strip()
    if "<local-command-stdout>" in text:
        o = re.search(r"<local-command-stdout>(.*?)</local-command-stdout>", text, re.S)
        return "Command output: " + (o.group(1).strip() if o else "")
    if "<local-command-caveat>" in text:
        return None
    if "<bash-input>" in text:  # a shell command the user ran with "!"
        i = re.search(r"<bash-input>(.*?)</bash-input>", text, re.S)
        return f"The user ran in the shell: {i.group(1).strip()}"
    if "<bash-stdout>" in text or "<bash-stderr>" in text:
        o = re.search(r"<bash-stdout>(.*?)</bash-stdout>", text, re.S)
        return "Shell output: " + clip(o.group(1).strip() if o else "", 300)
    if text.startswith("[Request interrupted"):
        return "The user interrupted the response."
    return False  # a real user message


def load(path):
    turns, seen, results, tool_blocks = [], set(), {}, []
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    for d in records:  # tool results first, so each call can show its output
        if d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), list):
            for b in d["message"]["content"]:
                if b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = (result_text(b.get("content")), b.get("is_error"))
    for d in records:
        u = d.get("uuid")
        if d.get("isSidechain") or d.get("type") not in ("user", "assistant") or (u and u in seen):
            continue
        seen.add(u)
        content = d["message"]["content"]
        if d["type"] == "user":
            if d.get("isMeta"):
                continue
            if d.get("isCompactSummary"):
                if turns:
                    turns[-1]["items"].append(("note", "The conversation was long, so Claude Code summarised the earlier part here and continued."))
                continue
            text, images = text_of(content)
            text = REMINDER.sub("", text).strip()
            if not text and not images:
                continue  # tool results only
            note = note_of(text)
            if note is None:
                continue
            if note:
                if turns:
                    turns[-1]["items"].append(("note", note))
                continue
            turns.append({"time": d.get("timestamp"), "user": user_html(text, images), "items": []})
        elif turns:
            for b in content:
                if b.get("type") == "text" and b["text"].strip():
                    turns[-1]["items"].append(("text", b["text"]))
                elif b.get("type") == "tool_use":
                    out, err = results.get(b["id"], ("", False))
                    turns[-1]["items"].append(("tool", b["name"], b.get("input", {}), out, err))
    return turns


def render(turns):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    first, last = when(turns[0]["time"]), when(turns[-1]["time"])
    toc, body = [], []
    for i, t in enumerate(turns, 1):
        plain = re.sub(r"<[^>]+>", " ", t["user"])
        toc.append(f'<li><a href="#turn-{i}"><b>Turn {i}</b> {html.escape(clip(html.unescape(" ".join(plain.split())), 90).split(chr(10))[0])}</a></li>')
        parts, n_tools = [], 0
        for item in t["items"]:
            if item[0] == "text":
                md.reset()
                parts.append(f'<div class="reply">{md.convert(fix_lists(item[1]))}</div>')
            elif item[0] == "note":
                parts.append(f'<div class="note">{html.escape(item[1])}</div>')
            else:
                _, name, inp, out, err = item
                n_tools += 1
                label = tool_label(name, inp)
                parts.append(
                    f'<details class="tool{" err" if err else ""}"><summary><span class="tn">{html.escape(name)}</span> '
                    f'{html.escape(label)}</summary><div class="io"><div class="io-h">input</div>'
                    f'<pre>{html.escape(clip(tool_input(name, inp), MAX_IN))}</pre>'
                    + (f'<div class="io-h">output</div><pre>{html.escape(clip(out, MAX_OUT))}</pre>' if out else "")
                    + "</div></details>")
        summary = f"Claude's response" + (f" · {n_tools} tool call{'s' if n_tools != 1 else ''}" if n_tools else "")
        body.append(
            f'<section class="turn" id="turn-{i}"><div class="th"><span class="tl">Turn {i}</span>'
            f'<span class="ts">{when(t["time"])}</span></div><div class="user">{t["user"]}</div>'
            f'<details class="resp"><summary>{summary}</summary><div class="rb">{"".join(parts) or "<p class=note>(no reply text)</p>"}</div></details></section>')
    return PAGE.format(n=len(turns), first=first, last=last, toc="".join(toc), body="".join(body))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project Session Transcript</title>
<style>
:root {{ --bg:#fbfaf7; --surface:#fff; --s2:#f3f1ec; --text:#1d1d1f; --muted:#66625b; --border:#e4e0d8; --accent:#c2410c; --user:#fff7ed; --code:#f6f4ef; --err:#b91c1c; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141413; --surface:#1d1d1b; --s2:#262623; --text:#ecebe6; --muted:#a29f97; --border:#34332f; --accent:#f08a4b; --user:#2a2118; --code:#232320; --err:#f87171; }} }}
:root[data-theme="dark"] {{ --bg:#141413; --surface:#1d1d1b; --s2:#262623; --text:#ecebe6; --muted:#a29f97; --border:#34332f; --accent:#f08a4b; --user:#2a2118; --code:#232320; --err:#f87171; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 0 16px 60px; }}
header {{ padding: 36px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 18px; }}
h1 {{ margin: 0 0 6px; font-size: clamp(1.6rem, 4vw, 2.2rem); }}
.meta {{ color: var(--muted); }}
.bar {{ position: sticky; top: 0; z-index: 5; background: var(--bg); padding: 10px 0; display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 1px solid var(--border); }}
button {{ font: inherit; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; }}
.toc {{ margin: 16px 0; }}
.toc summary {{ cursor: pointer; font-weight: 600; }}
.toc ol {{ list-style: none; padding: 0; margin: 10px 0 0; columns: 2 320px; font-size: .9rem; }}
.toc li {{ break-inside: avoid; margin-bottom: 4px; }}
.toc a {{ color: var(--text); text-decoration: none; }}
.toc a b {{ color: var(--accent); }}
.turn {{ margin: 22px 0; scroll-margin-top: 60px; }}
.th {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }}
.tl {{ font-weight: 700; color: var(--accent); }}
.ts {{ color: var(--muted); font-size: .85rem; }}
.user {{ background: var(--user); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; overflow-wrap: anywhere; }}
.user p {{ margin: 0 0 6px; }} .user p:last-child {{ margin: 0; }}
.paste {{ border: 1px dashed var(--border); border-radius: 8px; margin: 8px 0; background: var(--surface); }}
.paste-h, .io-h {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); padding: 6px 10px 0; }}
.paste pre {{ max-height: 280px; overflow: auto; }}
details.resp {{ margin-top: 8px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }}
details.resp > summary {{ cursor: pointer; padding: 10px 14px; font-weight: 600; }}
.rb {{ padding: 0 16px 12px; border-top: 1px solid var(--border); }}
.reply {{ overflow-wrap: anywhere; }}
.reply table {{ border-collapse: collapse; display: block; overflow-x: auto; margin: 10px 0; }}
.reply th, .reply td {{ border: 1px solid var(--border); padding: 5px 9px; text-align: left; vertical-align: top; }}
code {{ background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: .9em; }}
pre {{ background: var(--code); padding: 10px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-size: .82rem; margin: 6px 0; }}
pre code {{ background: none; padding: 0; }}
blockquote {{ border-left: 3px solid var(--accent); margin: 8px 0; padding: 2px 12px; color: var(--muted); }}
details.tool {{ margin: 6px 0; border: 1px solid var(--border); border-radius: 8px; background: var(--s2); font-size: .88rem; }}
details.tool > summary {{ cursor: pointer; padding: 6px 10px; color: var(--muted); overflow-wrap: anywhere; }}
details.tool.err > summary {{ color: var(--err); }}
.tn {{ font-family: ui-monospace, Consolas, monospace; font-weight: 600; color: var(--text); }}
.io {{ padding: 0 8px 6px; }}
.io pre {{ max-height: 360px; overflow: auto; }}
.note {{ color: var(--muted); font-style: italic; font-size: .86rem; margin: 6px 0; }}
</style></head><body><div class="wrap">
<header><h1>Steering Transformers: session transcript</h1>
<div class="meta">UMC301 Project 4 · Archana Yadav · one Claude Code session · {n} turns · {first} – {last}</div>
<p class="meta">Each user message is a turn. Click a turn's response to open it; inside, each tool call (command, file edit, search) opens on its own. Long tool inputs and outputs are shortened.</p></header>
<div class="bar"><button onclick="setAll(true)">Expand all responses</button><button onclick="setAll(false)">Collapse all</button><button onclick="window.scrollTo(0,0)">Top</button></div>
<details class="toc"><summary>All {n} turns</summary><ol>{toc}</ol></details>
{body}
</div>
<script>
function setAll(open) {{ document.querySelectorAll("details.resp").forEach(d => d.open = open); }}
</script>
</body></html>"""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    turns = load(src)
    if len(sys.argv) > 3:  # optional: keep only the first N turns (up to the /export)
        turns = turns[:int(sys.argv[3])]
    page = re.sub(r"[\w.+-]+@gmail\.com", "[email]", render(turns))  # no personal addresses in the page
    with open(dst, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"{len(turns)} turns -> {dst}")
