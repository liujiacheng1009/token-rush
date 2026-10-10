const vscode = require("vscode");
const path = require("path");
const http = require("http");
const https = require("https");

const ORIGIN = process.env.TOKENRUSH_ORIGIN || "http://127.0.0.1:8000";
const lib = ORIGIN.startsWith("https:") ? https : http;
const agent = ORIGIN.startsWith("https:") ? new https.Agent({ rejectUnauthorized: false }) : undefined;

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("tokenrush.addSelection", () => addSelection()),
    vscode.commands.registerCommand("tokenrush.addTerminal", () => addTerminal())
  );
  setInterval(pull, 700);
  setTimeout(() => {
    vscode.commands.executeCommand("workbench.action.closeAuxiliaryBar");
  }, 800);
}

function citeOf(editor) {
  if (!editor || editor.selection.isEmpty) return null;
  const sel = editor.selection;
  let start = sel.start.line + 1;
  let end = sel.end.line + 1;
  if (sel.end.character === 0 && end > start) end -= 1;
  const text = editor.document.getText(sel).replace(/\s+$/, "");
  if (!text) return null;
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  const file = editor.document.uri.fsPath;
  const rel = folder ? path.relative(folder.uri.fsPath, file) : path.basename(file);
  const label = start === end ? rel + ":" + start : rel + ":" + start + "-" + end;
  return { label, text: text.slice(0, 12000) };
}

function postCite(cite) {
  const data = Buffer.from(JSON.stringify(cite));
  const req = lib.request(ORIGIN + "/ide/cite", {
    method: "POST",
    agent,
    headers: { "Content-Type": "application/json", "Content-Length": data.length },
  }, (res) => {
    res.resume();
    if (res.statusCode !== 200) {
      vscode.window.showErrorMessage("没能加入对话：" + res.statusCode);
    }
  });
  req.on("error", (err) => vscode.window.showErrorMessage("没能加入对话：" + err.message));
  req.write(data);
  req.end();
}

function addSelection() {
  const cite = citeOf(vscode.window.activeTextEditor);
  if (!cite) return;
  postCite(cite);
}

async function addTerminal() {
  const before = await vscode.env.clipboard.readText();
  await vscode.commands.executeCommand("workbench.action.terminal.copySelection");
  await new Promise((resolve) => setTimeout(resolve, 60));
  const text = (await vscode.env.clipboard.readText()).replace(/\s+$/, "");
  if (!text || text === before) return;
  const name = (vscode.window.activeTerminal && vscode.window.activeTerminal.name) || "shell";
  postCite({ label: "shell: " + name, text: text.slice(0, 12000) });
  await vscode.env.clipboard.writeText(before);
}

function pull() {
  const req = lib.get(ORIGIN + "/ide/pull", { agent }, (res) => {
    const chunks = [];
    res.on("data", (c) => chunks.push(c));
    res.on("end", () => {
      try {
        if (JSON.parse(Buffer.concat(chunks).toString()).pull) addSelection();
      } catch (e) {}
    });
  });
  req.on("error", () => {});
  req.end();
}

module.exports = { activate };
