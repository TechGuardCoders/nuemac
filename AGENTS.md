# Installing Nuemac: instructions for an AI agent

You are setting up Nuemac, a local OpenAI-compatible LLM server, on the user's Mac. Follow these steps in order.
Every command is non-interactive except the one `sudo` line in step 3, which a person may have to run.

## 1. Check the Mac

```bash
uname -m; sw_vers -productVersion; sysctl -n hw.memsize; command -v brew || ls /opt/homebrew/bin/brew
```

Needs: `arm64`, macOS 14 or newer, 16 GB or more (`hw.memsize` >= 17179869184 is 16 GB), Homebrew, ~15 GB free
disk. If Homebrew is missing, stop and ask the user to install it from https://brew.sh (it needs their password).

## 2. Install

```bash
curl -fsSL https://raw.githubusercontent.com/TechGuardCoders/nuemac/main/install.sh | bash -s -- --yes
```

This takes 5-30 minutes (an ~12 GB download, resumed if interrupted; every file is checked against its SHA-256).
It picks the model for the Mac's memory: `inkling-2-mini-tg` (96K context) on 16 GB, `inkling-2-mini-oq3e`
(256K) on 24 GB or more. Other options: `--port N` (default 8080), `--model NAME`, `--no-start`, `--dry-run`.

## 3. GPU memory limit (may need a person)

macOS lets the GPU use only ~2/3 of memory by default; the model needs more. The installer raises the limit with
`sudo`. If it printed `NEEDS A PERSON`, ask the user to run, in their own terminal:

```bash
sudo sysctl iogpu.wired_limit_mb=14500
```

(the installer prints the exact number for this Mac: 14500 on 16 GB, 19300 on 24 GB), then run
`~/.nuemac/bin/nuemac restart`. The setting resets at reboot unless the user also accepted the boot job; to make it
permanent they can re-run the installer in a terminal.

## 4. Verify

```bash
~/.nuemac/bin/nuemac status        # "server up" and the engine version
~/.nuemac/bin/nuemac qa            # must print "quality check: 12/12" (11/12 is acceptable; less is a problem)
curl -s http://127.0.0.1:8080/v1/models
```

If `qa` fails or the server is not up, read `~/.nuemac/bin/nuemac logs` and report what it says. The most common
cause is the GPU limit from step 3, or another large app (a Docker VM, another model server) holding memory.

## 5. Hand over

Tell the user:
- Endpoint `http://127.0.0.1:8080/v1` (OpenAI-compatible, any API key), model id `inkling-2-mini`. Loopback only:
  nothing leaves the Mac.
- It supports streaming, tool calls, `reasoning_effort` (`low` / `medium` / `high`), up to 4 requests at once, and
  images as `image_url` parts (base64 `data:` URLs or http(s) links).
- The server starts at login. `nuemac stop` / `start` / `status` / `bench` manage it; `nuemac uninstall` removes the
  login service (files stay in `~/.nuemac`).

## Updating an existing install

Re-run the step 2 command; it keeps `~/.nuemac/settings`. To switch a 16 GB Mac from the older `inkling-2-mini`
build to `inkling-2-mini-tg`: `~/.nuemac/bin/nuemac use inkling-2-mini-tg` (downloads it, then restarts).
