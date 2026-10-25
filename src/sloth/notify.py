"""Desktop notifications with no extra dependencies: PowerShell toast on Windows, osascript on macOS,
notify-send on Linux, and print() everywhere else."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

# Windows only shows toasts from registered apps; Windows PowerShell's own ID always exists.
POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def windows_command(title: str, body: str) -> list[str]:
    xml = (
        '<toast><visual><binding template="ToastGeneric">'
        f"<text>{escape(title)}</text><text>{escape(body)}</text>"
        "</binding></visual></toast>"
    )
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$x = New-Object Windows.Data.Xml.Dom.XmlDocument
$x.LoadXml('{xml.replace("'", "''")}')
$t = [Windows.UI.Notifications.ToastNotification]::new($x)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{POWERSHELL_APP_ID}').Show($t)
"""
    # -EncodedCommand sidesteps every quoting problem with titles containing quotes or $.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


def command(title: str, body: str, platform: str = sys.platform) -> list[str] | None:
    if platform == "win32":
        return windows_command(title, body)
    if platform == "darwin":
        t, b = title.replace('"', "'"), body.replace('"', "'")
        return ["osascript", "-e", f'display notification "{b}" with title "{t}"']
    if shutil.which("notify-send"):
        return ["notify-send", title, body]
    return None


def toast(title: str, body: str) -> bool:
    """Show a notification; returns False (and prints) if no notifier works."""
    cmd = command(title, body)
    if cmd:
        try:
            subprocess.run(cmd, check=True, timeout=20, capture_output=True)
            return True
        except (OSError, subprocess.SubprocessError):
            pass
    print(f"{title}: {body}")  # last-resort fallback: the terminal
    return False
