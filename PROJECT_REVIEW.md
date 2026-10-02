# Project review — 2026-10-02

Reviewed the application entry point, Qt event dispatch and lifecycle, settings
persistence, tap and voice detection, cloud verification, OS action execution,
calibration/validation tools, Windows speech helper source, tests, and packaging.
Existing uncommitted work was retained.

## Fixes implemented

| Area | Problem and correction |
| --- | --- |
| Pause and mode changes | Queued tap/voice events and waiting OS actions could execute after pausing. Events now carry a session generation; dispatch checks pause, mode, calibration, shutdown, and voice enablement. Tap pause cancels pending gestures and rejects new audio until resumed. |
| Shutdown | Cleanup was bound to the original voice backend even after selecting a replacement. Cleanup now uses the window's current backend, removes the configuration listener, and cancels queued actions. |
| Calibration | A failed process launch could leave monitoring stopped; window activation could restart capture during calibration. Launch errors now restore the prior monitoring state, and calibration suppresses automatic restarts and dispatch. |
| Settings persistence | A dirty settings reload could treat adopted external values as local edits and overwrite newer changes. The disk baseline is now updated after merging. An interprocess lock protects the complete read/merge/replace operation. |
| Settings validation | Infinite integer/device values raised `OverflowError`. They now use defaults. Deferred updates apply the complete sanitized configuration so cross-setting constraints hold immediately. |
| Cloud decisions | Non-finite or out-of-range intent/risk values could pass policy checks. Invalid values now fail response validation. |
| Linux actions | Failed key injection and browser launch were reported as successful and played feedback. Failure now propagates to the caller. |
| Microphone changes | Queued speech was resampled using the newly selected microphone's rate. Each utterance now retains its capture rate. Old streams are closed even if stopping them fails. |
| Installation | Setuptools failed with multiple top-level packages. Explicit discovery/module lists, a console entry point, and bundled resource lookup now support wheel installation. Dependency bounds are consistent across the two manifests. |
| Windows startup/build | Redirected output could crash on the Persian app name. Console output now uses UTF-8. Nested batch checks now read the current command's exit status. |
| Code checks | Corrected the lint violations found during the baseline review. |

## Verification

- Baseline: 122 tests passed; Ruff reported 16 violations; wheel metadata generation failed.
- After fixes: 137 tests and 22 subtests passed, including mocked Qt GUI tests,
  a real two-process settings-save test, and a redirected CLI subprocess test.
- Ruff passed; configured mypy checks passed for the application modules.
  Most existing methods are untyped, so this is not comprehensive static verification.
- Built a wheel, installed it into a separate temporary virtual environment,
  and ran its installed command from outside the source directory. Existing local
  dependencies were supplied through `PYTHONPATH`; a fresh dependency download
  was not tested. Both icons and the Windows helper were present in that installation.
- Rendered the Qt window offscreen with mocked engines for a UI smoke check.
- Git whitespace checks passed.
- Added a Windows CI workflow for Python 3.11–3.13 covering lint, tests, type
  checks, wheel building, and launcher help. Local checks used Python 3.12;
  the hosted CI matrix has not been run yet.

## Further improvements and verification limits

1. Move slow engine shutdown/model-loading waits out of the Qt thread. Whisper
   currently joins workers synchronously, so a slow model load or in-flight
   transcription/cloud callback can delay a mode change or quit. Preserve the
   current prevention of overlapping recognition sessions when changing this.
2. Make voice calibration backend-specific. Its current Windows Speech confidence
   measurements are not equivalent to Whisper's scores. Provide an in-app
   calibration workflow for the standalone EXE.
3. Split the large GUI module into settings panels and an engine lifecycle
   controller; keep signal dispatch and action ownership explicit.
4. Add recorded audio fixtures spanning microphones, typing, speech, and desk
   impacts. Unit tests cannot establish real-world false-trigger rates.
5. Verify the standalone PyInstaller build on a clean Windows machine, including
   model download, device removal, suspend/resume, tray behavior, and actual
   shortcuts. The standalone EXE and C# helper were not rebuilt in this review.

No real microphone recording, paid cloud requests, desktop shortcut injection,
autostart changes, or publication were needed for the automated checks. Linux
actions were verified with mocks on Windows, not on a live Linux desktop.
