; Windows installer for shot-clipper.
; Built by installer\build-installer.ps1, which stages a clean `git archive`
; snapshot of the repo into installer\stage before invoking ISCC on this file.
; See installer\setup-deps.ps1 for what actually happens post-install.

#ifndef StageDir
  #define StageDir "stage"
#endif

#define AppVersion "0.1.0"

[Setup]
AppId={{8F1C7B2E-6B3A-4E2D-9C1A-2B7E5D4A9F31}
AppName=shot-clipper
AppVersion={#AppVersion}
AppPublisher=TAN PENG
DefaultDirName={localappdata}\shot-clipper
DefaultGroupName=shot-clipper
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=shot-clipper-setup
SetupIconFile={#StageDir}\installer\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\installer\icon.ico

[Files]
; Everything git-tracks (source, docs, installer scripts) - .gitignore already
; keeps .venv/, clips/, models/*.pt, data/jobs/, etc. out of the archive this
; stage directory was built from, so nothing extra needs excluding here.
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\shot-clipper"; Filename: "{app}\installer\launch-label-ui.bat"; IconFilename: "{app}\installer\icon.ico"; WorkingDir: "{app}"
Name: "{group}\Uninstall shot-clipper"; Filename: "{uninstallexe}"
Name: "{userdesktop}\shot-clipper"; Filename: "{app}\installer\launch-label-ui.bat"; IconFilename: "{app}\installer\icon.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup-deps.ps1"""; WorkingDir: "{app}"; StatusMsg: "Installing Python, ffmpeg, and model weights - this can take several minutes..."; Flags: waituntilterminated skipifsilent
Filename: "{app}\installer\launch-label-ui.bat"; Description: "Launch shot-clipper now"; Flags: postinstall skipifsilent nowait

[UninstallDelete]
; Deliberately not deleting {app}\data or {app}\clips here - those hold the
; user's hoop calibrations, labels, and cut clips. Inno's default uninstall
; only removes files it installed (tracked in the uninstall log), so files
; created later under data/ and clips/ (labels.json, new clip folders, etc.)
; are left behind automatically; this section exists just to document that
; as intentional rather than an oversight.

[Messages]
FinishedLabel=Setup has finished installing shot-clipper.%n%nYour clips and labels live under:%n{app}\data%n{app}\clips%n%nThese are kept if you ever uninstall - remove them yourself if you want a clean slate.
