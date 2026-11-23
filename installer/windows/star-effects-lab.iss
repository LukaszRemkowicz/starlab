; Built by installer\build-windows.ps1. Pass /DAppVersion= /DStageDir= /DPluginFile= /DOutDir= /DIconFile=
#define AppName "Star Effects Lab"
[Setup]
AppId={{6D0E5B4C-3A55-4B7E-9C58-5A2F7E1B9D01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Star Effects Lab
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableDirPage=yes
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppName}.exe
Compression=lzma2/ultra
SolidCompression=yes
OutputDir={#OutDir}
OutputBaseFilename={#AppName} {#AppVersion} Setup
WizardStyle=modern

[Components]
Name: "app"; Description: "{#AppName}"; Types: full custom; Flags: fixed
Name: "plugin"; Description: "Photoshop filter"; Types: full custom

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: app
Source: "{#PluginFile}"; DestDir: "{commoncf64}\Adobe\Plug-Ins\CC\{#AppName}"; Flags: ignoreversion; Components: plugin

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppName}.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppName}.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
