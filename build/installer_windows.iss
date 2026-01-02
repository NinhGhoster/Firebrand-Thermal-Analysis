; Inno Setup script for Windows installer
; Requires Inno Setup (iscc) installed on Windows.

#define MyAppName "FirebrandThermalAnalysis"
#define MyAppVersion "0.0.2"
#define MyAppPublisher "H. Nguyen"
#define MyAppExeName "FirebrandThermalAnalysis.exe"

[Setup]
AppId={{B1E3E4D1-6F66-4F8E-9D9E-0B2F7B19F1B2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename={#MyAppName}_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:";
