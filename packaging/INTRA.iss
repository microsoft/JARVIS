#define MyAppName "INTRA"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "INTRA"
#define MyAppExeName "INTRA.exe"

[Setup]
AppId={{B1D4FDE4-2F1A-4CC2-AE8B-7CEB0DB5D6A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\INTRA
DefaultGroupName=INTRA
OutputDir=..\dist
OutputBaseFilename=INTRA-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Files]
Source: "..\dist\INTRA\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\INTRA"; Filename: "{app}\INTRA.exe"; Parameters: "--desktop --bridge"
Name: "{commondesktop}\INTRA"; Filename: "{app}\INTRA.exe"; Parameters: "--desktop --bridge"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\INTRA.exe"; Parameters: "--desktop --bridge"; Description: "Launch INTRA"; Flags: nowait postinstall skipifsilent
