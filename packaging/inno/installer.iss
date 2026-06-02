#define AppName "NeuronSpectrum GUI"
#define AppVersion "1.0.0"
#define AppPublisher "NeuronSpectrum"
#define AppExeName "NeuronSpectrumGUI.exe"
#define SourceDir "..\..\dist\NeuronSpectrumGUI"
#define OutputDir "..\..\installer_dist"

[Setup]
AppId={{8F7D0F0A-2E79-4D1B-8A2E-47B85A4C4047}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\NeuronSpectrumGUI
DefaultGroupName=NeuronSpectrum GUI
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=NeuronSpectrumGUI_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать значок на рабочем столе"; GroupDescription: "Дополнительные значки:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NeuronSpectrum GUI"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\NeuronSpectrum GUI"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Запустить NeuronSpectrum GUI"; Flags: nowait postinstall skipifsilent
