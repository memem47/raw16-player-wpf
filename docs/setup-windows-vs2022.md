# Setup Guide (Windows + Visual Studio 2022)
対象：RawPlayerWpf.sln（RawPlayerWpf / ImageProcCli / CudaKernels）

## 0. 前提
- OS: Windows 10/11 64-bit
- GPU: NVIDIA（例：RTX 4060 Ti）
- CUDA: 12.6（※このソリューションはCUDAを使うため必須）
- Visual Studio: 2022 (Community/Professional/Enterprise いずれも可)
- 目標：x64 Debug/Release で WPFアプリが起動し、CPU/GPU処理が動作する状態を作る
## 1. 必要なインストール
### 1-1. Visual Studio 2022（必須）
Visual Studio Installer で次を入れます。
#### Workloads（推奨）
- .NET desktop development
- Desktop development with C++
- Game development with C++（CUDAとの相性で入れておくと楽な場合あり。無くてもOKなことが多い）
#### Individual components（重要）
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10/11 SDK
- C++/CLI support（C++/CLIプロジェクト用。これが無いと ImageProcCli が詰みます）
### 1-2. .NET Framework 4.8 Developer Pack（必須）
- WPF (.NET Framework 4.8) をビルドするのに必要です。
- “Runtime” ではなく “Developer Pack/Targeting Pack” が必要。
### 1-3. NVIDIA CUDA Toolkit 12.6（必須）
- CUDA 12.6 をインストール
- インストール後、nvcc が見えること（確認）
PowerShell:
```powershell
nvcc --version
```

Visual Studio に CUDA 統合（CUDA VS Integration）が入っている構成が望ましいです。
## 2. リポジトリの取得
```powershell
git clone <your-repo-url>
cd RawPlayerWpf
```
## 3. サードパーティ配置（onnxruntime）
### 3-1. フォルダ構成（例）
この構成で管理するのを推奨します：
```vbnet
third_party/
  onnxruntime/
    include/    (onnxruntime_cxx_api.h 等)
    lib/        (onnxruntime.lib)
    bin/        (onnxruntime.dll)
```
### 3-2. バージョン固定
- ヘッダ / lib / dll は必ず同一バージョンに揃える
- 例：ONNX Runtime 1.17.1 で統一
### 3-3. 実行時DLL配置
WPF実行フォルダに `onnxruntime.dll` が必要です。
プロジェクト（ImageProcCli）側で Post-build で xcopy している場合は、それが成功することを確認してください。
## 4. Visual Studio 2022 での設定（重要ポイント）
### 4-1. ソリューション構成を x64 に統一
Visual Studio 上部の構成で：
- `Debug` / `Release`
- Platform: x64

Any CPU / x86 が混じると、DLLロードやリンクで詰みます。
### 4-2. プロジェクトの依存関係（推奨）
ビルド順を安定させるため：
- `ImageProcCli` は `CudaKernels` に依存（GPU処理を呼ぶ）
- `RawPlayerWpf` は `ImageProcCli` を参照（WPFから呼ぶ）
Visual Studio:
- Solution Explorer → Solution 右クリック → Project Dependencies
  - `ImageProcCli` → `CudaKernels` をチェック
## 5. プロジェクト別の設定ポイント
### 5-1. CudaKernels（CUDA, static lib想定）
- 目的：CUDAカーネルをビルドして `.lib` を生成
- 生成物例：`x64\Debug\CudaKernels.lib`
#### 確認：
- CudaKernels の出力が `.exe` になっている場合は、構成が「Application」になっている可能性があります。
→ 通常は `Static Library (.lib)` が望ましいです（ただし現状の構成に合わせてOK）
### 5-2. ImageProcCli（C++/CLI DLL）
- 目的：WPFから呼べる .NETアセンブリ（ImageProcCli.dll）を生成
- ここが一番ハマりやすいポイント：
    - C++/CLI（/clr）
    - x64統一
    - 参照する native lib（CudaKernels.lib / onnxruntime.lib）
    - 実行時に必要な dll 配置（onnxruntime.dll, CUDA関連dll など）
#### onnxruntime の include/lib 設定
- C/C++ → Additional Include Directories
`third_party\onnxruntime\include`
- Linker → Additional Library Directories
`third_party\onnxruntime\lib`
- Linker → Input → Additional Dependencies
`onnxruntime.lib`
#### 実行フォルダへの dll コピー（例）
Post-Build Event:
```bat
xcopy /Y /D "$(SolutionDir)third_party\onnxruntime\bin\onnxruntime.dll" "$(TargetDir)"
```
### 5-3. RawPlayerWpf（WPF .NET Framework 4.8）
- 目的：RAW画像再生UI
- ImageProcCli を参照して、CPU/GPU/ONNX処理を切り替える
## 6. ビルド手順（成功ルート）
1. Visual Studio 2022 で RawPlayerWpf.sln を開く
2. 構成：Debug / x64 を選択
3. Build → Rebuild Solution
4. 起動プロジェクトが RawPlayerWpf になっていることを確認して F5
## 7. 実行時チェック
- RawPlayerWpf\bin\x64\Debug\（実行フォルダ）に以下があること
  - RawPlayerWpf.exe
  - ImageProcCli.dll
  - onnxruntime.dll（ONNX使用時）
- GPU denoise を使う場合
  - CUDA runtime DLL が見つかること（通常はCUDAインストールでPATHに入る）
## 8. よくあるエラーと対処（抜粋）
#### “onnxruntime.dll が見つからない / xcopy が失敗する”
- Post-build の xcopy パスが間違っている
- third_party\onnxruntime\bin\onnxruntime.dll が存在しない
- 対処：dllの実体パスを確認して、Post-build を修正
#### “defaultlib 'LIBCMT' は他のライブラリの使用と競合”
- ランタイムライブラリ（/MDd と /MTd）が混在
- 対処：プロジェクト間で Runtime Library を統一（まずは /MDd 推奨）
#### “x86 / AnyCPU が混じって dll ロード失敗”
- 対処：必ず x64 統一
## 9. （任意）ディレクトリを綺麗に保つ
- .gitignore で x64/Debug 等のビルド成果物を除外する
- すでに追跡された obj/pdb 等は git rm --cached で追跡解除してから commit

---

以上。